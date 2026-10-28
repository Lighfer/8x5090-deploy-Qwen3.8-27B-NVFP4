#!/usr/bin/env python3
"""使用当前镜像的PyTorch工具链重建同版本LMCache CUDA扩展，不变更Python包和依赖。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import tarfile

import torch

ARCHIVE_SHA256 = 'd6951e60ce8818a76aa42d0866a4ba85f1c95cd919355fff30ff24dcc3243f1e'
SOURCE_URL = 'https://files.pythonhosted.org/packages/84/25/3ccd46844c94e08897aa8d72716a538c07e4583410d84dbc6cd12cb60dc1/lmcache-0.5.6.tar.gz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-only', action='store_true', help='仅用于已有已核验CUDA扩展的增量派生构建')
    args = parser.parse_args()
    root = Path('/opt/gpustack-v032/lmcache-cuda')
    archive = root / 'lmcache-0.5.6.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA256
    assert importlib.metadata.version('lmcache') == '0.5.6'
    assert torch.__version__ == '2.14.0+cu130'
    assert torch._C._GLIBCXX_USE_CXX11_ABI
    assert not torch.cuda.is_initialized()
    source = root / 'source'
    source.mkdir()
    with tarfile.open(archive) as package:
        # 所有输入已固定散列，仍拒绝链接和目录跳转，解包只落在本次容器构建目录。
        members = package.getmembers()
        for member in members:
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                raise RuntimeError('源码包存在不安全的归档成员')
        package.extractall(source, members=members, filter='data')
    project = source / 'lmcache-0.5.6'
    os.chdir(project)
    sys.path.insert(0, str(project))
    from setuptools import Distribution
    from setup_extensions.build_profiles.cuda import CudaProfile
    from setup_extensions.common_cpp import build_common_cpp
    # CUDA与公共类型模块共享PageBufferShapeDesc等C++对象，必须来自同一套
    # Torch/PyBind工具链。不触发pip依赖解析、重装Torch或无关存储后端构建。
    profile = CudaProfile()
    common, commands = build_common_cpp(profile)
    common = [extension for extension in common if extension.name == 'lmcache.lmcache_native']
    assert len(common) == 1
    if args.native_only:
        previous = json.loads((root / 'abi.json').read_text())
        assert previous['torch'] == torch.__version__ and previous['source_sha256'] == ARCHIVE_SHA256
        extensions = common
    else:
        extensions, commands = profile.build()
        assert [extension.name for extension in extensions] == ['lmcache.cuda_ops']
        extensions.extend(common)
    # 当前运行镜像把cuSPARSE/cuSOLVER头文件随NVIDIA Python包安装，未放到
    # CUDA_HOME/include。使用同一cu13目录，避免下载另一套SDK或改变库版本。
    cuda_headers = Path(sysconfig.get_paths()['purelib']) / 'nvidia/cu13/include'
    assert all((cuda_headers / name).is_file() for name in ('cusparse.h', 'cusolverDn.h'))
    for extension in extensions:
        extension.include_dirs.append(str(cuda_headers))
    distribution = Distribution({'name': 'lmcache-cuda-abi-rebuild', 'ext_modules': extensions, 'cmdclass': commands})
    command = distribution.get_command_obj('build_ext')
    command.build_lib = str(root / 'compiled')
    command.build_temp = str(root / 'objects')
    command.inplace = False
    distribution.run_command('build_ext')
    outputs = list((root / 'compiled/lmcache').glob('*.so'))
    assert len(outputs) == (1 if args.native_only else 2)
    installed = Path(sysconfig.get_paths()['purelib']) / 'lmcache'
    for output in outputs:
        shutil.copyfile(output, installed / output.name)
    target = next(installed.glob('cuda_ops*.so'))
    common_target = next(installed.glob('lmcache_native*.so'))
    os.chdir(root)
    sys.path.remove(str(project))
    # 只读导入能够发现缺失符号；不能仅根据.so文件存在宣称CUDA扩展已可用。
    import lmcache.cuda_ops as native
    assert hasattr(native, 'multi_layer_block_kv_transfer')
    from lmcache.v1.platform.devices.cuda.device_ops import CudaDeviceOps
    operations = CudaDeviceOps()
    operations.ensure_native()
    assert not torch.cuda.is_initialized()
    report = {'lmcache': '0.5.6', 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'architecture': os.environ['TORCH_CUDA_ARCH_LIST'], 'source_url': SOURCE_URL,
              'source_sha256': ARCHIVE_SHA256, 'extension_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'common_extension_sha256': hashlib.sha256(common_target.read_bytes()).hexdigest(),
              'paired_native_build': True,
              'native_import_passed': True, 'cuda_initialized_during_build': False}
    (root / 'abi.json').write_text(json.dumps(report, indent=2) + '\n')
    # 仅清理本次容器内固定构建目录，原有包、旧镜像及生产缓存均不在该范围。
    for path in (source, root / 'objects', root / 'compiled'):
        assert path.parent == root
        shutil.rmtree(path)
    archive.unlink()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
