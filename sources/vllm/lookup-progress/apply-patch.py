#!/usr/bin/env python3
"""严格应用缓存结果推进差异；未知、混合或损坏输入不会被修改。"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def digest(data):
    """文件身份使用原始字节，包括换行和补丁上下文。"""
    return hashlib.sha256(data).hexdigest()


def package_root():
    """只查找已安装文件，不导入LMCache、Torch或CUDA以定位软件目录。"""
    for entry in sys.path:
        root = Path(entry)
        if (root / 'lmcache/integration/vllm/vllm_multi_process_adapter.py').is_file():
            return root
    raise RuntimeError('未找到已安装的LMCache源码')


def prepare(root, manifest):
    """先核对两份完整输入和差异，再在内存验证全部输出与Python语法。"""
    allowed = {
        'lmcache/integration/vllm/vllm_multi_process_adapter.py',
        'lmcache/integration/vllm/lmcache_mp_connector.py',
    }
    if set(manifest['original_files']) != allowed or set(manifest['final_files']) != allowed:
        raise RuntimeError('补丁目标不符合固定范围')
    original = {}
    for name in allowed:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError('源码路径不能经过符号链接离开软件目录')
        original[name] = path.read_bytes()
    if all(digest(original[name]) == expected for name, expected in manifest['final_files'].items()):
        return original, False
    if any(digest(original[name]) != expected for name, expected in manifest['original_files'].items()):
        raise RuntimeError('LMCache源码不是固定基线或完整末态，拒绝应用')
    patch = HERE / manifest['patch']
    if not patch.resolve().is_relative_to(HERE) or patch.is_symlink():
        raise RuntimeError('补丁路径越界')
    data = patch.read_bytes()
    if digest(data) != manifest['patch_sha256']:
        raise RuntimeError('补丁自身散列不符')
    # 复用同一构建配方已有的严格统一差异解析器，不引入系统patch或模糊匹配。
    apply_difference = runpy.run_path(str(HERE.parent / 'optimizations/apply-patches.py'))['apply_difference']
    current = apply_difference(original, data.decode('utf-8'))
    for name, content in current.items():
        if digest(content) != manifest['final_files'][name]:
            raise RuntimeError('输出源码散列不符：' + name)
        ast.parse(content, filename=name)
    return current, True


def main():
    """默认只核验；显式执行时采用同目录临时文件替换，不生成运行时字节码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = args.root or package_root()
    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    current, changed = prepare(root, manifest)
    if args.execute and changed:
        for name, content in current.items():
            target = root / name
            mode = target.stat().st_mode & 0o777
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(mode)
            temporary.replace(target)
    print(json.dumps({'passed': True, 'release': manifest['release'], 'files': len(current),
                      'already_applied': not changed, 'written': bool(changed and args.execute)}))


if __name__ == '__main__':
    main()
