#!/usr/bin/env python3
"""在固定旧运行镜像上构建纯Python修复层；只构建，不启动或切换模型。"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
from repository_common import ROOT, outside_repository


def inspect(reference):
    """通过Docker读取实际镜像身份，不能用某台机器的ID作公开下载地址。"""
    return json.loads(subprocess.check_output(['docker', 'image', 'inspect', reference]))[0]


def main():
    """先校验资源与标签，再在外部新目录构建；既有输出拒绝覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True)
    parser.add_argument('--base-image', required=True)
    parser.add_argument('--base-id', required=True, help='本次已确认的基础镜像实际本地ID')
    parser.add_argument('--image', required=True, help='不存在的新镜像标签')
    parser.add_argument('--cpus', type=int, required=True, choices=range(1, 5))
    parser.add_argument('--memory-gib', type=int, required=True, choices=range(2, 5))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    work = outside_repository(args.work, may_exist=False)
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', args.base_id):
        parser.error('基础身份必须为完整SHA256镜像ID')
    reference = r'[a-zA-Z0-9][a-zA-Z0-9._/:@-]*'
    if not re.fullmatch(reference, args.base_image) or not re.fullmatch(reference, args.image):
        parser.error('镜像引用格式不合法')
    if args.base_image == args.image:
        parser.error('基础和输出镜像不能同名')
    print(json.dumps({'dry_run': not args.execute, 'image': args.image,
                      'production_modified': False, 'gpu_models_started': False}), flush=True)
    if not args.execute:
        return
    base = inspect(args.base_image)
    if base['Id'] != args.base_id or base['Os'] != 'linux' or base['Architecture'] != 'amd64':
        raise RuntimeError('基础镜像身份、平台或架构不符')
    if subprocess.run(['docker', 'image', 'inspect', args.image], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('输出标签已存在，禁止覆盖')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    if available < (args.memory_gib + 10) * 1024 ** 3:
        raise RuntimeError('构建限额之外没有足够运行内存余量')
    work.mkdir(parents=True)
    shutil.copyfile(ROOT / 'sources/vllm/Dockerfile.lookup-progress', work / 'Dockerfile')
    shutil.copytree(ROOT / 'sources/vllm/lookup-progress', work / 'lookup-progress')
    # 继承的旧镜像须包含已有严格差异解析器；新补丁还会核对LMCache两份完整原文件。
    command = ['docker', 'build', '--pull=false', '--network=none', '--force-rm',
               '--memory=' + str(args.memory_gib) + 'g', '--cpu-period=100000',
               '--cpu-quota=' + str(args.cpus * 100000), '--build-arg',
               'BASE_IMAGE=' + args.base_image, '-t', args.image, str(work)]
    with (work / 'build-output.txt').open('w', encoding='utf-8') as output:
        subprocess.run(command, env=dict(os.environ, DOCKER_BUILDKIT='0'),
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    image = inspect(args.image)
    if image['RootFS']['Layers'][:len(base['RootFS']['Layers'])] != base['RootFS']['Layers']:
        raise RuntimeError('新镜像没有完整继承已确认的基础层')
    result = {'image': args.image, 'image_id': image['Id'], 'base_image_id': base['Id'],
              'cpu_checks_passed': True, 'gpu_inference_verified': False, 'production_modified': False}
    (work / 'build-result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
