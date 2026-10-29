#!/usr/bin/env python3
"""按逐文件固定 revision 获取当前模型并核对 SHA256；权重保存到目标模型目录，不放入源码仓库。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent


def verify(path, row):
    """存在、大小及完整内容散列均一致才视为同一模型输入，不能仅依据文件名或 revision。"""
    if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes']:
        raise RuntimeError('模型文件缺失、大小不符或为链接：' + row['path'])
    with path.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != row['sha256']:
            raise RuntimeError('模型文件 SHA256 不符：' + row['path'])


def main():
    """默认显示固定来源；执行时允许断点续传自己的 .download 文件，拒绝覆盖不同的正式权重。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, help='仓库外的完整模型目录；不同精度使用独立目录')
    parser.add_argument('--precision', choices=('nvfp4', 'fp8'), default='nvfp4', help='选择固定权重清单；默认保留原NVFP4行为')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--verify-only', action='store_true', help='仅完整校验已有模型，不访问网络或写文件')
    args = parser.parse_args()
    target = Path(args.target)
    if not target.is_absolute() or target.is_symlink() or target.resolve().is_relative_to(ROOT) or target.resolve() == Path('/'):
        raise RuntimeError('模型目标必须为仓库之外的绝对路径，且不能为根目录或符号链接')
    lock_name = 'model-downloads-fp8.json' if args.precision == 'fp8' else 'model-downloads.json'
    lock = json.loads((ROOT / 'locks' / lock_name).read_text(encoding='utf-8'))
    print(json.dumps({'dry_run': not args.execute and not args.verify_only, 'verify_only': args.verify_only,
                      'target': str(target), 'precision': args.precision, 'files': len(lock['files']), 'bytes': lock['total_bytes'],
                      'fixed_revision': True, 'production_containers_modified': False}, ensure_ascii=False), flush=True)
    if not args.execute and not args.verify_only:
        return
    if args.execute and not args.verify_only:
        target.mkdir(parents=True, exist_ok=True)
    network_used = False
    downloaded = 0
    for row in lock['files']:
        # 两种格式均固定发布来源和完整revision，不接受锁文件指向任意第三方下载地址。
        origin = ('https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/' if args.precision == 'fp8'
                  else 'https://modelscope.cn/models/nv-community/Qwen3.8-27B-NVFP4/resolve/')
        if Path(row['path']).name != row['path'] or not row['url'].startswith(origin + row['revision'] + '/'):
            raise RuntimeError('模型锁文件的路径或来源不符合固定资源')
        destination = target / row['path']
        if destination.exists() or args.verify_only:
            verify(destination, row)
            continue
        partial = destination.with_suffix(destination.suffix + '.download')
        if partial.is_symlink() or partial.exists() and partial.stat().st_size > row['bytes']:
            raise RuntimeError('断点文件不安全或大小超过目标：' + row['path'])
        if not partial.exists() or partial.stat().st_size < row['bytes']:
            network_used = True
            subprocess.run(['curl', '--proto', '=https', '--proto-redir', '=https', '--fail', '--location',
                            '--connect-timeout', '15', '--max-time', '7200', '--retry', '2', '--continue-at', '-',
                            '--output', str(partial), row['url']], check=True, timeout=21800)
        verify(partial, row)
        partial.replace(destination)
        downloaded += 1
    print(json.dumps({'passed': True, 'files': len(lock['files']), 'target': str(target),
                      'network_used': network_used, 'downloaded_files': downloaded}, ensure_ascii=False))


if __name__ == '__main__':
    main()
