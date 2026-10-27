#!/usr/bin/env python3
"""只打包已提交且通过审查的公开仓库；重新建立不含本机配置与引用日志的Git元数据。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
from repository_common import ROOT, outside_repository


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], stderr=subprocess.STDOUT)


def checked(command):
    """子检查失败直接停止归档，不能只打印警告后继续分发。"""
    subprocess.run([sys.executable, '-B', *map(str, command)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='仓库外尚不存在的ZIP路径')
    parser.add_argument('--deny-file', help='可选的仓库外私有标记列表')
    args = parser.parse_args()
    output = outside_repository(args.output, may_exist=False)
    if not (ROOT / '.git').is_dir() or Path(git(ROOT, 'rev-parse', '--show-toplevel').decode().strip()).resolve() != ROOT:
        raise ValueError('必须是独立Git仓库，不能使用父目录仓库')
    if git(ROOT, 'status', '--porcelain', '--untracked-files=all'):
        raise ValueError('源码必须全部提交，且不能有未登记文件')
    checked([ROOT / 'tools/verify-integrity.py'])
    extra = ['--deny-file', outside_repository(args.deny_file)] if args.deny_file else []
    checked([ROOT / 'tools/audit-public.py', '--with-git', *extra])
    git(ROOT, 'fsck', '--full', '--strict')
    with tempfile.TemporaryDirectory(prefix='public-source-package-') as directory:
        staged = Path(directory) / 'gpustack-deployment-guide'
        # 不复制原索引、配置、钩子、远端或引用日志；只克隆已审查的可达公开历史。
        subprocess.run(['git', 'clone', '--quiet', '--no-local', '--no-checkout', '--template=',
                        str(ROOT), str(staged)], check=True)
        git(staged, 'remote', 'remove', 'origin')
        shutil.rmtree(staged / '.git/logs', ignore_errors=True)
        (staged / '.git/config').write_text('[core]\n\trepositoryformatversion = 0\n\tfilemode = false\n\tbare = false\n\tlogallrefupdates = false\n', encoding='utf-8')
        git(staged, 'read-tree', 'HEAD')
        git(staged, 'checkout-index', '--all')
        checked([staged / 'tools/verify-integrity.py'])
        checked([staged / 'tools/audit-public.py', '--with-git', *extra])
        git(staged, 'fsck', '--full', '--strict')
        if git(staged, 'status', '--porcelain', '--untracked-files=all'):
            raise ValueError('归档副本与公开提交不一致')
        output.parent.mkdir(parents=True, exist_ok=True)
        # 使用排他创建防止覆盖旧包；任何失败都删除本次未完成输出。
        try:
            with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                for path in sorted(staged.rglob('*')):
                    if path.is_symlink():
                        raise ValueError('归档不能含符号链接')
                    if path.is_file():
                        archive.write(path, 'gpustack-deployment-guide/' + path.relative_to(staged).as_posix())
        except Exception:
            output.unlink(missing_ok=True)
            raise
    print(json.dumps({'passed': True, 'archive': str(output), 'includes_public_git': True,
                      'external_resources_included': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
