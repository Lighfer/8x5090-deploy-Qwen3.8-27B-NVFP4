#!/usr/bin/env python3
"""比较真实文件集合和原始字节；额外私有文件也会失败，不能仅检查已登记文件。"""
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from repository_files import repository_files

ROOT = Path(__file__).resolve().parent.parent


def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    expected = {row['path']: row for row in manifest['files']}
    actual = {p.relative_to(ROOT).as_posix(): p for p in repository_files(ROOT)}
    if set(expected) != set(actual):
        raise ValueError('真实文件集合与清单不同')
    for name, path in actual.items():
        content = path.read_bytes()
        if len(content) != expected[name]['bytes'] or hashlib.sha256(content).hexdigest() != expected[name]['sha256']:
            raise ValueError('文件内容与清单不符：' + name)
    sums = ''.join(row['sha256'] + '  ' + row['path'] + '\n' for row in manifest['files'])
    if (ROOT / 'SHA256SUMS').read_text(encoding='utf-8') != sums:
        raise ValueError('两份完整性清单不一致')
    print(json.dumps({'passed': True, 'files': len(actual), 'full_hash_verification': True}))


if __name__ == '__main__':
    main()
