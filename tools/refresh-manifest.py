#!/usr/bin/env python3
"""为已审查的公开源码登记散列；默认只报告差异，不自动认可内容安全。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from repository_files import repository_files

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    rows = []
    for path in sorted(repository_files(ROOT)):
        content = path.read_bytes()
        content.decode('utf-8')
        rows.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': len(content),
                     'sha256': hashlib.sha256(content).hexdigest()})
    previous = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8')) if (ROOT / 'MANIFEST.json').exists() else {'files': []}
    old, new = {r['path']: r for r in previous['files']}, {r['path']: r for r in rows}
    print(json.dumps({'dry_run': not args.execute, 'added': sorted(new.keys() - old.keys()),
                      'removed': sorted(old.keys() - new.keys()),
                      'changed': [name for name in sorted(new.keys() & old.keys()) if new[name] != old[name]],
                      'files': len(rows)}, ensure_ascii=False, indent=2))
    if args.execute:
        (ROOT / 'MANIFEST.json').write_text(json.dumps({'format': 1, 'files': rows}, ensure_ascii=False, indent=2) + '\n')
        (ROOT / 'SHA256SUMS').write_text(''.join(r['sha256'] + '  ' + r['path'] + '\n' for r in rows))


if __name__ == '__main__':
    main()
