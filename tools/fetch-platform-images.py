#!/usr/bin/env python3
"""只取得固定公开平台镜像；不启动容器，不修改Docker配置。"""
import argparse
import json
import subprocess
import sys
sys.dont_write_bytecode = True
from repository_common import ROOT, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    rows = read_json(ROOT / 'locks/platform-images.json')['images']
    references = [row['RepoDigests'][0] for row in rows]
    print(json.dumps({'dry_run': not args.execute, 'public_digests': references,
                      'containers_started': False}, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    for reference in references:
        subprocess.run(['docker', 'pull', reference], check=True)


if __name__ == '__main__':
    main()
