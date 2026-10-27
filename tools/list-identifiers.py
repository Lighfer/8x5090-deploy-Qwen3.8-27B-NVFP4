#!/usr/bin/env python3
"""通过已生成部署的只读接口列出真实编号；不在源码中保留另一台机器的编号。"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from repository_common import outside_repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True)
    args = parser.parse_args()
    root = outside_repository(args.deployment)
    spec = importlib.util.spec_from_file_location('private_deployment_api', root / 'gpustack-local-api.py')
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    result = {}
    for resource in ('clusters', 'workers', 'models'):
        rows = api.call('GET', '/v2/' + resource).get('items', [])
        result[resource] = [{key: row.get(key) for key in ('id', 'name')} for row in rows]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
