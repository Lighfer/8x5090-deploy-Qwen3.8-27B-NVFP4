#!/usr/bin/env python3
"""检查公开工作树、全部Git对象及元数据；报告只给规则和位置，不回显敏感值。"""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode = True
from repository_files import repository_files
from repository_common import outside_repository

ROOT = Path(__file__).resolve().parent.parent
NETWORK = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)')
JWT = re.compile(r'eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}')
TOKEN = re.compile(r'gpustack_[0-9a-f]{16}_[0-9a-f]{32}')
HOST_PATH = re.compile(r'(?i)(?:[A-Z]:[\\/](?:Users|Projects)[\\/]|/(?:data|home|root|mnt)/)')


def findings(content, label, denied):
    """只允许标准回环/通配地址；它们用于容器内部接口，不是用户部署地址。"""
    result = []
    text = content.decode('utf-8', errors='replace')
    for rule, pattern in [('host-path', HOST_PATH), ('jwt', JWT), ('registration-token', TOKEN)]:
        for match in pattern.finditer(text):
            result.append({'file': label, 'rule': rule, 'line': text.count('\n', 0, match.start()) + 1})
    for match in NETWORK.finditer(text):
        try:
            address = ipaddress.ip_address(match.group())
        except ValueError:
            continue
        if not address.is_loopback and not address.is_unspecified:
            result.append({'file': label, 'rule': 'literal-host-address', 'line': text.count('\n', 0, match.start()) + 1})
    for value in denied:
        if value and value in text:
            result.append({'file': label, 'rule': 'private-marker', 'line': text[:text.index(value)].count('\n') + 1})
    return result


def json_environment_findings(value, label):
    """模板可以保留空身份或明确标记，实际编号、地址和运行快照不能混入JSON资料。"""
    result = []
    private_keys = {'container_id', 'StartedAt', 'production_root', 'source_repository', 'entry_url',
                    'server_container', 'worker_container', 'proxy_container', 'model_id', 'worker_id', 'cluster_id'}
    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key in private_keys and child not in (None, 0, '') and not (
                        isinstance(child, str) and re.fullmatch(r'@@[A-Z_]+@@', child)):
                    result.append({'file': label, 'rule': 'runtime-json-identity', 'field': key})
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return result


def git_objects(root):
    """batch-all-objects包含未被分支引用的旧对象，防止只扫HEAD漏掉历史和悬空数据。"""
    rows = subprocess.check_output(['git', '-C', str(root), 'cat-file', '--batch-all-objects',
                                    '--batch-check=%(objectname) %(objecttype)'], text=True).splitlines()
    selected = [(row.split()[0], row.split()[1]) for row in rows if row.split()[1] in {'blob', 'tree', 'commit', 'tag'}]
    for identity, kind in selected:
        content = subprocess.check_output(['git', '-C', str(root), 'cat-file', kind, identity])
        yield 'git-object:' + identity, content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deny-file', help='可选的仓库外私有标记JSON列表，不加入源码')
    parser.add_argument('--with-git', action='store_true')
    parser.add_argument('--report', help='可选的仓库外审查结果文件')
    args = parser.parse_args()
    denied = []
    if args.deny_file:
        denied = json.loads(outside_repository(args.deny_file).read_text(encoding='utf-8'))
        if not isinstance(denied, list) or any(not isinstance(v, str) for v in denied):
            raise ValueError('私有标记必须是字符串列表')
    problems, file_count, object_count = [], 0, 0
    for path in repository_files(ROOT):
        file_count += 1
        content = path.read_bytes()
        label = path.relative_to(ROOT).as_posix()
        path_problems = findings(label.encode('utf-8'), 'path-sha256:' + hashlib.sha256(label.encode()).hexdigest(), denied)
        problems.extend(path_problems)
        if path_problems:
            label = 'path-sha256:' + hashlib.sha256(label.encode()).hexdigest()
        problems.extend(findings(content, label, denied))
        # 所有源码应为可审阅的UTF-8文本；不能用二进制文件绕过文本规则。
        content.decode('utf-8')
        if path.suffix == '.json':
            problems.extend(json_environment_findings(json.loads(content), label))
        if path.suffix == '.md' and re.search('dev' + 'env', content.decode('utf-8'), re.I):
            problems.append({'file': label, 'rule': 'obsolete-document-name'})
        if path.name.endswith('.example.json'):
            site = json.loads(content)
            def check(value, key=''):
                if isinstance(value, dict):
                    for name, child in value.items():
                        check(child, name)
                elif key not in {'format', 'model_id', 'worker_id', 'cluster_id'} and value is not None:
                    raise ValueError('环境示例不能预填部署身份或资源值')
            check(site)
    for name in ('MANIFEST.json', 'SHA256SUMS'):
        path = ROOT / name
        if path.exists():
            if path.is_symlink():
                raise ValueError('完整性清单不能是符号链接')
            problems.extend(findings(path.read_bytes(), name, denied))
    if args.with_git:
        if not (ROOT / '.git').is_dir():
            raise ValueError('需要真实独立Git仓库才能审查历史')
        for label, content in git_objects(ROOT):
            object_count += 1
            problems.extend(findings(content, label, denied))
        # 配置、引用日志、钩子等不是普通Git blob，也必须单独检查。
        for path in (ROOT / '.git').rglob('*'):
            if path.is_symlink():
                raise ValueError('Git元数据不能是符号链接')
            if path.is_file() and 'objects' not in path.relative_to(ROOT / '.git').parts:
                problems.extend(findings(path.read_bytes(), 'git-metadata:' + path.relative_to(ROOT / '.git').as_posix(), denied))
    result = {'passed': not problems, 'files_scanned': file_count,
              'git_objects_scanned': object_count, 'findings': problems,
              'scope': '静态模式、私有标记和全部Git对象；还需人工审查和功能验证'}
    if args.report:
        outside_repository(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
