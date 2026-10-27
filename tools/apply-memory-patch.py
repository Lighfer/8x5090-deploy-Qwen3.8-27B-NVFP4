#!/usr/bin/env python3
"""对固定GPUStack原始模块重放LXC内存差异；未知源拒绝，默认不写文件。"""
import argparse
import hashlib
from pathlib import Path
import re
import sys
# 在导入公共模块前禁用字节码缓存，保持冻结资料运行后仍不新增文件。
sys.dont_write_bytecode = True
from repository_common import ROOT, RELEASE, digest, outside_repository

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--execute', action='store_true')
args = parser.parse_args()
source, target = Path(args.input), outside_repository(args.output, may_exist=False)
assert digest(source) == '3d8fd8bda46d9b18892e8cf3e45d5298faf5441d525bd56002a2dfc682cc758b', '原始模块版本不符'
assert source.resolve() != target.resolve() and not target.exists(), '不得覆盖原文件或已有输出'
original = source.read_text(encoding='utf-8').splitlines(keepends=True)
difference = (ROOT / 'sources/worker-memory/fastfetch-lxc.patch').read_text(encoding='utf-8').splitlines(keepends=True)
result, cursor, index = [], 0, 2
while index < len(difference):
    header = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n?', difference[index])
    assert header, '不支持的差异头'
    start = int(header[1]) - 1
    old_count, new_count = int(header[2] or 1), int(header[4] or 1)
    assert start >= cursor
    result.extend(original[cursor:start])
    cursor, index, old_used, new_used = start, index + 1, 0, 0
    while index < len(difference) and not difference[index].startswith('@@ '):
        line = difference[index]
        assert line and line[0] in ' +-'
        if line[0] in ' -':
            assert cursor < len(original) and original[cursor] == line[1:], '差异上下文不匹配'
            cursor += 1
            old_used += 1
        if line[0] in ' +':
            result.append(line[1:])
            new_used += 1
        index += 1
    assert (old_used, new_used) == (old_count, new_count)
result.extend(original[cursor:])
content = ''.join(result).encode()
checksum = hashlib.sha256(content).hexdigest()
assert checksum == digest(RELEASE / 'worker/fastfetch.py') == '57895b533b7c8dfa98a7713075475841bca90364ccd731aa12b3a64050a81933'
compile(content, str(target), 'exec')
if args.execute:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
print({'dry_run': not args.execute, 'source_accepted': True, 'patched_sha256': checksum,
       'runtime_file_replaced': False, 'output': str(target)})
