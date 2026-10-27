#!/usr/bin/env python3
"""在固定 FlashInfer 输入上重放官方修复，禁止把补丁套到未知版本。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def sha256(content):
    """散列按原始字节计算，不通过换行规范化掩盖输入差异。"""
    return hashlib.sha256(content).hexdigest()


def apply_diff(original, patch):
    """逐块严格核对上下文；既不模糊匹配，也不接受缺少删除行的输入。"""
    source = original.splitlines(keepends=True)
    lines = patch.splitlines(keepends=True)
    result, cursor, index = [], 0, 0
    while index < len(lines):
        header = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', lines[index])
        if not header:
            index += 1
            continue
        begin = int(header[1]) - 1
        if begin < cursor:
            raise RuntimeError('补丁块次序重叠')
        result.extend(source[cursor:begin])
        cursor = begin
        removed = added = 0
        index += 1
        while index < len(lines) and not lines[index].startswith('@@ '):
            line = lines[index]
            if line.startswith((' ', '-')):
                if cursor >= len(source) or source[cursor] != line[1:]:
                    raise RuntimeError('官方补丁上下文不符，原始行：' + str(cursor + 1))
                cursor += 1
                removed += 1
            if line.startswith((' ', '+')):
                result.append(line[1:])
                added += 1
            if not line.startswith((' ', '-', '+')):
                raise RuntimeError('不支持的补丁行')
            index += 1
        if removed != int(header[2] or '1') or added != int(header[4] or '1'):
            raise RuntimeError('补丁块行数与头部不符')
    result.extend(source[cursor:])
    return ''.join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/usr/local/lib/python3.12/dist-packages/flashinfer/page.py'))
    parser.add_argument('--kernel', type=Path, help='可选的固定 GPU 内核头文件路径')
    parser.add_argument('--output', type=Path, help='默认在候选镜像内原位写入，不修改主机或运行容器')
    parser.add_argument('--execute', action='store_true', help='显式写入；默认只核查并报告')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    lock = json.loads((directory / 'inputs.json').read_text(encoding='utf-8'))
    original = args.input.read_bytes()
    if sha256(original) != lock['page_original_sha256']:
        raise RuntimeError('FlashInfer 原始源码身份不符；可能已包含修复，必须重新审查')
    kernel = args.kernel or args.input.parent / 'data/include/flashinfer/page.cuh'
    kernel_bytes = kernel.read_bytes()
    if sha256(kernel_bytes) != lock['kernel_sha256']:
        raise RuntimeError('FlashInfer 内核身份不符，不能假设它仍有比例因子检查')
    # 此 Python 优化依赖既有内核在写缓存前验证 CUDA 比例因子；不修改或编译内核。
    start = kernel_bytes.index(b'NVFP4QuantizeAppendPagedKVCacheWithSlotMappingKernel')
    guard = kernel_bytes[start:kernel_bytes.index(b'const IdType slot', start)]
    if not all(token in guard for token in (b'nvfp4_append_is_positive_finite_scale(k_scale)',
                                           b'nvfp4_append_is_positive_finite_scale(v_scale)', b'trap;')):
        raise RuntimeError('内核缺少必要的比例因子保护')
    patch = (directory / 'upstream.patch').read_bytes()
    if sha256(patch) != lock['patch_sha256']:
        raise RuntimeError('保存的官方补丁散列不符')
    patched = apply_diff(original.decode('utf-8'), patch.decode('utf-8')).encode('utf-8')
    if sha256(patched) != lock['page_patched_sha256']:
        raise RuntimeError('重放结果与登记的官方修复结果不一致')
    ast.parse(patched.decode('utf-8'))
    output = args.output or args.input
    if args.execute:
        output.write_bytes(patched)
    print(json.dumps({'applied': args.execute, 'original_sha256': sha256(original),
                      'patched_sha256': sha256(patched), 'kernel_unchanged': True,
                      'upstream_commit': lock['upstream_commit']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
