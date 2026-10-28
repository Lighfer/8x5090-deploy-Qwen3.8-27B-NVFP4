"""严格重放固定上游补丁；未知输入拒绝修改，所有输出核验后再写入。

不依赖Git或网络，不导入CUDA。清单登记每一步前后完整文件散列；统一差异只能
匹配完整上下文，不允许忽略空白或内容模糊匹配。末态已匹配时可重复执行。
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import os

HERE = Path(__file__).resolve().parent


def digest(data):
    """散列针对原始字节，补丁和文件的换行也属于身份。"""
    return hashlib.sha256(data).hexdigest()


def apply_difference(contents, patch):
    """重放同路径文件的统一差异，返回内存中的新内容，不接触文件系统。

    上游行号可能因未引入的无关提交而移动。行号不匹配时仅接受唯一的完整上下文，
    且调用方已核对整份输入散列；上下文不完整、重复或计数错误都立即停止。
    """
    result = dict(contents)
    lines = patch.splitlines(keepends=True)
    index = 0
    name = None
    delta = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith('--- a/'):
            name = line[6:].strip()
            index += 1
            if index >= len(lines) or lines[index].strip() != '+++ b/' + name or name not in result:
                raise RuntimeError('补丁文件路径不符合清单：' + str(name))
            delta = 0
        elif line.startswith('@@ '):
            match = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if not match or name is None:
                raise RuntimeError('补丁块头无效')
            old_start, old_count, _, new_count = match.groups()
            old_count, new_count = int(old_count or 1), int(new_count or 1)
            old, new = [], []
            index += 1
            while index < len(lines) and not lines[index].startswith(('@@ ', 'diff --git ', '--- a/')):
                item = lines[index]
                if item.startswith(' '):
                    old.append(item[1:]); new.append(item[1:])
                elif item.startswith('-'):
                    old.append(item[1:])
                elif item.startswith('+'):
                    new.append(item[1:])
                else:
                    raise RuntimeError('补丁存在不支持的行或缺失换行')
                index += 1
            if len(old) != old_count or len(new) != new_count:
                raise RuntimeError('补丁块计数不完整：' + name)
            current = result[name].decode('utf-8').splitlines(keepends=True)
            wanted = int(old_start) - 1 + delta
            if wanted < 0 or current[wanted:wanted + len(old)] != old:
                positions = [p for p in range(len(current) - len(old) + 1) if current[p:p + len(old)] == old]
                if len(positions) != 1:
                    raise RuntimeError('补丁上下文不是唯一完整匹配：' + name)
                wanted = positions[0]
            current[wanted:wanted + len(old)] = new
            result[name] = ''.join(current).encode('utf-8')
            delta += len(new) - len(old)
            continue
        index += 1
    return result


def prepare(root, manifest):
    """先完成全部身份与语法验证；任一失败都不会留下部分修改。"""
    original = {}
    for name in manifest['original_files']:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('vllm', 'flashinfer'):
            raise RuntimeError('目标路径不允许：' + name)
        target = root / name
        if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
            raise RuntimeError('目标不能通过链接离开软件目录')
        original[name] = target.read_bytes()
    if all(digest(original[name]) == value for name, value in manifest['final_files'].items()):
        return original, False
    if any(digest(original[name]) != value for name, value in manifest['original_files'].items()):
        raise RuntimeError('软件源码不是登记的基线或完整末态，拒绝重放')
    current = original
    for entry in manifest['patches']:
        data = (HERE / entry['file']).read_bytes()
        if digest(data) != entry['sha256']:
            raise RuntimeError('补丁自身散列不符')
        if any(digest(current[name]) != value for name, value in entry['before'].items()):
            raise RuntimeError('中间输入不符合清单')
        current = apply_difference(current, data.decode('utf-8'))
        if any(digest(current[name]) != value for name, value in entry['after'].items()):
            raise RuntimeError('重放结果不符合独立Git重放得到的散列')
    if any(digest(current[name]) != value for name, value in manifest['final_files'].items()):
        raise RuntimeError('最终源码散列不符')
    for name, data in current.items():
        ast.parse(data, filename=name)
    return current, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='包含vllm和flashinfer的软件包目录')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = args.root
    if root is None:
        spec = importlib.util.find_spec('vllm')
        root = Path(spec.origin).parent.parent
    manifest = json.loads((HERE / 'manifest.json').read_text())
    contents, changed = prepare(root, manifest)
    if args.execute and changed:
        for name, data in contents.items():
            target = root / name
            mode = target.stat().st_mode & 0o777
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            temporary.chmod(mode)
            temporary.replace(target)
    print(json.dumps({'passed': True, 'release': manifest['release'], 'files': len(contents),
                      'patches': len(manifest['patches']), 'already_applied': not changed,
                      'written': bool(args.execute and changed)}))


if __name__ == '__main__':
    main()
