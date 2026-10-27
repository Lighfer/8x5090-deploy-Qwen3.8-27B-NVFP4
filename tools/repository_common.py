"""共享路径与散列检查；用户环境、构建输出和部署输出必须在仓库之外。"""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / 'templates/runtime'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def outside_repository(path, *, may_exist=True):
    """解析真实路径后判断，不允许用符号链接把外部输出重新指向源码目录。"""
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError('必须提供仓库之外的绝对路径，且不能是符号链接')
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or resolved.is_relative_to(ROOT) or ROOT.is_relative_to(resolved):
        raise ValueError('环境或输出路径不能位于源码仓库内，也不能是仓库的父目录')
    if not may_exist and path.exists():
        raise ValueError('目标已经存在，拒绝覆盖')
    return resolved


def safe_path(root, name):
    """资源清单只允许普通相对文件，拒绝路径穿越和符号链接。"""
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('清单路径不是安全的相对路径')
    target = Path(root) / relative
    anchor = Path(root).resolve()
    # 即使链接仍指向同一安装目录，也拒绝间接写入，避免受管文件之间互相覆盖。
    if any(part.is_symlink() for part in (target, *target.parents) if part != Path(part.anchor)):
        raise ValueError('清单路径不能经过符号链接')
    if not target.resolve().is_relative_to(anchor):
        raise ValueError('清单文件越过根目录')
    return target


def output_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
