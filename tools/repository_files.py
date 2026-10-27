"""公开源码文件边界；私有配置和可再生产物即使被Git忽略也不能留在仓库中。"""
import os
from pathlib import Path

MANIFEST_FILES = {'MANIFEST.json', 'SHA256SUMS'}


def repository_files(root):
    root = Path(root).resolve(strict=True)
    for directory, folders, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(folders):
            path = base / name
            if path.is_symlink():
                raise ValueError('目录不能是符号链接：' + str(path.relative_to(root)))
            if base == root and name == '.git':
                folders.remove(name)
            elif name in {'__pycache__', 'node_modules', 'target', '_work', 'baks', 'logs'}:
                raise ValueError('公开树含运行或可再生目录：' + str(path.relative_to(root)))
        for name in files:
            path = base / name
            relative = path.relative_to(root)
            if base == root and name in MANIFEST_FILES:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('文件路径不安全：' + str(relative))
            if name == '.env' or name.endswith(('.pyc', '.zip', '.tar.gz', '.whl', '.deb', '.safetensors', '.gguf', '.sqlite', '.db', '.log')):
                raise ValueError('公开树含私有文件或外部产物：' + str(relative))
            if name.startswith('.') and name not in {'.gitignore', '.gitattributes'}:
                raise ValueError('公开树含未允许的隐藏文件：' + str(relative))
            yield path
