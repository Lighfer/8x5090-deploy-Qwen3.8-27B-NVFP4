#!/bin/sh
# 将工作目录固定为脚本所在目录，允许从任意目录调用；参数直接传给执行器，不拼接为命令文本。
# 前台仅跟踪进度，真正的切换由 systemd 执行，因此关闭 SSH 不会终止模型切换或回滚。
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec /usr/bin/python3 "$TASK_ROOT/switch-pd.py" "$@"
