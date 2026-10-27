#!/usr/bin/env python3
"""对固定 GPUStack Runtime 0.2.6 添加显式选择的 Worker PID 命名空间配置。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import typing
from types import SimpleNamespace

ORIGINAL_SHA256 = 'a7711fa5881477a52f1a92a25dc74af5ff29e18633fdb876d9c8f21303e4303e'
HELPER = '''def _configure_nixl_worker_pid(create_options: dict[str, Any], arguments: list[str], worker_name: str):
    """只让显式选择的 KV 传输容器共享 Worker 的 PID 空间，不进入宿主进程空间。"""
    selected = create_options.get("environment", {}).get("NIXL_SHARE_WORKER_PID")
    if selected in (None, "0"):
        return
    if selected != "1":
        raise OperationError("NIXL_SHARE_WORKER_PID must be 0 or 1")
    if not any(arg == "--kv-transfer-config" or arg.startswith("--kv-transfer-config=") for arg in arguments):
        return
    # P、D 是不同工作负载；原 pid_shared 仅共享各自 pause 容器，无法提供跨角色 CUDA IPC。
    # 使用现有 Worker 名称作为共同命名空间，保留各自的网络、IPC、GPU、文件系统与启停归属。
    if not worker_name:
        raise OperationError("GPUStack mirrored Worker name is required for shared NIXL PID namespace")
    create_options["pid_mode"] = "container:" + worker_name


'''
CALL = '''            # 显式模型开关只改变 KV 传输角色的 PID 空间；Router 与缓存容器保持原部署方式。
            _configure_nixl_worker_pid(
                create_options,
                (c.execution.command or []) + (c.execution.args or []) if c.execution else [],
                envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME,
            )

'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--verification', required=True)
    args = parser.parse_args()
    original = Path(args.input).read_bytes()
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('Runtime source differs from the reviewed 0.2.6 file')
    text = original.decode()
    first, second = '_NAME = "docker"\n', '            # Parameterize resources.\n'
    if text.count(first) != 1 or text.count(second) != 1:
        raise ValueError('Runtime patch anchors are not unique')
    text = text.replace(first, HELPER + first, 1).replace(second, CALL + second, 1)
    compile(text, args.output, 'exec')
    # 提取同一实际函数核对默认不变、Router隔离及缺失Worker时明确失败，避免测试一份另写的实现。
    parsed = ast.parse(text)
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
        and node.name == '_configure_nixl_worker_pid')
    namespace = {'Any': typing.Any, 'OperationError': RuntimeError}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'actual-runtime-function', 'exec'), namespace)
    configure = namespace[function.name]
    rows = []
    for name, flag, arguments, expected in (
        ('未选择的模型保持原样', None, ['--kv-transfer-config={}'], None),
        ('显式关闭保持原样', '0', ['--kv-transfer-config={}'], None),
        ('等号形式的角色参数', '1', ['--kv-transfer-config={}'], 'container:worker-reference'),
        ('独立形式的角色参数', '1', ['--kv-transfer-config', '{}'], 'container:worker-reference'),
        ('Router不共享PID', '1', ['--kv-connector', 'nixl'], None),
    ):
        environment = {} if flag is None else {'NIXL_SHARE_WORKER_PID': flag}
        options = {'environment': environment, 'network_mode': 'container:unchanged', 'ipc_mode': 'container:unchanged'}
        before = dict(options)
        configure(options, arguments, 'worker-reference')
        assert options == {**before, **({'pid_mode': expected} if expected else {})}
        rows.append({'name': name, 'passed': True})
    try:
        configure({'environment': {'NIXL_SHARE_WORKER_PID': '1'}}, ['--kv-transfer-config={}'], '')
    except RuntimeError:
        rows.append({'name': '缺失Worker时拒绝静默退化', 'passed': True})
    else:
        raise AssertionError('Missing Worker name did not fail')
    # GPUStack把vLLM完整命令放在Entrypoint，Cmd可以为空；必须同时核对真实调用点的两部分命令。
    actual_calls = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == '_configure_nixl_worker_pid']
    assert len(actual_calls) == 1
    for name, command, arguments in (
        ('实际Entrypoint承载完整命令', ['vllm', 'serve', '--kv-transfer-config', '{}'], []),
        ('Cmd承载角色参数', ['vllm', 'serve'], ['--kv-transfer-config={}']),
    ):
        options = {'environment': {'NIXL_SHARE_WORKER_PID': '1'}}
        scope = {**namespace, 'create_options': options,
            'c': SimpleNamespace(execution=SimpleNamespace(command=command, args=arguments)),
            'envs': SimpleNamespace(GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME='worker-reference')}
        eval(compile(ast.Expression(body=actual_calls[0]), 'actual-runtime-call', 'eval'), scope)
        assert options['pid_mode'] == 'container:worker-reference'
        rows.append({'name': name, 'passed': True})
    result = text.encode()
    Path(args.output).write_bytes(result)
    report = {'runtime_version': '0.2.6', 'original_sha256': ORIGINAL_SHA256,
        'patched_sha256': hashlib.sha256(result).hexdigest(), 'checks': rows,
        'passed': all(row['passed'] for row in rows), 'host_pid_used': False}
    Path(args.verification).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
