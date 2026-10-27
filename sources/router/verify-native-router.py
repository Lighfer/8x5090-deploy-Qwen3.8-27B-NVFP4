"""构建时验证原生扩展与启动器保持兼容，不关闭平台依赖的多路由模式。"""
import importlib.metadata
from pathlib import Path

import vllm_router_rs
from vllm_router.launch_router import parse_router_args

assert importlib.metadata.version('vllm-router') == '0.1.16'
assert hasattr(vllm_router_rs, 'Router')
# 启动脚本沿用官方实现，禁止混入之前仅供诊断的单路由包装器。
# 启动器位置由当前解释器环境决定，不依赖旧版虚拟环境的绝对路径。
import shutil
launcher = Path(shutil.which('vllm-router')).read_text()
assert 'single-pd-router.py' not in launcher
arguments = ['--enable-igw', '--vllm-pd-disaggregation', '--kv-connector', 'nixl',
             '--host', '127.0.0.1', '--port', '18116',
             '--prefill-policy', 'cache_aware', '--decode-policy', 'round_robin', '--balance-abs-threshold', '2',
             '--prefill', 'http://127.0.0.1:41000', '--decode', 'http://127.0.0.1:41002']
parsed = parse_router_args(arguments)
assert parsed.enable_igw and parsed.vllm_pd_disaggregation and parsed.kv_connector == 'nixl'
assert parsed.prefill_policy == 'cache_aware' and parsed.decode_policy == 'round_robin' and parsed.balance_abs_threshold == 2
print('NATIVE_ROUTER_ABI_AND_IGW_ARGUMENTS_PASS')
