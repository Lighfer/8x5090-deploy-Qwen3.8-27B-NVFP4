#!/usr/bin/env python3
"""在CPU上验证镜像内真实查询方法的顺序、限额和清理，不访问缓存服务或GPU。"""
import ast
from collections import OrderedDict
import importlib.util
import json
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def installed_root():
    """通过文件定位软件包，不执行第三方包的初始化代码。"""
    for entry in sys.path:
        if (Path(entry) / 'lmcache/integration/vllm/vllm_multi_process_adapter.py').is_file():
            return Path(entry)
    raise RuntimeError('缺少已安装的软件目录')


def source_methods(root):
    """提取已安装源码中的原方法；只替换其外部通信边界，不改写被测状态机。"""
    source = root / 'lmcache/integration/vllm/vllm_multi_process_adapter.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LMCacheMPSchedulerAdapter')
    wanted = {'maybe_submit_lookup_request', 'check_lookup_result', 'progress_pending_lookups', 'cleanup_lookup_result'}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if len(nodes) != len(wanted):
        raise AssertionError('查询推进方法不完整')
    for node in nodes:
        node.decorator_list = []
    namespace = {'time': time, 'logger': logging.getLogger('lookup-progress-cpu'), '_LookupAck': SimpleNamespace}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    return {name: namespace[name] for name in wanted}


class Future:
    """可手动完成的通信边界，记录是否发生阻塞等待或重复消费。"""

    def __init__(self, value=None, ready=False, error=None):
        self.value, self.ready, self.error = value, ready, error
        self.result_calls = 0

    def query(self):
        """就绪查询不等待，只返回当前标记。"""
        return self.ready

    def result(self, timeout=None):
        """被测非阻塞路径必须只消费就绪future，并传入零等待期限。"""
        if not self.ready or timeout != 0:
            raise AssertionError('出现阻塞等待或消费尚未就绪的future')
        self.result_calls += 1
        if self.error:
            raise self.error
        return self.value


class Client:
    """每次状态查询创建一个future；最终结果由用例显式完成。"""

    def __init__(self):
        self.queries = []
        self.lookups = []

    def lookup(self, key, world_size):
        """登记原提交方法发出的LOOKUP，并返回尚未确认的future。"""
        future = Future()
        self.lookups.append((key.request_id, future))
        return future

    def query_prefetch_status(self, request_id):
        """不连接任何真实服务，只登记请求顺序。"""
        future = Future()
        self.queries.append((request_id, future))
        return future


def adapter(root, requests=('hot',), servers=('one',)):
    """构造原方法实际访问的元数据；省略会连接服务的适配器构造函数。"""
    methods = source_methods(root)
    clients = {name: Client() for name in servers}
    value = SimpleNamespace(
        _nonblocking_lookup_status=True, _lookup_progress=OrderedDict(),
        _pending_lookups=set(), _finished_lookup_results={}, _per_server_hits={},
        _lookup_status={}, _lookup_params={}, tp_size=2,
        _server_urls=list(servers), req_clients=clients, is_healthy=True,
        _mq_timeout=300.0, lmcache_tokens_per_chunk=1536, freed_tails=[], timed_out=[],
        _health_events={name: SimpleNamespace(clear=lambda: None) for name in servers},
        _unacked_lookups={})
    def create_key(*args, **kwargs):
        """保留请求标识，不生成或哈希任何真实业务token。"""
        key = SimpleNamespace(request_id=kwargs['request_id'])
        key.no_worker_id_version = lambda: key
        return key
    value._create_key = create_key
    value._ensure_heartbeat_started = lambda: None
    value._free_inconsistent_lookup_locks = lambda *args: value.freed_tails.append(args)
    value._mark_lookup_timed_out = lambda name: value.timed_out.append(name)
    for name, method in methods.items():
        setattr(value, name, MethodType(method, value))
    for request_id in requests:
        value.maybe_submit_lookup_request(request_id, token_ids=[1] * 3072)
    return value


def acknowledge(value):
    """模拟服务端确认LOOKUP已登记，避免状态查询早于作业存在。"""
    for ack in value._unacked_lookups.values():
        for future in ack.futures.values():
            future.ready = True


class ProgressChecks(unittest.TestCase):
    """验证结果推进与准入解耦后，原通信、回退和清理语义仍然成立。"""

    root = None

    def make(self, *args, **kwargs):
        """每个用例取得全新元数据，避免完成值或计数跨用例泄漏。"""
        return adapter(self.root, *args, **kwargs)

    def test_ack_must_precede_status_and_result_is_consumed_once(self):
        value = self.make()
        value.progress_pending_lookups()
        self.assertEqual(value.req_clients['one'].queries, [])
        acknowledge(value)
        value.progress_pending_lookups()
        value.progress_pending_lookups()
        self.assertEqual(len(value.req_clients['one'].queries), 1)
        future = value.req_clients['one'].queries[0][1]
        future.value, future.ready = 20, True
        value.progress_pending_lookups()
        self.assertEqual(value.check_lookup_result('hot'), 30720)
        self.assertEqual(value.check_lookup_result('hot'), 30720)
        self.assertEqual(future.result_calls, 1)
        self.assertEqual(len(value._lookup_progress), 0)
        self.assertEqual(value.freed_tails, [])

    def test_zero_hit_is_finished_not_restarted(self):
        value = self.make()
        acknowledge(value)
        value.progress_pending_lookups()
        future = value.req_clients['one'].queries[0][1]
        future.value, future.ready = 0, True
        value.progress_pending_lookups()
        value.progress_pending_lookups()
        self.assertEqual(value._finished_lookup_results, {'hot': 0})
        self.assertEqual(len(value.req_clients['one'].queries), 1)

    def test_multiserver_keeps_minimum_and_frees_only_excess_tail(self):
        value = self.make(servers=('one', 'two'))
        acknowledge(value)
        value.progress_pending_lookups()
        for name, hits in [('one', 20), ('two', 18)]:
            future = value.req_clients[name].queries[0][1]
            future.value, future.ready = hits, True
        value.progress_pending_lookups()
        self.assertEqual(value._finished_lookup_results, {'hot': 18 * 1536})
        self.assertEqual(value.freed_tails, [('hot', {'one': 20, 'two': 18}, 18)])
        value.progress_pending_lookups()
        self.assertEqual(len(value.freed_tails), 1)

    def test_cleanup_cancels_progress_and_does_not_restore_late_result(self):
        value = self.make()
        acknowledge(value)
        value.progress_pending_lookups()
        future = value.req_clients['one'].queries[0][1]
        value.cleanup_lookup_result('hot')
        future.value, future.ready = 20, True
        value.progress_pending_lookups()
        self.assertNotIn('hot', value._finished_lookup_results)
        for name in ['_pending_lookups', '_unacked_lookups', '_lookup_status',
                     '_per_server_hits', '_lookup_params', '_lookup_progress']:
            self.assertNotIn('hot', getattr(value, name))

    def test_ack_timeout_finishes_as_miss_without_status_query(self):
        value = self.make()
        value._unacked_lookups['hot'].submitted_at -= 301
        value.progress_pending_lookups()
        self.assertEqual(value.timed_out, ['one'])
        self.assertEqual(value._finished_lookup_results, {'hot': 0})
        self.assertEqual(value.req_clients['one'].queries, [])

    def test_status_timeout_and_unhealthy_server_are_misses(self):
        value = self.make()
        acknowledge(value)
        value.progress_pending_lookups()
        future, submitted = value._lookup_status['hot']['one']
        value._lookup_status['hot']['one'] = (future, submitted - 301)
        value.progress_pending_lookups()
        self.assertEqual(value.timed_out, ['one'])
        self.assertEqual(value._finished_lookup_results, {'hot': 0})
        other = self.make()
        other.is_healthy = False
        other.progress_pending_lookups()
        self.assertEqual(other._finished_lookup_results, {'hot': 0})
        self.assertEqual(other.req_clients['one'].queries, [])

    def test_blocking_mode_is_never_called_from_metadata_progress(self):
        value = self.make()
        value._nonblocking_lookup_status = False
        value.check_lookup_result = lambda _: self.fail('不应进入阻塞查询')
        self.assertEqual(value.progress_pending_lookups(), 0)

    def test_count_limit_rotates_fairly_and_cleanup_is_constant_removal(self):
        value = self.make(requests=tuple('request' + str(i) for i in range(100)))
        seen = []
        value.check_lookup_result = lambda name: seen.append(name)
        self.assertEqual(value.progress_pending_lookups(max_requests=7, time_budget_s=1), 7)
        self.assertEqual(value.progress_pending_lookups(max_requests=7, time_budget_s=1), 7)
        self.assertEqual(seen, ['request' + str(i) for i in range(14)])
        value.cleanup_lookup_result('request14')
        value.progress_pending_lookups(max_requests=1, time_budget_s=1)
        self.assertEqual(seen[-1], 'request15')

    def test_soft_time_limit_returns_after_current_nonblocking_operation(self):
        value = self.make(requests=('first', 'second'))
        seen = []
        value.check_lookup_result = lambda name: seen.append(name)
        with mock.patch.object(time, 'monotonic', side_effect=[100.0, 100.002]):
            self.assertEqual(value.progress_pending_lookups(), 1)
        self.assertEqual(seen, ['first'])
        self.assertEqual(list(value._lookup_progress), ['second', 'first'])

    def test_unexpected_error_remains_visible_and_keeps_progress_entry(self):
        value = self.make()
        value.check_lookup_result = mock.Mock(side_effect=RuntimeError('synthetic failure'))
        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            value.progress_pending_lookups()
        self.assertIn('hot', value._lookup_progress)

    def test_patch_identity_rejects_unknown_input_and_accepts_complete_output(self):
        spec = importlib.util.spec_from_file_location('lookup_progress_apply', HERE / 'apply-patch.py')
        patcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patcher)
        manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
        values, changed = patcher.prepare(self.root, manifest)
        self.assertFalse(changed)
        self.assertEqual(set(values), set(manifest['final_files']))
        invalid = dict(manifest, original_files={})
        with self.assertRaisesRegex(RuntimeError, '固定范围'):
            patcher.prepare(self.root, invalid)
        with tempfile.TemporaryDirectory(prefix='lookup-source-identity-') as temporary:
            root = Path(temporary)
            for name, content in values.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            target = root / next(iter(values))
            target.write_bytes(target.read_bytes() + b'\n# synthetic unknown input\n')
            before = {name: (root / name).read_bytes() for name in values}
            with self.assertRaisesRegex(RuntimeError, '固定基线'):
                patcher.prepare(root, manifest)
            self.assertEqual(before, {name: (root / name).read_bytes() for name in values})


def main():
    """结果只代表CPU状态机检查；不声明已完成PD、MTP或GPU性能验收。"""
    root = installed_root()
    ProgressChecks.root = root
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProgressChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'passed': result.wasSuccessful(), 'checks': result.testsRun,
                      'gpu_inference_verified': False}))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
