#!/usr/bin/env python3
"""在独立临时目录和内存平台中验证切换边界，不访问 Docker、GPU 或真实管理接口。"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from support import generated_fixture
import urllib.parse

MODULE_TEMPORARY = tempfile.TemporaryDirectory(prefix='switch-module-')
MODULE_ROOT = Path(MODULE_TEMPORARY.name)
generated_fixture(MODULE_ROOT)
spec = importlib.util.spec_from_file_location("switch", MODULE_ROOT / "switch-pd.py")
switch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(switch)



class FakeHost:
    """模拟具有缓存引用约束的平台，只接受公开接口，不复用执行器的目标构造函数。"""

    def __init__(self, root, initial="1p1d-cache", fail=None):
        self.root = Path(root)
        self.config = switch.read_json(self.root / "switch-config.json")
        self.model = switch.read_json(self.root / "switch-profiles" / (initial + ".json"))
        self.model.update(id=1, cluster_id=1, owner_principal_id=1, replicas=1, updated_at="original", access_policy="authed")
        # 平台 API 会补出 YAML 里未出现的默认字段，用实际响应形态验证重复执行。
        self.model["disaggregation"]["vendor"] = None
        self.model.setdefault('gather', {})
        self.model.setdefault('speculative_config', {'enabled': False})
        self.caches = {}
        self.next_id = 100
        self.generation = 0
        self.fail = fail
        self.failures = 0
        self.changes = []
        self.validated = True
        self.active_nginx = (self.root / "entrypoint/nginx.conf").read_text()
        self.opened_validated = []
        self.container_identity = {name: {"Name": "/" + name, "Id": name + "-id", "RestartCount": 0,
            "State": {"Running": True, "StartedAt": "unchanged"}, "Config": {"Labels": {}}}
            for name in (self.config["server_container"], self.config["worker_container"], self.config["proxy_container"], "external-business-fixture")}
        if initial != "1p1d-no-cache":
            service = switch.read_json(self.root / "cache-service.json")
            self.caches[11] = {**service, "id": 11, "healthy": True, "state": "running"}
            for role in self.model["roles"]:
                if role["name"] == "prefill":
                    role["extended_kv_cache"] = {"enabled": True, "mode": "shared", "cache_service_id": 11,
                        "chunk_size": self.config["chunk_size"]}

    def failure(self, phase):
        """指定边界只失败一次，允许验证独立回滚能否完成；永久失败另外显式指定。"""
        if self.fail == phase and self.failures == 0:
            self.failures += 1
            raise TimeoutError("模拟边界响应中断：" + phase)

    def instances(self):
        if not self.model.get("replicas"):
            return []
        result = []
        gpu = 0
        for role in self.model["roles"]:
            for number in range(role["replicas"]):
                item = {"id": len(result) + 1, "name": self.model["name"] + "-" + role["name"] + "-f" + str(number),
                    "role": role["name"], "worker_ip": urllib.parse.urlsplit(self.config["entry_url"]).hostname, "port": 40001 + len(result), "state": "running"}
                if role["name"] != "router":
                    tp = int(switch.parse_options(role["backend_parameters"])["--tensor-parallel-size"])
                    item["gpu_indexes"] = [int(v.rsplit(":", 1)[1]) for v in role["gpu_selector"]["gpu_ids"]][number * tp:(number + 1) * tp]
                    gpu += tp
                result.append(item)
        return result

    def call(self, method, route, body=None):
        if route == "/v2/models/import" and method == "POST":
            if body.get("dry_run") is not True:
                raise AssertionError("测试不允许部署导入写操作")
            return {"valid": True, "dry_run": True}
        if route == "/v2/models/1/instances":
            return {"items": self.instances()}
        if route == "/v2/models/1":
            if method == "GET":
                return copy.deepcopy(self.model)
            self.model.update(copy.deepcopy(body))
            self.model["updated_at"] = "changed-" + str(len(self.changes))
            self.changes.append(("model", self.model["replicas"]))
            self.generation += 1
            if self.model["replicas"]:
                self.validated = False
                self.failure("after-model-start")
            return copy.deepcopy(self.model)
        if route == "/v2/cache-services":
            if method == "GET":
                return {"items": copy.deepcopy(list(self.caches.values()))}
            self.failure("before-cache-create")
            if self.caches:
                raise AssertionError("旧缓存未退出就创建了第二份内存池")
            self.next_id += 1
            value = {**copy.deepcopy(body), "id": self.next_id, "healthy": True, "state": "running"}
            self.caches[value["id"]] = value
            self.changes.append(("cache-create", value["id"]))
            self.failure("after-cache-create")
            return copy.deepcopy(value)
        if route.startswith("/v2/cache-services/"):
            pieces = route.split("/")
            identifier = int(pieces[3])
            references = [r for r in self.model["roles"] if
                (r.get("extended_kv_cache") or {}).get("enabled") and
                (r.get("extended_kv_cache") or {}).get("cache_service_id") == identifier]
            if route.endswith("/models"):
                return {"items": [{"id": 1}] if references else []}
            if method == "DELETE":
                if references:
                    raise AssertionError("删除了仍被模型引用的缓存")
                del self.caches[identifier]
                self.changes.append(("cache-delete", identifier))
                self.failure('after-cache-delete')
                return None
            return copy.deepcopy(self.caches[identifier])
        raise AssertionError((method, route))

    def run(self, arguments, timeout=60):
        if len(arguments) > 1 and arguments[1].endswith('/verify-model-features.py'):
            assert switch.MARKER in self.active_nginx
            switch.write_json(Path(arguments[arguments.index('--output') + 1]), {'all_passed': True})
            return ''
        if len(arguments) > 1 and arguments[1].endswith('/verify-v032-migration.py'):
            # 隔离实现仅核对维护条件和报告交接，不发请求或调用生产Docker。
            if switch.MARKER not in self.active_nginx:
                raise AssertionError('扩展验证不能在入口开放时运行')
            job = Path(arguments[arguments.index('--job') + 1])
            switch.write_json(job / 'extended-verification.json', {'passed': True, 'isolated_fake': True})
            self.extended_was_isolated = True
            return ''
        if arguments[:3] == ["docker", "image", "inspect"]:
            return self.config["image_ids"][arguments[-1]] + "\n"
        if arguments[:2] == ["docker", "inspect"]:
            return self.config["control_plane_images"][arguments[-1]] + "\n"
        if arguments[0] == "nvidia-smi":
            if any(str(item).startswith('--query-compute-apps=') for item in arguments):
                return ''
            return "".join(str(index) + ", NVIDIA GeForce RTX 5090\n" for index in self.config["hardware_gpu_ids"])
        if arguments[:2] == ["docker", "stop"]:
            self.container_identity[self.config["proxy_container"]]["State"]["Running"] = False
        return ""

    def container_logs(self, name, since):
        if self.fail == 'startup-gpu-oom' and self.failures == 0:
            self.failures += 1
            return 'Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory'
        if self.fail == 'recoverable-autotune-oom':
            return 'memory allocation failed with OOM\nAutotuner falling back\nApplication startup complete'
        return ''

    def request(self, path, body=None, token=None, base=None, timeout=15):
        gated = switch.MARKER in self.active_nginx
        if path in ("/v1/chat/completions", "/v1/responses") and gated:
            import re
            expected = re.search(r'\$http_x_gpustack_switch != "([^"]+)"', self.active_nginx).group(1)
            if token != expected:
                return 503, "{}"
        if path == "/metrics":
            return 200, 'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 0\n'
        if path == "/health":
            return 200, "OK"
        if path == "/workers":
            workers = [{"url": switch.endpoint(i), "worker_type": i["role"], "model_id": self.config["model_name"], "is_healthy": True}
                for i in self.instances() if i["role"] != "router"]
            if self.fail == "missing-router-member":
                workers = workers[:-1]
            return 200, json.dumps({"workers": workers})
        if path == "/v1/models":
            return 200, json.dumps({"object": "list", "data": [{"id": self.config["model_name"]}]})
        if path == "/v1/chat/completions":
            if not self.model.get('replicas'):
                return 503, '{}'
            if not body:
                return 422, "{}"
            if body.get("stream"):
                return 200, 'data: {"choices":[{"delta":{"content":"42"}}]}\n\ndata: [DONE]\n\n'
            return 200, json.dumps({"choices": [{"message": {"content": "42", "reasoning": "17+25=42"}}]})
        if path == "/v1/responses":
            self.failure("responses")
            self.validated = True
            if body.get('stream'):
                return 200, '\n\n'.join('data: ' + json.dumps(item) for item in [
                    {'type': 'response.content_part.added'}, {'type': 'response.reasoning_text.delta', 'delta': '17+25=42'},
                    {'type': 'response.output_text.delta', 'delta': '42'}, {'type': 'response.content_part.done'},
                    {'type': 'response.completed', 'response': {'status': 'completed'}}])
            return 200, json.dumps({"status": "completed", "output": [{"text": "42"}]})
        raise AssertionError(path)

    def memory(self):
        return {"MemTotal": (self.config["cache_gib"] + 80) * 1024 ** 3, "MemAvailable": (self.config["minimum_free_gib"] + 16 + (0 if self.caches else self.config["cache_gib"])) * 1024 ** 3,
                "events": {"oom": 0, "oom_kill": 0}}

    def boot_id(self):
        return "isolated-boot-original"

    def containers(self):
        result = [copy.deepcopy(v) for v in self.container_identity.values() if v["State"]["Running"]]
        result += [{"Name": "/" + i["name"], "Config": {"Labels": {"runtime.gpustack.ai/workload": i["name"]}}} for i in self.instances()]
        result += [{"Name": "/" + self.config["cache_name"] + "-server-fake", "Config": {"Labels": {"type": "cache-service", "runtime.gpustack.ai/workload": self.config["cache_name"] + "-server-fake"}}}
                   for service in self.caches.values()]
        return result

    def inspect_proxy(self):
        return self.container_identity[self.config["proxy_container"]]

    def reload_nginx(self):
        self.active_nginx = (self.root / "entrypoint/nginx.conf").read_text()
        if switch.MARKER not in self.active_nginx:
            self.opened_validated.append(self.validated)

    def synchronize_entry(self):
        pass

    def parameters(self, job):
        self.failure("parameters")
        if self.fail == "permanent-parameters":
            raise switch.SwitchError("实际配置持续不正确")
        switch.write_json(Path(job) / "verified-model.json", switch.payload(self.model))
        # 实际参数验证器同时写完整报告；保持模拟Host与真实Host的文件契约一致。
        switch.write_json(Path(job) / 'parameters.json', {'passed': True, 'checks': []})
        return {"passed": True, "instances": len(self.instances()) - 1, "all_gpus": True}

    def sleep(self, seconds):
        # 资源状态同步完成，模拟环境无需等待真实的 GPU 加载时间。
        pass


class SwitchingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gpustack-switch-unit-")
        self.root = Path(self.temporary.name)
        self.site = generated_fixture(self.root)
        (self.root / "baks/pd-switch").mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def test_command_line_status_reaches_the_real_dispatcher(self):
        # 使用真实参数解析和main，防止只测试内部函数而遗漏命令入口的属性或路径错误。
        with mock.patch.object(switch, 'ROOT', self.root), \
             mock.patch.object(switch.os, 'geteuid', return_value=0), \
             mock.patch.object(switch, 'status', return_value=0) as status, \
             mock.patch.object(switch.sys, 'argv', ['switch-pd.py', 'status']):
            self.assertEqual(switch.main(), 0)
            status.assert_called_once_with(self.root)

    def prepare(self, initial, target, fail=None, restart=True):
        host = FakeHost(self.root, initial, fail)
        args = SimpleNamespace(mode=target, drain_timeout=0, restart=restart)
        job = switch.prepare_job(host, args)
        return host, switch.Switch(host, job)

    def assert_layout(self, host, mode):
        roles = {r["name"]: r for r in host.model["roles"]}
        for name in ("prefill", "decode"):
            replicas, tp = switch.MODE_SHAPES[mode][name]
            self.assertEqual(roles[name]["replicas"], replicas)
            flags = switch.parse_options(roles[name]["backend_parameters"])
            self.assertEqual(int(flags["--tensor-parallel-size"]), tp)
            self.assertEqual(flags["--speculative-config"]["num_speculative_tokens"], 3)
            self.assertEqual(flags["--default-chat-template-kwargs"]["reasoning_effort"], "xhigh")
        self.assertFalse((host.model.get("extended_kv_cache") or {}).get("enabled"))
        self.assertFalse((roles["decode"].get("extended_kv_cache") or {}).get("enabled"))
        self.assertEqual(host.model["id"], 1)
        self.assertEqual(host.model["name"], self.site["names"]["model"])
        self.assertEqual(host.model["access_policy"], "authed")
        if mode.endswith("no-cache"):
            self.assertFalse(host.caches)
            self.assertFalse((roles["prefill"].get("extended_kv_cache") or {}).get("enabled"))
        else:
            self.assertEqual(len(host.caches), 1)
            identifier = next(iter(host.caches))
            self.assertEqual(roles["prefill"]["extended_kv_cache"]["cache_service_id"], identifier)
        live = switch.read_json(host.root / "pd-model-live.json")
        live_p = next(r for r in live["roles"] if r["name"] == "prefill")
        self.assertEqual((live_p.get("extended_kv_cache") or {}).get("cache_service_id"),
                         (roles["prefill"].get("extended_kv_cache") or {}).get("cache_service_id"))
        yaml_text = (host.root / "pd-model-stopped.yaml").read_text()
        self.assertIn('name: ' + json.dumps(self.site['names']['model']) + '\n', yaml_text)
        # 顶层流式值可以独立按JSON解析，验证导出没有省略完整角色或缓存编号。
        restored = {line.split(": ", 1)[0]: json.loads(line.split(": ", 1)[1]) for line in yaml_text.splitlines()}
        self.assertEqual(restored["replicas"], 0)
        self.assertEqual(restored["roles"], live["roles"])
        self.assertNotIn(switch.MARKER, host.active_nginx)
        self.assertTrue(all(host.opened_validated))

    def test_all_nine_switch_directions(self):
        # 每一对源/目标均重新建独立环境，验证缓存开关和 TP4/TP2 的组合边界。
        for initial in switch.MODES:
            for target in switch.MODES:
                with self.subTest(initial=initial, target=target):
                    host, task = self.prepare(initial, target)
                    task.execute()
                    self.assertEqual(task.state["phase"], "committed")
                    self.assert_layout(host, target)

    def test_healthy_same_mode_does_not_restart(self):
        host, task = self.prepare("1p1d-cache", "1p1d-cache", restart=False)
        task.execute()
        self.assertEqual(task.state["phase"], "unchanged")
        self.assertEqual(host.changes, [])
        self.assert_layout(host, "1p1d-cache")

    def test_health_only_switch_and_rollback_never_send_generation(self):
        # 在内存平台验证维护边界，任何带正文的生成调用都会直接失败；不接触真实模型。
        # 同时覆盖成功切换和启动失败后回滚，避免恢复过程偷偷改用默认生成探测。
        for fail in (None, 'after-model-start'):
            with self.subTest(fail=fail):
                host, task = self.prepare('1p1d-cache', '2p2d-cache', fail=fail)
                task.save(health_only=True)
                original = host.request

                def request(path, body=None, **kwargs):
                    if path in ('/v1/chat/completions', '/v1/responses') and body:
                        self.fail('健康检查模式发送了生成请求')
                    return original(path, body, **kwargs)

                host.request = request
                task.execute()
                expected = 'rolled-back' if fail else 'committed'
                self.assertEqual(task.state['phase'], expected)
                report = switch.read_json(task.job / ('rollback-verification.json' if fail else 'verification.json'))
                self.assertEqual(report['verification_level'], 'health_only')
                self.assertEqual(report['inference_requests_sent'], 0)
                self.assertFalse(report['functional_inference_verified'])
                self.assertTrue(report['members']['passed'])
                self.assertNotIn(switch.MARKER, host.active_nginx)

    def test_container_snapshot_tolerates_confirmed_concurrent_removal(self):
        # 独立模拟ps返回后旧模型容器被平台删除；仍然保留其余容器的真实检查结果。
        host = object.__new__(switch.Host)
        lists = iter(['stable removed stable2', 'stable stable2'])

        def run(arguments):
            if arguments == ['docker', 'ps', '-q']:
                return next(lists)
            if arguments[-1] == 'removed':
                raise switch.SwitchError('error: no such object: removed')
            return json.dumps([{'Id': arguments[-1]}])

        host.run = run
        self.assertEqual(host.containers(), [{'Id': 'stable'}, {'Id': 'stable2'}])

    def test_container_snapshot_does_not_hide_docker_failure(self):
        # Docker不可用或对象仍在列表中都不是已经确认的正常删除，必须让事务保留故障状态。
        for message in ('Cannot connect to Docker daemon', 'error: no such object: still-running'):
            host = object.__new__(switch.Host)

            def run(arguments):
                if arguments == ['docker', 'ps', '-q']:
                    return 'still-running'
                raise switch.SwitchError(message)

            host.run = run
            with self.assertRaises(switch.SwitchError):
                host.containers()

    def test_preflight_rejects_escaped_json_even_with_matching_checksum(self):
        # 文件散列只证明内容身份，不能证明命令行语法正确；过度转义即使已登记也必须在停机前拒绝。
        host = FakeHost(self.root)
        path = self.root / 'switch-profiles/1p1d-cache.json'
        model = switch.read_json(path)
        parameters = model['roles'][0]['backend_parameters']
        for index, argument in enumerate(parameters):
            if argument.startswith('--limit-mm-per-prompt='):
                parameters[index] = argument.replace('"', '\\"')
        switch.write_json(path, model)
        host.config['profile_sha256']['1p1d-cache'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(switch.SwitchError, '标准JSON'):
            switch.preflight(host, '1p1d-cache')
        self.assertFalse(host.changes)
        self.assertNotIn(switch.MARKER, host.active_nginx)

    def test_every_mode_stops_releases_cache_and_starts_same_mode(self):
        # 上次方案存储独立于事务日志；停止释放GPU和专用池，启动重新分配缓存编号。
        for mode in switch.MODES:
            with self.subTest(mode=mode):
                host = FakeHost(self.root, mode)
                args = SimpleNamespace(mode='stop', drain_timeout=0, restart=False)
                switch.resolve_command(host, args)
                task = switch.Switch(host, switch.prepare_job(host, args))
                task.execute()
                self.assertEqual(task.state['phase'], 'committed')
                self.assertEqual(host.model['replicas'], 0)
                self.assertFalse(host.caches)
                saved = switch.read_json(self.root / 'model-state.json')
                self.assertFalse(saved['running'])
                self.assertEqual(saved['last_mode'], mode)
                self.assertIn(switch.STOP_MARKER, host.active_nginx)
                # 重复停止必须保持同一方案；不要求已经停机的服务提供推理响应。
                again = SimpleNamespace(mode='stop', drain_timeout=0, restart=False)
                switch.resolve_command(host, again)
                repeat = switch.Switch(host, switch.prepare_job(host, again))
                repeat.execute()
                self.assertEqual(repeat.state['phase'], 'committed')
                # 模拟丢弃全部旧事务后，start依然只依赖model-state与正式配置。
                start = SimpleNamespace(mode='start', drain_timeout=0, restart=False)
                switch.resolve_command(host, start)
                self.assertEqual(start.mode, mode)
                resumed = switch.Switch(host, switch.prepare_job(host, start))
                resumed.execute()
                self.assertEqual(resumed.state['phase'], 'committed')
                self.assertTrue(switch.read_json(self.root / 'model-state.json')['running'])
                self.assertNotIn(switch.STOP_MARKER, host.active_nginx)
                self.assert_layout(host, mode)

    def test_failed_start_restores_stopped_gate_and_last_mode(self):
        host = FakeHost(self.root, '1p2d-6gpu-cache')
        stop = SimpleNamespace(mode='stop', drain_timeout=0, restart=False)
        switch.resolve_command(host, stop)
        task = switch.Switch(host, switch.prepare_job(host, stop))
        task.execute()
        saved = switch.read_json(self.root / 'model-state.json')
        host.fail = 'after-model-start'
        start = SimpleNamespace(mode='start', drain_timeout=0, restart=False)
        switch.resolve_command(host, start)
        resumed = switch.Switch(host, switch.prepare_job(host, start))
        resumed.execute()
        self.assertEqual(resumed.state['phase'], 'rolled-back')
        self.assertEqual(host.model['replicas'], 0)
        self.assertFalse(host.caches)
        self.assertEqual(switch.read_json(self.root / 'model-state.json'), saved)
        self.assertIn(switch.STOP_MARKER, host.active_nginx)
        self.assertEqual(host.request('/v1/chat/completions', {})[0], 503)

    def test_stop_failure_restores_running_mode(self):
        host = FakeHost(self.root, '1p1d-cache', fail='after-cache-delete')
        args = SimpleNamespace(mode='stop', drain_timeout=0, restart=False)
        switch.resolve_command(host, args)
        task = switch.Switch(host, switch.prepare_job(host, args))
        task.execute()
        self.assertEqual(task.state['phase'], 'rolled-back')
        self.assert_layout(host, '1p1d-cache')

    def test_unknown_running_configuration_is_not_guessed_on_stop(self):
        host = FakeHost(self.root)
        host.model['roles'][0]['backend_parameters'].append('--unknown-option=1')
        with self.assertRaisesRegex(switch.SwitchError, '不一致'):
            switch.resolve_command(host, SimpleNamespace(mode='stop'))
        self.assertEqual(host.changes, [])

    def test_known_previous_image_stops_and_starts_latest_profile(self):
        """更新模板后旧引擎仍可停机；start保留拓扑并读取最新镜像及缓存定义。"""
        for mode in switch.MODES:
            with self.subTest(mode=mode):
                host = FakeHost(self.root, mode)
                legacy = 'local/previous-runtime:locked'
                identity = 'sha256:' + '1' * 64
                host.config['previous_runtime_images'] = {legacy: identity}
                host.config['image_ids'][legacy] = identity
                host.model['image_name'] = legacy
                for cache in host.caches.values():
                    cache['config']['image'] = legacy
                stop = SimpleNamespace(mode='stop', drain_timeout=0, restart=False)
                switch.resolve_command(host, stop)
                self.assertEqual(stop.mode, mode)
                task = switch.Switch(host, switch.prepare_job(host, stop))
                task.execute()
                self.assertEqual(task.state['phase'], 'committed')
                start = SimpleNamespace(mode='start', drain_timeout=0, restart=False)
                switch.resolve_command(host, start)
                resumed = switch.Switch(host, switch.prepare_job(host, start))
                resumed.execute()
                self.assertEqual(resumed.state['phase'], 'committed')
                wanted = switch.read_json(self.root / 'switch-profiles' / (mode + '.json'))
                self.assertEqual(host.model['image_name'], wanted['image_name'])
                self.assertEqual(start.mode, mode)
                for cache in host.caches.values():
                    self.assertEqual(cache['config']['image'], switch.read_json(self.root / 'cache-service.json')['config']['image'])

    def test_previous_image_does_not_hide_modified_parameters(self):
        """旧版本兼容不能忽略其他参数差异，也不能触发任何模型写入。"""
        host = FakeHost(self.root)
        legacy = 'local/previous-runtime:locked'
        identity = 'sha256:' + '2' * 64
        host.config['previous_runtime_images'] = {legacy: identity}
        host.config['image_ids'][legacy] = identity
        host.model['image_name'] = legacy
        host.model['roles'][0]['backend_parameters'].append('--unknown-option=1')
        with self.assertRaisesRegex(switch.SwitchError, '不一致'):
            switch.resolve_command(host, SimpleNamespace(mode='stop'))
        self.assertEqual(host.changes, [])

    def test_previous_image_rejects_changed_image_identity(self):
        """旧标签指向不同镜像时拒绝过渡识别，不能只凭标签名称放行。"""
        host = FakeHost(self.root)
        legacy = 'local/previous-runtime:locked'
        host.config['previous_runtime_images'] = {legacy: 'sha256:' + '3' * 64}
        host.config['image_ids'][legacy] = 'sha256:' + '4' * 64
        host.model['image_name'] = legacy
        with self.assertRaisesRegex(switch.SwitchError, '登记与镜像锁不一致'):
            switch.resolve_command(host, SimpleNamespace(mode='stop'))
        host.config['previous_runtime_images'][legacy] = host.config['image_ids'][legacy]
        host.run = lambda arguments, timeout=60: 'sha256:' + '5' * 64
        with self.assertRaisesRegex(switch.SwitchError, '内容已变化'):
            switch.resolve_command(host, SimpleNamespace(mode='stop'))
        self.assertEqual(host.changes, [])

    def test_unregistered_image_is_not_guessed_on_stop(self):
        """相同参数配合未知镜像仍拒绝识别，已登记镜像列表不会扩大到任意版本。"""
        host = FakeHost(self.root)
        host.model['image_name'] = 'local/unknown-runtime:latest'
        with self.assertRaisesRegex(switch.SwitchError, '不一致'):
            switch.resolve_command(host, SimpleNamespace(mode='stop'))
        self.assertEqual(host.changes, [])

    def test_enabled_top_level_speculation_is_not_a_default(self):
        """只有明确关闭的空配置可以规范化，启用顶层草稿部署仍属于实质差异。"""
        host = FakeHost(self.root)
        host.model['speculative_config'] = {'enabled': True}
        with self.assertRaisesRegex(switch.SwitchError, '不一致'):
            switch.resolve_command(host, SimpleNamespace(mode='stop'))
        self.assertEqual(host.changes, [])

    def test_nonempty_gather_is_not_a_default(self):
        """不能把真实采集配置当作平台补出的空字段忽略。"""
        host = FakeHost(self.root)
        host.model['gather'] = {'candidate-setting': True}
        with self.assertRaisesRegex(switch.SwitchError, '不一致'):
            switch.resolve_command(host, SimpleNamespace(mode='stop'))
        self.assertEqual(host.changes, [])

    def test_six_gpu_cache_server_also_excludes_reserved_cards(self):
        host, task = self.prepare('1p1d-cache', '1p2d-6gpu-cache')
        definition = switch.read_json(task.job / 'target-cache.json')
        self.assertEqual(definition['config']['env']['CUDA_VISIBLE_DEVICES'], ','.join(map(str, self.site['gpus']['six_card_order'])))
        role = {r['name']: r for r in switch.read_json(task.job / 'target-profile.json')['roles']}
        self.assertEqual(role['prefill']['gpu_selector']['gpu_ids'], [self.site['names']['node'] + ':cuda:' + str(i) for i in self.site['gpus']['six_card_order'][:2]])
        self.assertEqual(role['decode']['gpu_selector']['gpu_ids'], [self.site['names']['node'] + ':cuda:' + str(i) for i in self.site['gpus']['six_card_order'][2:]])

    def test_start_from_stopped_does_not_wait_for_nonexistent_requests(self):
        host, task = self.prepare('1p1d-no-cache', '1p1d-no-cache')
        host.model['replicas'] = 0
        task.save(drain_timeout=120)
        host.sleep = lambda seconds: self.fail('已经停止的模型不应空等排空超时')
        task.drain()

    def test_reserved_cards_in_use_rejects_eight_card_switch_before_mutation(self):
        host = FakeHost(self.root, '1p2d-6gpu-cache')
        original = host.run
        host.run = lambda args, **kw: '123, external-model' if any(str(a).startswith('--query-compute-apps=') for a in args) else original(args, **kw)
        with self.assertRaisesRegex(switch.SwitchError, '其他计算进程'):
            switch.preflight(host, '1p1d-cache')
        self.assertFalse(host.changes)
        self.assertNotIn(switch.MARKER, host.active_nginx)
        # 保持六卡模式时不检查预留卡上的其他用途，不干扰预留卡正常使用。
        self.assertTrue(switch.preflight(host, '1p2d-6gpu-cache')['dry_run'])

    def test_same_mode_extended_check_isolates_without_restart(self):
        host, task = self.prepare("1p1d-no-cache", "1p1d-no-cache", restart=False)
        task.save(extended_check=True)
        task.execute()
        self.assertEqual(task.state["phase"], "unchanged")
        self.assertTrue(host.extended_was_isolated)
        self.assertEqual(host.changes, [])
        self.assertNotIn(switch.MARKER, host.active_nginx)
        self.assertTrue(switch.read_json(task.job / 'verification.json')['extended']['passed'])

    def test_faults_restore_original_with_new_cache_id(self):
        for phase in ("before-cache-create", "after-cache-create", "after-model-start", "startup-gpu-oom", "parameters", "responses"):
            with self.subTest(phase=phase):
                host, task = self.prepare("1p1d-cache", "2p2d-cache", fail=phase)
                task.execute()
                self.assertEqual(task.state["phase"], "rolled-back")
                self.assert_layout(host, "1p1d-cache")
                self.assertNotIn(11, host.caches)

    def test_rollback_to_original_without_cache(self):
        host, task = self.prepare("1p1d-no-cache", "2p2d-cache", fail="parameters")
        task.execute()
        self.assertEqual(task.state["phase"], "rolled-back")
        self.assert_layout(host, "1p1d-no-cache")

    def test_first_deployment_failure_restores_stopped_state(self):
        host = FakeHost(self.root, '1p1d-no-cache', fail='after-model-start')
        host.model['replicas'] = 0
        args = argparse.Namespace(mode='2p2d-cache', drain_timeout=0, restart=True)
        task = switch.Switch(host, switch.prepare_job(host, args))
        task.execute()
        self.assertEqual(task.state['phase'], 'rolled-back')
        self.assertEqual(host.model['replicas'], 0)
        self.assertNotIn(switch.MARKER, host.active_nginx)
        self.assertEqual(host.request('/v1/chat/completions', {})[0], 503)

    def test_recovered_autotune_warning_does_not_abort(self):
        host, task = self.prepare('1p1d-no-cache', '2p2d-cache', fail='recoverable-autotune-oom')
        task.execute()
        self.assertEqual(task.state['phase'], 'committed')
        self.assert_layout(host, '2p2d-cache')

    def test_independent_recovery_after_incomplete_cache_creation(self):
        host, task = self.prepare("1p1d-cache", "2p2d-cache")
        task.install_gate()
        task.stop_model()
        task.delete_cache()
        # 平台已创建，但执行进程未拿到响应；新的恢复实例只能从平台独占名称找回它。
        host.call("POST", "/v2/cache-services", switch.read_json(task.job / "target-cache.json"))
        restored = switch.Switch(host, task.job)
        restored.rollback()
        self.assertEqual(restored.state["phase"], "rolled-back")
        self.assert_layout(host, "1p1d-cache")

    def test_rollback_failure_keeps_gate_and_can_retry(self):
        host, task = self.prepare("1p1d-cache", "2p2d-cache", fail="permanent-parameters")
        task.execute()
        self.assertEqual(task.state["phase"], "rollback-failed")
        self.assertIn(switch.MARKER, host.active_nginx)
        self.assertEqual(host.request("/v1/chat/completions", {})[0], 503)
        host.fail = None
        retry = switch.Switch(host, task.job)
        retry.rollback()
        self.assertEqual(retry.state["phase"], "rolled-back")
        self.assert_layout(host, "1p1d-cache")

    def test_partial_router_members_are_rejected(self):
        host, task = self.prepare("2p2d-cache", "2p2d-cache", fail="missing-router-member")
        with self.assertRaises(switch.SwitchError):
            task.members(host.instances())

    def test_preflight_rejects_modified_profile_before_mutation(self):
        host = FakeHost(self.root)
        path = self.root / "switch-profiles/2p2d-cache.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(switch.SwitchError, "散列"):
            switch.preflight(host, "2p2d-cache")
        self.assertEqual(host.changes, [])
        self.assertNotIn(switch.MARKER, host.active_nginx)

    def test_another_model_cache_reference_blocks_preflight(self):
        host = FakeHost(self.root)
        original = host.call
        host.call = lambda method, route, body=None: {"items": [{"id": 1}, {"id": 2}]} if route.endswith("/models") else original(method, route, body)
        with self.assertRaisesRegex(switch.SwitchError, "其他模型"):
            switch.preflight(host, "2p2d-cache")
        self.assertEqual(host.changes, [])

    def test_global_lock_blocks_parallel_switch(self):
        with switch.transaction_lock(self.root):
            with self.assertRaises(switch.SwitchError):
                with switch.transaction_lock(self.root):
                    self.fail("并发锁未生效")

    def test_recovery_paths_cannot_escape_backup_directory(self):
        with self.assertRaises(switch.SwitchError):
            switch.checked_job(self.root, "../../elsewhere")

    def test_reboot_refreshes_service_identity_without_losing_original_model(self):
        host, task = self.prepare("1p1d-cache", "2p2d-cache")
        task.install_gate()
        task.stop_model()
        host.boot_id = lambda: "isolated-boot-new"
        host.container_identity[host.config["worker_container"]]["State"]["StartedAt"] = "rebooted"
        task.rebase_after_reboot()
        task.rollback()
        self.assertEqual(task.state["phase"], "rolled-back")
        self.assertTrue((task.job / "protected-before-previous-boot.json").exists())
        self.assert_layout(host, "1p1d-cache")


if __name__ == "__main__":
    unittest.main(verbosity=2)
