"""在合成安装目录验证预备升级状态和恢复，不连接生产环境。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from support import generated_fixture


class NextStartTests(unittest.TestCase):
    """覆盖停止意图、开机边界、事务绑定、部分恢复和外部修改保护。"""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='prepared-release-')
        self.root = Path(self.temporary.name)
        generated_fixture(self.root)
        spec = importlib.util.spec_from_file_location('prepared_release_fixture', self.root / 'next-start.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.old = b'{"version":"original"}\n'
        self.new = b'{"version":"candidate"}\n'
        backup = self.root / 'baks/prepared/files'
        backup.mkdir(parents=True)
        (backup / 'example.json').write_bytes(self.old)
        (self.root / 'baks/pd-switch').mkdir()
        (self.root / 'example.json').write_bytes(self.new)
        self.plan = {'phase': 'prepared', 'release': 'synthetic-release', 'prepared_boot_id': 'old-boot',
                     'files': {'example.json': {'backup': 'baks/prepared/files/example.json',
                        'before_sha256': hashlib.sha256(self.old).hexdigest(),
                        'after_sha256': hashlib.sha256(self.new).hexdigest()}}}
        self.module.save(self.root / 'next-start-release.json', self.plan)
        self.module.save(self.root / 'model-state.json', {'running': True, 'last_mode': '2p2d-cache'})

    def tearDown(self):
        self.temporary.cleanup()

    def state(self):
        return self.module.read(self.root / 'next-start-release.json')

    def test_same_boot_does_not_apply(self):
        self.assertFalse(self.module.boot_should_apply(self.root, 'old-boot'))
        self.assertTrue(self.module.boot_should_apply(self.root, 'new-boot'))

    def test_pending_fp8_target_overrides_start_but_not_stop(self):
        """准备目标不冒充当前成功模式，只有start读取预案，stop仍识别正在运行的权重。"""
        from test_switch import FakeHost, switch
        host = FakeHost(self.root, '2p2d-cache')
        plan = self.state()
        plan['target_mode'] = switch.FP8_MODE
        self.module.save(self.root / 'next-start-release.json', plan)
        args = switch.resolve_command(host, SimpleNamespace(mode='start'))
        self.assertEqual(args.mode, switch.FP8_MODE)
        args = switch.resolve_command(host, SimpleNamespace(mode='stop'))
        self.assertEqual(args.mode, '2p2d-cache')
        plan['phase'] = 'applied'
        self.module.save(self.root / 'next-start-release.json', plan)
        self.assertEqual(switch.resolve_command(host, SimpleNamespace(mode='start')).mode, '2p2d-cache')

    def test_boot_transition_distinguishes_records_from_processes(self):
        """无运行容器可直接接管；有部分恢复进程必须等待，不能根据旧记录强停。"""
        model = {'replicas': 1, 'roles': [{'replicas': 2}, {'replicas': 2}, {'replicas': 1}]}
        items = [{'state': 'running'}] * 5
        self.assertEqual(self.module.boot_transition(model, items, False), 'empty')
        self.assertEqual(self.module.boot_transition(model, items[:1], True), 'wait')
        self.assertEqual(self.module.boot_transition(model, items, True), 'ready')

    def test_boot_without_model_processes_can_clear_stale_records(self):
        """新开机、事务绑定、模型容器确实为空三项同时满足才允许跳过排空。"""
        from test_switch import FakeHost, switch
        host = FakeHost(self.root, '1p1d-cache')
        args = SimpleNamespace(mode='2p2d-cache', drain_timeout=0, restart=True, operation='switch')
        job = switch.prepare_job(host, args)
        task = switch.Switch(host, job)
        original = host.containers
        host.containers = lambda: [c for c in original() if not task.owned_model(c)]
        task.drain()
        self.assertEqual(len(task.instances()), 3)

    def test_same_boot_with_empty_processes_cannot_bypass_drain(self):
        """同一开机周期的维护仍须按引擎状态排空，不借缺失容器掩盖未知队列。"""
        from test_switch import FakeHost, switch
        host = FakeHost(self.root, '1p1d-cache')
        plan = self.state()
        plan['prepared_boot_id'] = host.boot_id()
        self.module.save(self.root / 'next-start-release.json', plan)
        args = SimpleNamespace(mode='2p2d-cache', drain_timeout=0, restart=True, operation='switch')
        job = switch.prepare_job(host, args)
        task = switch.Switch(host, job)
        original = host.containers
        host.containers = lambda: [c for c in original() if not task.owned_model(c)]
        with self.assertRaisesRegex(switch.SwitchError, '排空超时'):
            task.drain()

    def test_new_boot_with_model_processes_still_requires_drain(self):
        """存在模型容器时不能借开机状态跳过真实队列核对。"""
        from test_switch import FakeHost, switch
        host = FakeHost(self.root, '1p1d-cache')
        args = SimpleNamespace(mode='2p2d-cache', drain_timeout=0, restart=True, operation='switch')
        job = switch.prepare_job(host, args)
        task = switch.Switch(host, job)
        self.assertTrue(any(task.owned_model(c) for c in host.containers()))
        with self.assertRaisesRegex(switch.SwitchError, '排空超时'):
            task.drain()

    def test_stopped_service_stays_stopped(self):
        self.module.save(self.root / 'model-state.json', {'running': False})
        self.assertFalse(self.module.boot_should_apply(self.root, 'new-boot'))

    def test_success_is_bound_to_one_job(self):
        self.module.bind(self.root, 'job-a')
        self.module.complete(self.root, True, 'job-b')
        self.assertEqual(self.state()['phase'], 'activating')
        self.module.complete(self.root, True, 'job-a')
        self.assertEqual(self.state()['phase'], 'applied')
        self.module.bind(self.root, 'job-b')
        self.module.complete(self.root, False, 'job-b')
        self.assertEqual((self.root / 'example.json').read_bytes(), self.new)

    def test_failure_restores_exact_old_files(self):
        self.module.bind(self.root, 'job-a')
        self.module.complete(self.root, False, 'job-a')
        self.assertEqual((self.root / 'example.json').read_bytes(), self.old)
        self.assertEqual(self.state()['phase'], 'failed-restored')
        self.assertFalse(self.module.boot_should_apply(self.root, 'new-boot'))
        self.module.complete(self.root, False, 'job-a')

    def test_partial_restore_can_resume(self):
        self.module.bind(self.root, 'job-a')
        state = self.state()
        state['phase'] = 'restoring'
        self.module.save(self.root / 'next-start-release.json', state)
        (self.root / 'example.json').write_bytes(self.old)
        self.module.complete(self.root, False, 'job-a')
        self.assertEqual(self.state()['phase'], 'failed-restored')

    def test_foreign_modification_is_preserved(self):
        self.module.bind(self.root, 'job-a')
        foreign = b'foreign maintenance\n'
        (self.root / 'example.json').write_bytes(foreign)
        with self.assertRaisesRegex(RuntimeError, '其他维护'):
            self.module.complete(self.root, False, 'job-a')
        self.assertEqual((self.root / 'example.json').read_bytes(), foreign)

    def test_damaged_backup_refuses_activation(self):
        (self.root / 'baks/prepared/files/example.json').write_bytes(b'damaged')
        with self.assertRaisesRegex(RuntimeError, '备份身份'):
            self.module.bind(self.root, 'job-a')
        self.assertEqual(self.state()['phase'], 'prepared')

    def test_paths_cannot_escape_installation(self):
        for name in ('../outside', '/outside'):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.module.safe(self.root, name)

    def test_formal_switch_failure_restores_prepared_files(self):
        """实际切换执行器的失败分支必须调用文件恢复，不能只恢复模型快照。"""
        from test_switch import FakeHost, switch
        host = FakeHost(self.root, '1p1d-cache', fail='after-model-start')
        args = SimpleNamespace(mode='2p2d-cache', drain_timeout=0, restart=True, operation='switch')
        job = switch.prepare_job(host, args)
        task = switch.Switch(host, job)
        self.assertTrue(task.state['require_drained'])
        task.drain = lambda: None
        task.execute()
        self.assertEqual(task.state['phase'], 'rolled-back')
        self.assertEqual(self.state()['phase'], 'failed-restored')
        self.assertEqual((self.root / 'example.json').read_bytes(), self.old)
        self.assertNotIn(switch.MARKER, host.active_nginx)

    def test_formal_switch_success_marks_release_applied(self):
        """成功只登记当前升级事务；正常后续方案切换不重放历史升级。"""
        from test_switch import FakeHost, switch
        host = FakeHost(self.root, '1p1d-cache')
        args = SimpleNamespace(mode='2p2d-cache', drain_timeout=0, restart=True, operation='switch')
        job = switch.prepare_job(host, args)
        task = switch.Switch(host, job)
        task.drain = lambda: None
        task.execute()
        self.assertEqual(task.state['phase'], 'committed')
        self.assertEqual(self.state()['phase'], 'applied')
        self.assertEqual((self.root / 'example.json').read_bytes(), self.new)

    def add_export(self):
        """把正式切换会更新的文件纳入预备清单，复现发布和导出共用文件的情况。"""
        name = 'pd-model-stopped.yaml'
        (self.root / 'baks/prepared/files' / name).write_bytes(self.old)
        (self.root / name).write_bytes(self.new)
        plan = self.state()
        plan['files'][name] = {'backup': 'baks/prepared/files/' + name,
            'before_sha256': hashlib.sha256(self.old).hexdigest(),
            'after_sha256': hashlib.sha256(self.new).hexdigest()}
        self.module.save(self.root / 'next-start-release.json', plan)
        return name

    def test_registered_export_is_accepted_for_success(self):
        name = self.add_export()
        self.module.bind(self.root, 'job-a')
        content = b'generated runtime export\n'
        self.module.register_exports(self.root, 'job-a', {name: content})
        # 登记先于实际写入，两种合法中断位置都必须能核验。
        self.module.validated_files(self.root, self.state())
        (self.root / name).write_bytes(content)
        self.module.complete(self.root, True, 'job-a')
        self.assertEqual(self.state()['phase'], 'applied')

    def test_partial_export_can_restore_original_files(self):
        name = self.add_export()
        self.module.bind(self.root, 'job-a')
        content = b'generated runtime export\n'
        self.module.register_exports(self.root, 'job-a', {name: content})
        (self.root / name).write_bytes(content)
        self.module.complete(self.root, False, 'job-a')
        self.assertEqual((self.root / name).read_bytes(), self.old)
        self.assertEqual((self.root / 'example.json').read_bytes(), self.old)

    def test_export_registration_does_not_accept_foreign_changes(self):
        name = self.add_export()
        self.module.bind(self.root, 'job-a')
        self.module.register_exports(self.root, 'job-a', {name: b'authorized export\n'})
        (self.root / name).write_bytes(b'foreign maintenance\n')
        with self.assertRaisesRegex(RuntimeError, '其他维护'):
            self.module.complete(self.root, False, 'job-a')
        self.assertEqual((self.root / name).read_bytes(), b'foreign maintenance\n')

    def test_export_registration_checks_owner_and_whitelist(self):
        name = self.add_export()
        self.module.bind(self.root, 'job-a')
        with self.assertRaisesRegex(RuntimeError, '激活事务'):
            self.module.register_exports(self.root, 'job-b', {name: b'export'})
        with self.assertRaisesRegex(RuntimeError, '未允许'):
            self.module.register_exports(self.root, 'job-a', {'example.json': b'export'})

    def test_formal_switch_tracks_generated_exports_before_completion(self):
        """完整执行器必须接受其自身导出；仅用独立example文件无法覆盖这条调用链。"""
        from test_switch import FakeHost, switch
        self.add_export()
        host = FakeHost(self.root, '2p2d-cache')
        args = SimpleNamespace(mode=switch.FP8_MODE, drain_timeout=0, restart=True, operation='switch')
        job = switch.prepare_job(host, args)
        task = switch.Switch(host, job)
        task.drain = lambda: None
        task.execute()
        self.assertEqual(task.state['phase'], 'committed')
        self.assertEqual(self.state()['phase'], 'applied')
        self.assertEqual(self.state()['files']['pd-model-stopped.yaml']['export_job'], job.name)
