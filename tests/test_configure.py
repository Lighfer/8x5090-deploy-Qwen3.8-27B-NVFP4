"""验证外部输入、全部模板、编号适配及不覆盖保护；不访问真实服务。"""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from support import ROOT, configure, fake_site, fake_images


class ConfigureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='deployment-input-tests-')
        self.base = Path(self.temporary.name)
        self.site = fake_site(self.base)
        self.path = self.base / 'site-private.json'
        self.store()

    def tearDown(self):
        self.temporary.cleanup()

    def store(self):
        self.path.write_text(json.dumps(self.site), encoding='utf-8')

    def test_missing_environment_cannot_generate(self):
        self.path.write_text((ROOT / 'config/site.example.json').read_text())
        with self.assertRaises((TypeError, ValueError)):
            configure.prepare(self.path, execute=True, image_reader=fake_images)
        self.assertFalse(Path(self.site['paths']['deployment']).exists())

    def test_preview_is_read_only(self):
        result = configure.prepare(self.path)
        self.assertTrue(result['dry_run'])
        self.assertFalse(Path(self.site['paths']['deployment']).exists())

    def test_two_different_environments_render_every_python_template(self):
        for shifted in (False, True):
            with self.subTest(shifted=shifted):
                site = fake_site(self.base, shifted=shifted)
                site['paths']['deployment'] = str(self.base / ('install-' + str(shifted)))
                private = self.base / ('site-' + str(shifted) + '.json')
                private.write_text(json.dumps(site))
                configure.prepare(private, execute=True, image_reader=fake_images)
                target = Path(site['paths']['deployment'])
                for path in target.rglob('*.py'):
                    ast.parse(path.read_text(), filename=path.name)
                for path in target.rglob('*'):
                    if path.is_file():
                        self.assertNotIn('@@', path.read_text(), path.name)
                config = json.loads((target / 'switch-config.json').read_text())
                self.assertEqual(config['entry_url'], 'http://' + site['host']['address'] + ':' + str(site['ports']['inference']))
                self.assertEqual(config['hardware_gpu_ids'], sorted(site['gpus']['eight_card_order']))
                self.assertEqual(config['initial_mode'], site['default_mode'])
                for mode in configure.MODES:
                    model = json.loads((target / 'switch-profiles' / (mode + '.json')).read_text())
                    self.assertEqual(model['local_path'], site['paths']['model_directory'])
                    self.assertEqual(model['worker_selector']['node'], site['names']['worker_label'])
                    self.assertEqual(model['cluster_name'], site['names']['cluster'])
                    self.assertEqual(model['image_name'], site['images']['runtime'])
                nginx = (target / 'entrypoint/nginx.conf').read_text()
                self.assertIn('listen ' + str(site['ports']['inference']) + ';', nginx)

    def test_registration_preserves_token_and_runtime_files(self):
        configure.prepare(self.path, execute=True, image_reader=fake_images)
        target = Path(self.site['paths']['deployment'])
        env = (target / '.env').read_text() + 'TOKEN_FIXTURE=' + os.urandom(12).hex() + '\n'
        (target / '.env').write_text(env)
        self.site['identity'] = {'model_id': 47, 'worker_id': 83, 'cluster_id': 19}
        self.store()
        configure.prepare(self.path, execute=True, configure_only=True, image_reader=fake_images)
        self.assertEqual((target / '.env').read_text(), env)
        config = json.loads((target / 'switch-config.json').read_text())
        self.assertEqual(config['model_id'], 47)
        self.assertIn('model_id=47', (target / 'verify-live-parameters.py').read_text())
        self.assertIn('worker_id=83', (target / 'entrypoint/metrics-bridge.py').read_text())
        self.assertEqual(json.loads((target / 'cache-service.json').read_text())['cluster_id'], 19)

    def test_modified_generated_file_is_rejected_before_write(self):
        configure.prepare(self.path, execute=True, image_reader=fake_images)
        target = Path(self.site['paths']['deployment'])
        (target / 'verify-deployment.py').write_text('# changed\n')
        before = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob('*') if p.is_file()}
        self.site['identity'] = {'model_id': 9, 'worker_id': 8, 'cluster_id': 7}
        self.store()
        with self.assertRaises(ValueError):
            configure.prepare(self.path, execute=True, configure_only=True, image_reader=fake_images)
        after = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_gpu_duplicates_and_port_overlap_are_rejected(self):
        for field in ('gpu', 'port'):
            with self.subTest(field=field):
                site = copy.deepcopy(self.site)
                if field == 'gpu':
                    site['gpus']['six_card_order'][0] = site['gpus']['six_card_order'][1]
                else:
                    site['ports']['management'] = site['ports']['inference']
                self.path.write_text(json.dumps(site))
                with self.assertRaises(ValueError):
                    configure.prepare(self.path)

    def test_image_failure_happens_before_target_creation(self):
        def fail(site):
            raise RuntimeError('missing-image')
        with self.assertRaises(RuntimeError):
            configure.prepare(self.path, execute=True, image_reader=fail)
        self.assertFalse(Path(self.site['paths']['deployment']).exists())

    def test_unknown_fields_are_rejected(self):
        self.site['guessed_default'] = True
        self.store()
        with self.assertRaises(ValueError):
            configure.prepare(self.path)

    def test_missing_numa_mapping_keeps_binding_disabled(self):
        result = configure.generated_files(self.site, Path(self.site['paths']['deployment']), fake_images(self.site))
        for mode in configure.MODES:
            profile = json.loads(result['switch-profiles/' + mode + '.json'])
            for scope in [profile, *profile['roles']]:
                self.assertNotIn('--numa-bind', scope.get('backend_parameters', []))

    def test_explicit_numa_mapping_binds_only_homogeneous_role_pools(self):
        order = self.site['gpus']['eight_card_order']
        self.site['gpus']['numa_nodes'] = {str(g): index // 4 for index, g in enumerate(order)}
        self.store()
        configure.checked_site(self.path)
        result = configure.generated_files(self.site, Path(self.site['paths']['deployment']), fake_images(self.site))
        profile = json.loads(result['switch-profiles/2p2d-cache.json'])
        for role in profile['roles']:
            if role['name'] in ('prefill', 'decode'):
                args = role['backend_parameters']
                start = args.index('--numa-bind-nodes') + 1
                expected = str(self.site['gpus']['numa_nodes'][role['gpu_selector']['gpu_ids'][0].rsplit(':', 1)[1]])
                self.assertEqual(args[start:], [expected] * 2)
        six = json.loads(result['switch-profiles/1p2d-6gpu-cache.json'])
        for role in six['roles']:
            selected = (role.get('gpu_selector') or {}).get('gpu_ids', [])
            nodes = {self.site['gpus']['numa_nodes'][g.rsplit(':', 1)[1]] for g in selected}
            if len(nodes) > 1:
                self.assertNotIn('--numa-bind', role['backend_parameters'])

    def test_partial_numa_mapping_is_rejected(self):
        self.site['gpus']['numa_nodes'] = {}
        self.store()
        with self.assertRaises(ValueError):
            configure.checked_site(self.path)

    def test_child_symlink_is_rejected_before_registration_writes(self):
        configure.prepare(self.path, execute=True, image_reader=fake_images)
        target = Path(self.site['paths']['deployment'])
        destination = self.base / 'must-remain-unmodified.txt'
        destination.write_text('unchanged', encoding='utf-8')
        (target / '.env').unlink()
        (target / '.env').symlink_to(destination)
        self.site['identity'] = {'model_id': 47, 'worker_id': 83, 'cluster_id': 19}
        self.store()
        with self.assertRaises(ValueError):
            configure.prepare(self.path, execute=True, configure_only=True, image_reader=fake_images)
        self.assertEqual(destination.read_text(), 'unchanged')

    def test_parent_symlink_cannot_redirect_private_output(self):
        original = configure.ROOT
        from repository_common import outside_repository
        alias = self.base / 'source-alias'
        alias.symlink_to(original, target_is_directory=True)
        with self.assertRaises(ValueError):
            outside_repository(alias / 'should-not-exist.json', may_exist=False)
        self.assertFalse((original / 'should-not-exist.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
