"""验证缓存可见卡组的契约，兼容正式回滚，且拒绝漏P卡或侵占预留卡。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from support import generated_fixture


class CacheScopeTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='cache-scope-tests-')
        root=Path(self.temporary.name)/'private'
        generated_fixture(root)
        spec=importlib.util.spec_from_file_location('live_parameters',root/'verify-live-parameters.py')
        self.module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.allowed=[17,31,47,59,71,83]
        self.prefill=[71,83]

    def tearDown(self):
        self.temporary.cleanup()

    def service(self, values):
        return {'config': {'env': {'CUDA_VISIBLE_DEVICES': ','.join(map(str,values))}}}

    def test_prefill_only_scope_is_accepted(self):
        self.assertEqual(self.module.cache_gpu_scope(self.service(self.prefill),self.allowed,self.prefill),self.prefill)

    def test_original_full_scope_remains_valid_for_rollback(self):
        self.assertEqual(self.module.cache_gpu_scope(self.service(self.allowed),self.allowed,self.prefill),self.allowed)

    def test_reserved_missing_and_duplicate_gpu_are_rejected(self):
        for values in ([71],[71,83,97],[71,83,83],[]):
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                self.module.cache_gpu_scope(self.service(values),self.allowed,self.prefill)
