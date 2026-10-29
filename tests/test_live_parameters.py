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

    def test_fp8_runtime_configuration_requires_actual_metrics(self):
        """普通接口与引擎指标必须同时匹配；缺失证据不能按成功处理。"""
        expected = {'--max-model-len': '262144', '--block-size': '1024', '--mamba-block-size': '1024'}
        cards = {'data': [{'id': 'synthetic-model', 'max_model_len': 262144}]}
        metric = ('vllm:cache_config_info{block_size="1024",mamba_block_size="1024",'
                  'cache_dtype="fp8_e4m3",mamba_ssm_cache_dtype="bfloat16",num_gpu_blocks="512"} 1.0\n')
        result = self.module.fp8_runtime_evidence(expected, 'synthetic-model', cards, metric)
        self.assertTrue(result['passed'])
        for bad in ('', metric.replace('block_size="1024"', 'block_size="1536"'),
                    metric.replace('num_gpu_blocks="512"', 'num_gpu_blocks="0"')):
            with self.subTest(metric=bad), self.assertRaises(RuntimeError):
                self.module.fp8_runtime_evidence(expected, 'synthetic-model', cards, bad)
        cards['data'][0]['max_model_len'] = 204800
        with self.assertRaisesRegex(RuntimeError, '共同固定上限'):
            self.module.fp8_runtime_evidence(expected, 'synthetic-model', cards, metric)
