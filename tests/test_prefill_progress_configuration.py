"""确保生成配方保留独立的P分段规则，且完整构建不会遗漏缓存推进修复。"""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent


class PrefillConfigurationChecks(unittest.TestCase):
    """直接读取正式配方，防止配置生成或构建顺序与说明不一致。"""

    def test_all_pd_prefills_keep_adaptive_threshold_and_decode_has_no_override(self):
        """阈值属于P，不把分段设置误加到D，也不删除原并发与批预算。"""
        for path in (ROOT / 'recipes').glob('*.json'):
            value = json.loads(path.read_text(encoding='utf-8'))
            for role in value.get('roles', []):
                parameters = role.get('backend_parameters', [])
                options = [p for p in parameters if p.startswith('--long-prefill-token-threshold')]
                if role['name'] == 'prefill':
                    self.assertEqual(options, ['--long-prefill-token-threshold=1536',
                                               '--long-prefill-token-threshold-adaptive'], path.name)
                    self.assertTrue(any(p.startswith('--max-num-seqs=') for p in parameters))
                    self.assertTrue(any(p.startswith('--max-num-batched-tokens=') for p in parameters))
                elif role['name'] == 'decode':
                    self.assertEqual(options, [], path.name)

    def test_full_build_applies_python_fix_after_native_extension_build(self):
        """后续安装原LMCache源码不能覆盖已应用的Python差异。"""
        text = (ROOT / 'sources/vllm/Dockerfile').read_text(encoding='utf-8')
        self.assertLess(text.index('build-lmcache-cuda.py'), text.index('lookup-progress/apply-patch.py'))
        self.assertIn('lookup-progress/verify-cpu.py', text)

    def test_stack_points_to_reviewable_local_manifest(self):
        """本地修改单独登记，不冒充上游提交或GPU性能验收。"""
        stack = json.loads((ROOT / 'STACK.json').read_text(encoding='utf-8'))
        entry = stack['lookup_progress_fix']
        manifest = json.loads((ROOT / entry['manifest']).read_text(encoding='utf-8'))
        self.assertEqual(entry['name'], manifest['release'])
        self.assertFalse(entry['gpu_performance_verified'])
        self.assertEqual(len(manifest['final_files']), 2)


if __name__ == '__main__':
    unittest.main()
