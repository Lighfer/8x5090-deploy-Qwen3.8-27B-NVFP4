"""验证不同启动形式中的真实权重身份，避免相同对外名称掩盖错误模型目录。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from support import generated_fixture


class ModelIdentityTests(unittest.TestCase):
    """只构造合成命令行，不调用Docker或加载实际模型。"""

    def test_model_path_comes_from_actual_command(self):
        with tempfile.TemporaryDirectory(prefix='model-identity-') as temporary:
            root = Path(temporary)
            generated_fixture(root)
            spec = importlib.util.spec_from_file_location('identity_fixture', root / 'verify-live-parameters.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = str(root / 'fixture-models/fp8')
            self.assertEqual(module.command_model_path(['vllm', 'serve', model, '--port', '12345']), model)
            self.assertEqual(module.command_model_path(['python', '-m', 'api_server', '--model=' + model]), model)
            self.assertIsNone(module.command_model_path(['vllm', 'serve', '--served-model-name', 'same-name']))

    def test_router_log_rotation_requires_complete_live_evidence(self):
        """丢失启动日志时仍需全部健康成员与原生PD模式；冲突日志不得被替代证据掩盖。"""
        with tempfile.TemporaryDirectory(prefix='router-evidence-') as temporary:
            root = Path(temporary)
            generated_fixture(root)
            spec = importlib.util.spec_from_file_location('router_evidence_fixture', root / 'verify-live-parameters.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            flags = {'--enable-igw': True, '--vllm-pd-disaggregation': True, '--kv-connector': 'nixl',
                     '--prefill-policy': 'cache_aware', '--decode-policy': 'round_robin'}
            members = {('prefill', 'http://prefill.invalid'), ('decode', 'http://decode.invalid')}
            workers = [{'worker_type': role, 'url': url, 'is_healthy': True} for role, url in members]
            actual = module.router_evidence(flags, '', {'router_manager': True}, workers, members)
            self.assertTrue(actual['stage_policies_match'])
            self.assertIsNone(actual['actual_stage_policies'])
            self.assertTrue(actual['native_multi_router_mode'])
            self.assertFalse(module.router_evidence(flags, '', {'router_manager': True}, workers[:1], members)['stage_policies_match'])
            self.assertFalse(module.router_evidence(flags, '', {}, workers, members)['native_multi_router_mode'])
            conflicting = 'PD stage policies active: prefill=round_robin, decode=round_robin'
            self.assertFalse(module.router_evidence(flags, conflicting, {'router_manager': True}, workers, members)['stage_policies_match'])
