"""验证扩展检查能够读取事务路径，且多图探测包含真实独立图片。"""
import base64
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

from support import generated_fixture


class ActivationValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='activation-check-')
        self.root = Path(self.temp.name) / 'private'
        generated_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def load(self, name):
        """加载已生成的独立验证器，避免把带占位符的模板当作实际脚本。"""
        spec = importlib.util.spec_from_file_location(name.replace('-', '_'), self.root / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_extended_job_keeps_path_type_and_rejects_open_gate(self):
        """必须先成功读取Path状态，再以未隔离为由停止；路径错误会使本断言失败。"""
        module = self.load('verify-v032-migration')
        job = self.root / 'synthetic-job'
        job.mkdir()
        (job / 'state.json').write_text(json.dumps({'gate_token': 'synthetic-only'}))
        with mock.patch.object(sys, 'argv', ['verify', '--job', str(job)]), \
                mock.patch.object(module, 'request', return_value=(200, '{}')), \
                self.assertRaisesRegex(RuntimeError, '入口隔离'):
            module.main()

    def test_64_images_are_distinct_and_have_declared_dimensions(self):
        """用小尺寸验证数据结构和PNG头；真实百万像素负载只在隔离GPU验收中发送。"""
        module = self.load('verify-model-features')
        images = module.image_messages(64, 32)[0]['content'][1:]
        urls = [image['image_url']['url'] for image in images]
        self.assertEqual(len(images), 64)
        self.assertEqual(len(set(urls)), 64)
        for value in urls:
            png = base64.b64decode(value.split(',', 1)[1])
            self.assertEqual(struct.unpack('>II', png[16:24]), (32, 32))
