"""验证静态页面推导与模型精度标识；不使用权重、服务器或GPU。"""
import importlib.util
from pathlib import Path
import unittest

from support import ROOT

spec = importlib.util.spec_from_file_location('fp8_layout', ROOT / 'tools/analyze-fp8-layout.py')
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class FP8LayoutTests(unittest.TestCase):
    def config(self):
        """合成结构使用独立字段，不从现场导入配置或业务输入。"""
        text = {'num_key_value_heads': 4, 'head_dim': 256,
                'linear_num_key_heads': 16, 'linear_num_value_heads': 48,
                'linear_key_head_dim': 128, 'linear_value_head_dim': 128,
                'linear_conv_kernel_dim': 4, 'mtp_num_hidden_layers': 1,
                'layer_types': ['linear_attention'] * 48 + ['full_attention'] * 16,
                'max_position_embeddings': 262144}
        return {'text_config': text, 'vision_config': {'patch_size': 16, 'spatial_merge_size': 2},
                'quantization_config': {'quant_method': 'fp8', 'weight_block_size': [128, 128]}}

    def test_smaller_block_is_raised_when_state_does_not_fit(self):
        """768不能容纳含草稿窗口的状态，不能把请求值当成生效值。"""
        row = layout.page_sizes(self.config()['text_config'], 768)
        self.assertEqual(row['effective_block_size'], 1536)
        self.assertEqual(row['state_bytes'], 423936)

    def test_1024_fits_without_raising_and_reduces_padding(self):
        text = self.config()['text_config']
        old, new = layout.page_sizes(text, 1536), layout.page_sizes(text, 1024)
        self.assertEqual(new['effective_block_size'], 1024)
        self.assertEqual(new['page_bytes'], 524288)
        self.assertEqual(new['padding_bytes'], 100352)
        self.assertEqual(old['page_bytes'] - new['page_bytes'], 262144)
        self.assertEqual(new['kernel_blocks_per_manager_block'], 16)

    def test_default_scales_and_image_budget_are_not_quality_acceptance(self):
        report = layout.layout_report(self.config(), {'size': {'longest_edge': 16777216}},
                                      {'weight_map': {'synthetic.weight': 'synthetic.safetensors'}})
        self.assertFalse(report['kv_scales']['calibrated'])
        self.assertFalse(report['gpu_startup_verified'])
        self.assertEqual(report['kv_scales']['k'], 1.0)
        self.assertEqual(report['common_max_model_len'], 262144)
        self.assertEqual(report['images']['maximum_tokens_for_64_images'], 65536)
        self.assertEqual(report['images']['remaining_for_text_output_and_wrappers'], 196608)

    def test_calibrated_checkpoint_cannot_be_labelled_as_default(self):
        with self.assertRaisesRegex(ValueError, '已含尺度'):
            layout.layout_report(self.config(), {}, {'weight_map': {'model.layers.3.self_attn.attn.k_scale': 'scale.safetensors'}})

    def test_changed_mtp_structure_requires_review(self):
        config = self.config()
        config['text_config']['mtp_num_hidden_layers'] = 2
        with self.assertRaisesRegex(ValueError, '结构变化'):
            layout.layout_report(config, {}, {'weight_map': {}})

    def test_invalid_head_partition_is_rejected(self):
        text = self.config()['text_config']
        text['num_key_value_heads'] = 3
        with self.assertRaisesRegex(ValueError, '切分'):
            layout.page_sizes(text, 1024)
