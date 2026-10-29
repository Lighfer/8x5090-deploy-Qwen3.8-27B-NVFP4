#!/usr/bin/env python3
"""只读分析固定FP8模型的尺度与页面布局；不加载权重、不导入Torch、不使用GPU。"""
import argparse
import hashlib
import json
from math import ceil, lcm
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from repository_common import ROOT, outside_repository, output_json, read_json


def page_sizes(text, block_size, *, tp=4, speculative_tokens=3):
    """重放固定版本的GDN形状和混合缓存页面对齐公式。

    GDN卷积历史包含草稿窗口，时间状态保留BF16；FP8只用于普通注意力KV。
    FlashInfer GDN检查点按64个token对齐。管理块可以拆为64-token内核块，
    所以不能把管理块1024误判为底层只支持16/32/64而不可使用。
    返回值只描述页面字节，不包含激活、图捕获、内存池和全部混合分组开销。
    """
    if type(block_size) is not int or block_size <= 0:
        raise ValueError('块大小必须是正整数')
    if any(text[key] % tp for key in ('num_key_value_heads', 'linear_num_key_heads', 'linear_num_value_heads')):
        raise ValueError('模型头数不能按指定TP完整切分')
    conv_channels = (2 * text['linear_num_key_heads'] * text['linear_key_head_dim']
                     + text['linear_num_value_heads'] * text['linear_value_head_dim']) // tp
    conv_bytes = conv_channels * (text['linear_conv_kernel_dim'] - 1 + speculative_tokens) * 2
    temporal_bytes = (text['linear_num_value_heads'] // tp * text['linear_key_head_dim']
                      * text['linear_value_head_dim'] * 2)
    state_bytes = conv_bytes + temporal_bytes
    bytes_per_token = 2 * (text['num_key_value_heads'] // tp) * text['head_dim']
    alignment = lcm(max(16, block_size), 64)
    effective = alignment * ceil(state_bytes / (alignment * bytes_per_token))
    effective = max(block_size, effective)
    page_bytes = effective * bytes_per_token
    return {'requested_block_size': block_size, 'effective_block_size': effective,
            'conv_state_bytes': conv_bytes, 'temporal_state_bytes': temporal_bytes,
            'state_bytes': state_bytes, 'page_bytes': page_bytes,
            'padding_bytes': page_bytes - state_bytes,
            'padding_fraction_of_page': (page_bytes - state_bytes) / page_bytes,
            'kernel_block_size': 64, 'kernel_blocks_per_manager_block': effective // 64}


def layout_report(config, preprocessor, index):
    """使用模型元数据计算候选布局；默认尺度回退与业务精度验收分别报告。"""
    text, vision = config['text_config'], config['vision_config']
    quant = config.get('quantization_config', {})
    if quant.get('quant_method') != 'fp8' or quant.get('weight_block_size') != [128, 128]:
        raise ValueError('只分析固定官方分组FP8格式，不能重写其他量化权重')
    if text.get('mtp_num_hidden_layers') != 1 or text['layer_types'].count('full_attention') != 16:
        raise ValueError('主模型或MTP结构变化，必须重新审查布局公式')
    scales = sorted(n for n in index['weight_map']
                    if re.search(r'(?:[qkv]_scale|kv_scale|[qkv]_proj\.output_scale)$', n))
    # 有校准张量的派生检查点必须另建来源和精度验收记录，不能沿用默认1.0结论。
    if scales:
        raise ValueError('检查点已含尺度张量，不能把它标记为默认静态1.0')
    native = text['max_position_embeddings']
    image_factor = vision['patch_size'] * vision['spatial_merge_size']
    image_tokens = 1048576 // image_factor ** 2
    return {'format': 1, 'verification_level': 'metadata_and_source_math_only',
            'kv_scales': {'policy': 'vllm_static_default', 'k': 1.0, 'v': 1.0,
                          'calibrated': False, 'business_accuracy_verified': False},
            'tensor_parallel_size': 4, 'speculative_tokens': 3,
            'mamba_state_dtype': 'bfloat16',
            'candidate_pages': [page_sizes(text, size) for size in (768, 896, 1024, 1536)],
            'selected_block_size': 1024, 'selected_cpu_chunk_size': 1024,
            'common_max_model_len': native,
            'context_policy': 'same_fixed_native_limit_with_startup_capacity_check',
            'images': {'limit': 64, 'processor_max_pixels': 1048576,
                       'maximum_tokens_per_image': image_tokens,
                       'maximum_tokens_for_64_images': 64 * image_tokens,
                       'remaining_for_text_output_and_wrappers': native - 64 * image_tokens,
                       'checkpoint_default_max_pixels': preprocessor['size']['longest_edge']},
            'gpu_startup_verified': False, 'gpu_oom_freedom_verified': False,
            'capacity_note': '实际GPU池还受权重、MTP、视觉编码、工作区与分组影响；只在启动容量和运行参数检查通过后恢复入口。'}


def analyze(directory):
    """只读校验少量固定元数据；不扫描几十GiB分片，避免与业务争用磁盘。"""
    directory = outside_repository(directory)
    locked = read_json(ROOT / 'locks/model-downloads-fp8.json')
    rows = {row['path']: row for row in locked['files']}
    values = {}
    for name in ('config.json', 'preprocessor_config.json', 'model.safetensors.index.json'):
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 ** 2:
            raise ValueError('元数据不存在、过大或为符号链接：' + name)
        content = path.read_bytes()
        if (len(content) != rows[name]['bytes']
                or hashlib.sha256(content).hexdigest() != rows[name]['sha256']):
            raise ValueError('固定模型元数据身份不符：' + name)
        values[name] = json.loads(content)
    return layout_report(values['config.json'], values['preprocessor_config.json'],
                         values['model.safetensors.index.json'])


def main():
    """模型与可选结果均由调用者提供外部路径，公开仓库不收集现场资料。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    destination = outside_repository(args.output, may_exist=False) if args.output else None
    report = analyze(args.model_directory)
    if destination:
        output_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
