#!/usr/bin/env python3
"""用镜像原调度器和多连接器验证零预算时的缓存进度，不执行GPU推理。"""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
from transformers import OPTConfig
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lookup_progress_cpu', HERE / 'verify-cpu.py')
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)
init_none_hash(sha256)
HASHER = get_request_block_hasher(1536, sha256)


class Connector:
    """计算入口仅替换外部KV导入；元数据路径实际经过原MultiConnector及LMCache。"""

    supports_divergent_local_hybrid_hits = False
    requires_kv_delivery = False

    def __init__(self):
        """不调用会连接真实服务的构造函数，只配置本实验访问的对象字段。"""
        self.adapter = cpu.adapter(cpu.installed_root())
        self.calls = []
        self.allocations = []
        child = object.__new__(LMCacheMPConnector)
        child.scheduler_adapter = self.adapter
        child.lazy_offload = False
        child._process_retrieve_requests = lambda *_: None
        child._process_new_requests = lambda *_: None
        child._process_cached_requests = lambda *_: None
        child._report_block_allocation_deltas = lambda *_: None
        self.multi = object.__new__(MultiConnector)
        self.multi._connectors = [child]
        self.multi._extra_async_saves = {}

    def on_new_request(self, request):
        """已提交查询的元数据由适配器夹具提供，不发出网络请求。"""
        return None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """结果就绪后返回零外部导入，隔离计算准入与GPU传输行为。"""
        self.calls.append(request.request_id)
        if request.request_id != 'hot':
            return 0, False
        value = self.adapter.check_lookup_result('hot')
        return (None, False) if value is None else (0, False)

    def update_state_after_alloc(self, request, *args):
        """记录是否发生GPU块准入，不实际分配设备存储。"""
        self.allocations.append(request.request_id)

    def build_connector_meta(self, output):
        """调用原多连接器转发，再进入打补丁后的原LMCache元数据方法。"""
        return self.multi.build_connector_meta(output)

    def has_pending_block_frees(self):
        """实验没有真实外部块释放任务。"""
        return False

    def has_pending_push_work(self):
        """实验没有NIXL推送工作，保持原拉取路径的调度条件。"""
        return False


def request(name, size, token):
    """只构造合成token和调度字段，不读取模型权重、提示词或业务数据。"""
    params = SamplingParams(max_tokens=1)
    params.update_from_generation_config({}, 2)
    return Request(request_id=name, prompt_token_ids=[token] * size, sampling_params=params,
                   pooling_params=None, block_hasher=HASHER)


def experiment(model_dir, threshold, cold_count, asynchronous, waiting_hot=True):
    """先挂起缓存请求，再让冷长请求占预算，核验消息推进与计算准入各自的条件。"""
    model = ModelConfig(model=str(model_dir), dtype='float16', skip_tokenizer_init=True, max_model_len=204800)
    cache = CacheConfig(block_size=1536, enable_prefix_caching=True, mamba_cache_mode='align')
    cache.num_gpu_blocks = 1050
    config = VllmConfig(model_config=model, cache_config=cache, device_config=DeviceConfig(device='cpu'),
                       scheduler_config=SchedulerConfig(max_num_seqs=6, max_num_batched_tokens=9216,
                           max_model_len=204800, enable_chunked_prefill=True, is_encoder_decoder=False,
                           long_prefill_token_threshold=threshold, long_prefill_token_threshold_adaptive=True,
                           watermark=0.0, async_scheduling=asynchronous))
    kv = KVCacheConfig(num_blocks=1050, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(['attention'], FullAttentionSpec(block_size=1536, num_kv_heads=1, head_size=1, dtype=torch.float32)),
        KVCacheGroupSpec(['mamba'], MambaSpec(block_size=1536, shapes=((1, 1),), dtypes=(torch.float32,), mamba_cache_mode='align'))])
    register_all_kvcache_specs(config)
    scheduler = (AsyncScheduler if asynchronous else Scheduler)(vllm_config=config, kv_cache_config=kv,
        structured_output_manager=StructuredOutputManager(config), block_size=1536, hash_block_size=1536, log_stats=True)
    # 非并行MTP在该预算分支不增加额外输入占位；不执行草稿生成或验证。
    scheduler.vllm_config.speculative_config = SimpleNamespace(max_num_new_slots_for_drafting=0)
    connector = Connector()
    scheduler.connector = connector
    if waiting_hot:
        scheduler.add_request(request('hot', 2000, 31))
        scheduler.schedule()
    else:
        connector.adapter.cleanup_lookup_result('hot')
    for index in range(cold_count):
        scheduler.add_request(request('cold' + str(index), 100000, index + 1))
    scheduler.schedule()
    cpu.acknowledge(connector.adapter)
    connector.calls.clear()
    connector.allocations.clear()
    # 此轮发出状态查询，下一轮元数据回调观察最终回复；均不能依赖等待队列准入。
    before = scheduler.schedule()
    if waiting_hot:
        future = connector.adapter.req_clients['one'].queries[0][1]
        future.value, future.ready = 20, True
    connector.calls.clear()
    connector.allocations.clear()
    output = scheduler.schedule()
    result = {'threshold': threshold, 'cold_running': cold_count, 'async': asynchronous,
              'waiting_hot': waiting_hot, 'tokens': output.num_scheduled_tokens,
              'admission_checked': 'hot' in connector.calls, 'hot_allocated': 'hot' in connector.allocations,
              'result_ready': 'hot' in connector.adapter._finished_lookup_results,
              'cache_locks_freed': len(connector.adapter.freed_tails),
              'gpu_initialized': torch.cuda.is_initialized()}
    if waiting_hot:
        assert result['result_ready'] and result['cache_locks_freed'] == 0
        if threshold == 3072 and cold_count == 3:
            assert not result['admission_checked'] and not result['hot_allocated']
            assert output.total_num_scheduled_tokens == 9216
        if threshold == 1536:
            # 元数据回调已经缓存结果后，下一轮必须能正常准入；此前没有提前分配块。
            final = scheduler.schedule()
            assert 'hot' in final.num_scheduled_tokens or 'hot' in output.num_scheduled_tokens
    else:
        assert output.num_scheduled_tokens == {'cold0': 9216}
    return result


def main():
    """生成临时合成模型配置；全部验证在CPU元数据层完成，不初始化CUDA。"""
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    with tempfile.TemporaryDirectory(prefix='lookup-progress-scheduler-') as temporary:
        model = Path(temporary)
        OPTConfig(vocab_size=128, hidden_size=64, ffn_dim=128, num_hidden_layers=2,
                  num_attention_heads=2, word_embed_proj_dim=64, max_position_embeddings=204800,
                  architectures=['OPTForCausalLM']).save_pretrained(model)
        rows = [experiment(model, threshold, count, asynchronous) for asynchronous in [False, True]
                for threshold in [3072, 1536] for count in [1, 2, 3]]
        controls = [experiment(model, threshold, 1, asynchronous, False) for asynchronous in [False, True]
                    for threshold in [3072, 1536]]
    assert not torch.cuda.is_initialized()
    print('SCHEDULER_RESULT=' + json.dumps({'passed': True, 'cases': rows, 'controls': controls,
                                           'gpu_inference_verified': False}))


if __name__ == '__main__':
    main()
