#!/usr/bin/env python3
"""检查回移补丁的CPU逻辑；不初始化CUDA，也不声称完成GPU推理验收。

从已经打补丁的实际软件文件提取函数，避免导入整个推理服务。GPU写入边界用
CPU张量参考实现替换；补丁中的排队、类型转换、数据回收和回放逻辑保持原样。
"""
import array
import ast
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, replace
from functools import partial
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(importlib.util.find_spec('vllm').origin).parent.parent


def selected(path, names, namespace):
    """仅执行指定定义；完整文件的身份由补丁清单另行严格验证。"""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError('目标定义缺失：' + path)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, 'exec'), namespace)
    return namespace


class CpuBuffer:
    """以CPU存储模拟UVA边界，不替换被测缓冲池的轮换和扩容代码。"""
    def __init__(self, size, dtype):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()

    def uva(self, n=None):
        return self.cpu if n is None else self.cpu[:n]


class CpuWriteKernel:
    """保持写入索引、偏移和累积长度的接口，执行可独立核对的CPU参考写入。"""
    def __getitem__(self, grid):
        def write(target, stride, indices, starts, contents, ends, groups, **options):
            if options['MULTI_GROUP']:
                raise AssertionError('本参考实现仅覆盖单张量写入边界')
            previous = 0
            for index, start, end in zip(indices.tolist(), starts.tolist(), ends.tolist()):
                target[index, start:start + end - previous] = contents[previous:end]
                previous = end
        return write


def staging_namespace():
    """导入实际的类型映射，保持与已固定Torch版本的一致性。"""
    mapping = ast.parse((ROOT / 'vllm/utils/torch_utils.py').read_text())
    namespace = dict(torch=torch, np=np, array=array, partial=partial,
                     _DEFAULT_MAX_CONCURRENCY=2, _WRITE_BLOCK_SIZE=1024,
                     _BUFFER_TYPES=(np.ndarray, array.array), UvaBuffer=CpuBuffer,
                     NonUvaBuffer=CpuBuffer, is_uva_available=lambda: True,
                     _num_write_chunks=lambda n: max((n + 1023) // 1024, 1),
                     _apply_write_kernel=CpuWriteKernel())
    node = next(n for n in mapping.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TORCH_DTYPE_TO_ARRAY_TYPECODE' for t in n.targets))
    namespace['TORCH_DTYPE_TO_ARRAY_TYPECODE'] = eval(compile(ast.Expression(node.value), '<dtype-map>', 'eval'), namespace)
    return selected('vllm/v1/worker/gpu/buffer_utils.py', {'UvaBufferPool', 'StagedWriteTensor'}, namespace)


@dataclass
class Feature:
    """仅保留调度器和编码器缓存实际访问的输入字段。"""
    data: object
    mm_position: object


class BackportChecks(unittest.TestCase):
    """覆盖类型暂存、并发缓冲、混合KV视图和多模态回收后的回放。"""
    def test_staged_mixed_chunks(self):
        ns = staging_namespace()
        value = ns['StagedWriteTensor']((3, 8), torch.int32, torch.device('cpu'))
        data = np.array([7, 8], dtype=np.int64)
        value.stage_write(0, 1, [3, 4])
        value.stage_write(1, 2, data)
        value.stage_write(2, 0, array.array('q', [9, 10]))
        value.stage_write_elem(0, 6)
        value.apply_write()
        self.assertEqual(value.gpu.tolist(), [[6, 3, 4, 0, 0, 0, 0, 0], [0, 0, 7, 8, 0, 0, 0, 0], [9, 10, 0, 0, 0, 0, 0, 0]])
        self.assertEqual(value._staged_write_len, 0)
        value.apply_write()

    def test_live_buffer_clear_and_failed_conversion(self):
        ns = staging_namespace()
        value = ns['StagedWriteTensor']((2, 4), torch.int32, torch.device('cpu'))
        value.stage_write(0, 0, [1, 2])
        view = memoryview(value._staged_write_contents)
        value.clear_staged_writes()
        self.assertEqual(list(view), [1, 2])
        with self.assertRaises((TypeError, OverflowError)):
            value.stage_write(0, 0, [1, 'invalid'])
        self.assertEqual(value._staged_write_len, 0)
        self.assertEqual(value._staged_write_indices, [])
        value.stage_write(1, 0, [4])
        value.apply_write()
        self.assertEqual(value.gpu[1, 0].item(), 4)

    def test_buffer_ring_and_growth(self):
        ns = staging_namespace()
        pool = ns['UvaBufferPool'](1, torch.int32, max_concurrency=2)
        first = pool.copy_chunks_to_uva([np.array([1, 2, 3])], 3)
        second = pool.copy_chunks_to_uva([array.array('i', [8])], 1)
        self.assertEqual(first.tolist(), [1, 2, 3])
        self.assertEqual(second.tolist(), [8])
        self.assertGreaterEqual(pool._uva_bufs[1].cpu.shape[0], 3)

    def test_hybrid_copy_does_not_skip_different_view(self):
        ns = selected('vllm/v1/worker/utils.py', {'copy_kv_cache_blocks_inplace'},
                      dict(torch=torch, np=np, async_tensor_h2d=lambda x, device: torch.as_tensor(x, device=device)))
        storage = torch.arange(64, dtype=torch.int32)
        attention = storage[:16].view(2, 8)
        mamba = attention[:, :4]
        expected = attention[0].clone()
        ns['copy_kv_cache_blocks_inplace']([mamba, mamba, attention], 2, [(0, 1)])
        self.assertTrue(torch.equal(attention[1], expected))

    def test_encoder_payload_release_and_restore(self):
        ns = selected('vllm/v1/worker/gpu/mm/encoder_cache.py', {'EncoderCache'}, dict(torch=torch, replace=replace))
        position = SimpleNamespace(offset=0, length=10)
        original = [Feature({'pixels': [1, 2]}, position)]
        cache = ns['EncoderCache']()
        cache.add_request('request', original)
        cache.update_encoder_inputs('request', {0: None})
        self.assertIsNone(cache.mm_features['request'][0].data)
        self.assertEqual(original[0].data, {'pixels': [1, 2]})
        cache.update_encoder_inputs('request', {0: original[0].data})
        self.assertEqual(cache.mm_features['request'][0].data, original[0].data)
        cache.remove_request('request')
        self.assertEqual(cache.mm_features, {})

    def test_scheduler_confirmed_progress_and_rewind(self):
        tree = ast.parse((ROOT / 'vllm/v1/core/sched/scheduler.py').read_text())
        method = next(n for cls in tree.body if isinstance(cls, ast.ClassDef) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_encoder_input_updates')
        namespace = {}
        exec('from __future__ import annotations\n' + ast.unparse(method), namespace)
        update = namespace['_get_encoder_input_updates']
        request = SimpleNamespace(mm_features=[Feature({'pixels': [1]}, SimpleNamespace(offset=5, length=5)), Feature({'address': 1}, SimpleNamespace(offset=0, length=2))],
                                  num_computed_tokens=20, num_output_placeholders=3, num_in_flight_tokens=12)
        scheduler = SimpleNamespace(use_v2_model_runner=True, _pending_encoder_input_ids={}, requests={'r': request}, is_encoder_decoder=False)
        new = [SimpleNamespace(req_id='r', mm_features=request.mm_features)]
        self.assertEqual(update(scheduler, {'r': 1}, new, {}), {})
        self.assertEqual(scheduler._pending_encoder_input_ids['r'], {0})
        request.num_in_flight_tokens = 0
        self.assertEqual(update(scheduler, {'r': 1}, [], {}), {'r': {0: None}})
        request.num_computed_tokens = 5
        self.assertEqual(update(scheduler, {'r': 1}, [], {'r': [0]}), {'r': {0: {'pixels': [1]}}})
        self.assertEqual(scheduler._pending_encoder_input_ids['r'], {0})
        scheduler.use_v2_model_runner = False
        self.assertEqual(update(scheduler, {'r': 1}, [], {}), {})

    def test_parameter_wrapper_removed(self):
        tree = ast.parse((ROOT / 'vllm/model_executor/parameter.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BasevLLMParameter')
        self.assertNotIn('__torch_function__', [n.name for n in cls.body if isinstance(n, ast.FunctionDef)])

    def test_cache_affinity_helpers(self):
        spec = importlib.util.spec_from_file_location('cache_numa_check', HERE / 'cache-numa.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [dict(pci='00000000:20:00.0', uuid='GPU-b'), dict(pci='00000000:10:00.0', uuid='GPU-a')]
        self.assertEqual(module.select_gpu_rows(rows, '0', 'PCI_BUS_ID')[0]['uuid'], 'GPU-a')
        self.assertEqual(module.select_gpu_rows(rows, 'GPU-b', '')[0]['uuid'], 'GPU-b')
        self.assertEqual(module.select_gpu_rows(rows, '-1', ''), [])
        with self.assertRaises(ValueError):
            module.select_gpu_rows(rows, '0', '')
        self.assertEqual(module.expand_cpu_list('1-3,7'), {1, 2, 3, 7})

    def test_cuda_was_not_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)
