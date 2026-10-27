#!/usr/bin/env python3
"""只用 CPU 验证官方修复的边界；不连接 GPU，不启动模型。"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import types
from typing import Tuple, Union


class Tensor:
    """记录辅助函数的控制流；任何 CUDA 值回读或隐式取值都会使检查失败。"""
    def __init__(self, value=1.0, *, device='cuda:0', dtype='float32', count=1, contiguous=True):
        self.value, self.device, self.dtype = value, device, dtype
        self.count, self.contiguous_value = count, contiguous
        self.is_cuda = device.startswith('cuda')

    def numel(self):
        return self.count

    def is_contiguous(self):
        return self.contiguous_value

    def item(self):
        if self.is_cuda:
            raise AssertionError('发生 CUDA 标量回读')
        return self.value

    def to(self, *, device):
        return Tensor(self.value, device=device, dtype=self.dtype, count=self.count)

    def contiguous(self):
        return self

    def cpu(self):
        raise AssertionError('发生 CUDA 到 CPU 同步拷贝')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/usr/local/lib/python3.12/dist-packages/flashinfer/page.py'))
    args = parser.parse_args()
    lock = json.loads(Path(__file__).with_name('inputs.json').read_text(encoding='utf-8'))
    content = args.input.read_bytes()
    if hashlib.sha256(content).hexdigest() != lock['page_patched_sha256']:
        raise RuntimeError('待验证文件不是固定补丁成品')
    tree = ast.parse(content)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_as_float32_scalar_tensors')
    module = ast.Module(body=[function], type_ignores=[])
    capturing = [False]
    fake_torch = types.SimpleNamespace(Tensor=Tensor, float32='float32', device=str,
                                      tensor=lambda values, **kw: Tensor(values[0], **kw))
    namespace = {'Tuple': Tuple, 'Union': Union, 'torch': fake_torch, 'math': math,
                 '_is_stream_capturing_on_device': lambda device: capturing[0]}
    exec(compile(module, str(args.input), 'exec'), namespace)
    convert = namespace['_as_float32_scalar_tensors']
    checks = []

    def accepted(name, values):
        result = convert(tuple((str(i), value) for i, value in enumerate(values)), device='cuda:0')
        assert len(result) == len(values)
        checks.append(name)

    def rejected(name, value):
        try:
            convert((('scale', value),), device='cuda:0')
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError('未拒绝非法输入：' + name)

    accepted('CUDA正常比例因子无需CPU回读', [Tensor(), Tensor(2.0)])
    accepted('CUDA非法数值交由既有内核验证', [Tensor(0.0), Tensor(float('nan'))])
    accepted('普通浮点数与CPU标量仍可使用', [1.0, Tensor(2.0, device='cpu')])
    for bad in (0.0, -1.0, float('nan'), float('inf')):
        rejected('拒绝非法普通浮点数', bad)
        rejected('拒绝非法CPU标量', Tensor(bad, device='cpu'))
    rejected('拒绝非标量张量', Tensor(count=2))
    rejected('拒绝错误类型', Tensor(dtype='float16'))
    capturing[0] = True
    accepted('图捕获继续接受同设备连续float32标量', [Tensor()])
    for name, value in [('图捕获拒绝普通浮点数', 1.0), ('图捕获拒绝CPU值', Tensor(device='cpu')),
                        ('图捕获拒绝其他设备', Tensor(device='cuda:1')),
                        ('图捕获拒绝不连续值', Tensor(contiguous=False))]:
        rejected(name, value)
    print(json.dumps({'passed': True, 'checks': checks, 'gpu_accessed': False,
                      'scope': '源码身份和辅助函数CPU控制流，不代表GPU内核、模型正确性或性能验收'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
