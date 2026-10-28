"""共享CPU缓存进程的可选CPU亲和性；只改变新进程，不改主机或GPU归属。

仅在GPUSTACK_CACHE_CPU_AFFINITY=auto时启用。读取CUDA可见GPU对应的NUMA节点，
与进程当前允许CPU求交集后绑定。数据采用普通内存分配的首次触碰策略；不宣称
实现严格内存绑定。识别失败或权限不足时保留原行为，不影响模型可启动性。
"""
import csv
import os
from pathlib import Path
import subprocess
import sys


def expand_cpu_list(value):
    """解析内核cpulist格式，拒绝倒序和负数。"""
    result = set()
    for group in value.strip().split(','):
        first, dash, last = group.partition('-')
        start, end = int(first), int(last if dash else first)
        if start < 0 or end < start:
            raise ValueError('CPU列表无效')
        result.update(range(start, end + 1))
    return result


def select_gpu_rows(rows, visible, order):
    """按CUDA语义解析序号或UUID；序号只在明确PCI_BUS_ID顺序时使用。"""
    if not visible or visible.strip().lower() in ('all', 'none', 'void', '-1'):
        return []
    ordered = sorted(rows, key=lambda row: row['pci'].lower())
    selected = []
    for token in visible.split(','):
        token = token.strip()
        if token.isdigit():
            if order != 'PCI_BUS_ID' or int(token) >= len(ordered):
                raise ValueError('CUDA序号不能安全对应PCI顺序')
            selected.append(ordered[int(token)])
        else:
            matches = [r for r in ordered if r['uuid'].startswith(token)]
            if len(matches) != 1:
                raise ValueError('GPU标识不唯一')
            selected.append(matches[0])
    return selected


def local_cpu_set(rows, allowed, sys_root=Path('/sys')):
    """只返回当前cgroup已经允许且靠近这些GPU的CPU，不扩大权限范围。"""
    nodes = set()
    for row in rows:
        domain, bus, slot = row['pci'].lower().rsplit(':', 2)
        pci = f'{int(domain, 16):04x}:{bus}:{slot}'
        node = int((sys_root / 'bus/pci/devices' / pci / 'numa_node').read_text())
        if node < 0:
            raise ValueError('GPU没有明确NUMA节点')
        nodes.add(node)
    local = set()
    for node in nodes:
        local.update(expand_cpu_list((sys_root / f'devices/system/node/node{node}/cpulist').read_text()))
    return local & set(allowed), nodes


def activate():
    """Python启动钩子只在缓存定义明确启用时工作，异常时安全回退。"""
    if os.environ.get('GPUSTACK_CACHE_CPU_AFFINITY') != 'auto' or not hasattr(os, 'sched_getaffinity'):
        return
    try:
        output = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id', '--format=csv,noheader'],
                                check=True, capture_output=True, text=True, timeout=5).stdout
        rows = [{'index': int(x[0]), 'uuid': x[1].strip(), 'pci': x[2].strip()}
                for x in csv.reader(output.splitlines())]
        selected = select_gpu_rows(rows, os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.environ.get('CUDA_DEVICE_ORDER', ''))
        if not selected:
            return
        cpus, nodes = local_cpu_set(selected, os.sched_getaffinity(0))
        if not cpus:
            raise ValueError('GPU附近没有cgroup允许的CPU')
        os.sched_setaffinity(0, cpus)
        print('[gpustack-numa] cache CPU affinity nodes=' + ','.join(map(str, sorted(nodes))), file=sys.stderr)
    except Exception as error:
        print('[gpustack-numa] affinity unavailable; preserving original execution: ' + type(error).__name__, file=sys.stderr)


if __name__ == '__main__':
    activate()
