#!/usr/bin/env python3
"""验证非连续缓存块经过CPU内存读写后逐位一致，不加载模型或启动缓存服务。"""
import json
import os

import torch
import lmcache.cuda_ops as cuda_ops
import lmcache.lmcache_native as native


def check(dtype):
    """分别覆盖NVFP4使用的字节存储和BF16状态，显式传入物理块步长。"""
    device=torch.device('cuda:0')
    layers,blocks,block_size,head_size=2,16,1536,64
    padding=128
    source=[]
    target=[]
    # 容器只应暴露本次授权的一张显卡；cuda:0是容器内部的逻辑设备约定。
    for layer in range(layers):
        parent=torch.arange(blocks*(block_size*head_size+padding),device=device,dtype=torch.int32)
        parent=((parent+layer)%251).to(dtype).reshape(blocks,-1)
        source.append(parent.as_strided((blocks,block_size,head_size),(parent.stride(0),head_size,1)))
        empty=torch.zeros_like(parent)
        target.append(empty.as_strided(source[-1].shape,source[-1].stride()))
    shape=native.PageBufferShapeDesc()
    shape.kv_size=1
    shape.nl=layers
    shape.nb=blocks
    shape.bs=block_size
    shape.nh=1
    shape.hs=head_size
    shape.element_size=source[0].element_size()
    shape.block_stride_elems=source[0].stride(0)
    assert shape.block_stride_elems>block_size*head_size
    tokens_per_object=block_size*2
    objects=[torch.zeros((1,layers,tokens_per_object,head_size),dtype=dtype,pin_memory=True) for _ in range(2)]
    original_ids=[1,3,4,6]
    restored_ids=[8,10,12,14]
    for tensors,ids,direction in [(source,original_ids,native.TransferDirection.D2H),
                                  (target,restored_ids,native.TransferDirection.H2D)]:
        pointers=torch.tensor([t.data_ptr() for t in tensors],dtype=torch.int64,device=device)
        selected=torch.tensor(ids,dtype=torch.int64,device=device)
        cuda_ops.multi_layer_block_kv_transfer(pointers,[t.data_ptr() for t in objects],selected,device,
            direction,shape,tokens_per_object,native.EngineKVFormat.NL_X_NB_BS_HS,0)
        torch.cuda.synchronize()
    for source_id,target_id in zip(original_ids,restored_ids):
        for left,right in zip(source,target):
            assert torch.equal(left[source_id],right[target_id]),'缓存往返后数据不一致'
    return {'dtype':str(dtype),'block_stride_elems':shape.block_stride_elems,'blocks_checked':len(original_ids)*layers}


def main():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible or ',' in visible or torch.cuda.device_count()!=1:
        raise RuntimeError('核验容器必须通过调用者明确授权且只暴露一张GPU')
    torch.cuda.reset_peak_memory_stats()
    results=[check(torch.uint8),check(torch.bfloat16)]
    peak=torch.cuda.max_memory_allocated()
    assert peak<64*1024**2
    print(json.dumps({'passed':True,'cases':results,'peak_tensor_bytes':peak,'model_loaded':False}))


if __name__=='__main__':
    main()
