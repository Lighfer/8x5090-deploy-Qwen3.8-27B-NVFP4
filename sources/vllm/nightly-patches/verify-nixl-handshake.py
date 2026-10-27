#!/usr/bin/env python3
"""在独立回环端口运行实际监听函数，验证异常客户端不会中断后续合法握手。"""
import argparse
import ast
import contextlib
import json
import logging
from pathlib import Path
import socket
import threading
import time
import typing
import msgspec
import zmq


@contextlib.contextmanager
def socket_context(kind, address):
    """与 NIXL 包装相同的 ROUTER/REQ 生命周期；只绑定临时回环端口，不连接生产实例。"""
    context=zmq.Context()
    channel=context.socket(kind)
    channel.linger=0
    if kind==zmq.ROUTER:channel.bind(address)
    else:channel.connect(address)
    try:yield channel
    finally:context.destroy(linger=0)


def load_listener(path):
    """提取安装文件中的原始函数体，避免导入模型或分配任何 GPU 内存。"""
    tree=ast.parse(Path(path).read_text())
    cls=next(item for item in tree.body if isinstance(item,ast.ClassDef) and item.name=='NixlBaseConnectorScheduler')
    node=next(item for item in cls.body if isinstance(item,ast.FunctionDef) and item.name=='_nixl_handshake_listener')
    node.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    ast.fix_missing_locations(module)
    scope=dict(zmq=zmq,msgspec=msgspec,threading=threading,time=time,Any=typing.Any,
               zmq_ctx=socket_context,make_zmq_path=lambda protocol,host,port:f'{protocol}://{host}:{port}',
               GET_META_MSG=b'get_meta',logger=logging.getLogger('nixl-isolated-test'))
    exec(compile(module,str(path),'exec'),scope)
    return scope['_nixl_handshake_listener']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py')
    parser.add_argument('--expect-original-failure',action='store_true')
    args=parser.parse_args()
    with socket.socket() as available:
        available.bind(('127.0.0.1',0));port=available.getsockname()[1]
    address=f'tcp://127.0.0.1:{port}'
    ready=threading.Event();stop=threading.Event();errors=[]
    listener=load_listener(args.source)
    def serve():
        try:listener({(0,0):b'known-rank-metadata'},ready,stop,'127.0.0.1',port)
        except Exception as error:errors.append(type(error).__name__+': '+str(error))
    worker=threading.Thread(target=serve,daemon=True);worker.start()
    if not ready.wait(3):raise RuntimeError('Listener did not become ready')
    def request(payload,kind=zmq.REQ,expect_reply=False):
        context=zmq.Context();client=context.socket(kind);client.linger=0;client.rcvtimeo=1000
        try:
            client.connect(address);client.send(payload)
            if expect_reply:
                reply=client.recv_multipart()
                if len(reply)!=2 or reply[0]!=b'known-rank-metadata' or not isinstance(msgspec.msgpack.decode(reply[1]),float):
                    raise RuntimeError('Legitimate handshake reply changed')
            else:time.sleep(.05)
        finally:context.destroy(linger=0)
    good=msgspec.msgpack.encode((b'get_meta',0,0))
    request(good,expect_reply=True)
    # DEALER 到 ROUTER 只有路由标识和正文两帧，原始实现会在固定的三变量解包处退出。
    request(good,kind=zmq.DEALER)
    worker.join(.2)
    if args.expect_original_failure:
        result=dict(original_listener_failed=not worker.is_alive(),errors=errors,production_ports_used=False)
        print(json.dumps(result),flush=True)
        if worker.is_alive() or not errors:raise RuntimeError('Original failure was not reproduced')
        return
    request(good,expect_reply=True)
    for invalid in [b'\xc1',msgspec.msgpack.encode([]),msgspec.msgpack.encode((b'wrong',0,0)),
                    msgspec.msgpack.encode((b'get_meta',99,0)),msgspec.msgpack.encode((b'get_meta',[],0))]:
        request(invalid)
        request(good,expect_reply=True)
    # 按误发健康检查的协议发一个 HTTP 请求。是否被 ZMTP 识别并交付取决于 libzmq；后续合法握手必须仍成功。
    with socket.create_connection(('127.0.0.1',port),timeout=1) as http:
        http.sendall(b'GET /health HTTP/1.1\r\nHost: localhost\r\nUser-Agent: Python-urllib/3.13\r\nConnection: close\r\n\r\n')
        time.sleep(.1)
    request(good,expect_reply=True)
    stop.set();worker.join(2)
    result=dict(valid_before_and_after=True,malformed_cases=7,stopped_cleanly=not worker.is_alive(),errors=errors,production_ports_used=False)
    print(json.dumps(result),flush=True)
    if worker.is_alive() or errors:raise RuntimeError('Patched listener did not survive malformed input')


if __name__=='__main__':main()
