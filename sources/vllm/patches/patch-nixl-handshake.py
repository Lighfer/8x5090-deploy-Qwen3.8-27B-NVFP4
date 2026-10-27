#!/usr/bin/env python3
"""为 vLLM 0.32.0 的 NIXL 元数据监听增加异常报文保护，不改变合法握手协议。"""
import ast
import hashlib
import json
from pathlib import Path
import sys

MARKER = 'ignore_nixl_malformed_handshake_v1'
ORIGINAL = '''                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue
                # Decode (GET_META_MSG, pp_rank, tp_rank).
                msg, target_pp_rank, target_tp_rank = msgspec.msgpack.decode(msg)
                logger.debug(
                    "Received message for pp rank %s, tp rank %s",
                    target_pp_rank,
                    target_tp_rank,
                )
                if msg != GET_META_MSG:
                    logger.warning("Connection listener got unexpected message %s", msg)
'''
REPLACEMENT = '''                try:
                    frames = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue
                # ignore_nixl_malformed_handshake_v1
                # REQ 握手应包含路由标识、空分隔帧和正文。其他协议或异常客户端可能少发帧，
                # 必须丢弃当前消息并继续监听，避免后台线程退出而 HTTP 健康检查仍报告正常。
                if len(frames) != 3 or frames[1] != b"":
                    logger.warning("Ignoring malformed NIXL handshake envelope")
                    continue
                identity, _, payload = frames
                try:
                    message = msgspec.msgpack.decode(payload)
                except (msgspec.DecodeError, ValueError):
                    logger.warning("Ignoring undecodable NIXL handshake payload")
                    continue
                if not isinstance(message, (list, tuple)) or len(message) != 3:
                    logger.warning("Ignoring malformed NIXL handshake fields")
                    continue
                msg, target_pp_rank, target_tp_rank = message
                # 只接受存在的整数 PP/TP 秩；不把任意客户端正文写入日志。
                if (
                    msg != GET_META_MSG
                    or type(target_pp_rank) is not int
                    or type(target_tp_rank) is not int
                    or (target_pp_rank, target_tp_rank) not in encoded_data
                ):
                    logger.warning("Ignoring unsupported NIXL handshake target")
                    continue
                logger.debug(
                    "Received message for pp rank %s, tp rank %s",
                    target_pp_rank,
                    target_tp_rank,
                )
'''


def main():
    path = Path(sys.argv[1] if len(sys.argv)>1 else '/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py')
    original = path.read_text()
    if MARKER in original:
        print(json.dumps(dict(state='already_patched',sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
        return
    # 对固定上游源码核对散列，避免后续升级错误套用旧补丁。
    if hashlib.sha256(original.encode()).hexdigest() != 'd075560cd8bb6cef27b77fb13dc75bf738e537cb492751fa3c7f006091a86215':
        raise RuntimeError('vLLM 0.32.0 NIXL source SHA256 mismatch')
    if original.count(ORIGINAL) != 1:
        raise RuntimeError('Installed NIXL source does not match the reviewed 0.32.0 implementation')
    updated = original.replace(ORIGINAL, REPLACEMENT)
    ast.parse(updated)
    # 原内容与替换内容均完成校验后才写入；构建失败不会影响正在运行的镜像。
    path.write_text(updated)
    print(json.dumps(dict(state='patched',before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                         after_sha256=hashlib.sha256(updated.encode()).hexdigest())))


if __name__=='__main__':main()
