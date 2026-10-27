#!/usr/bin/env python3
"""使用回环 HTTP 模拟 P/D，验证真实原生 Router 的策略、成员管理及请求内容，不接触 GPU。"""
import argparse
import collections
import concurrent.futures
import contextlib
import http.server
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

MODEL = 'mock-pd-model'


def api(base, route, method='GET', body=None, timeout=15):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + route, data, {'Content-Type': 'application/json', 'X-Prefer-PD': 'true'}, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class DummyWorker:
    """只记录分流和测试正文，不加载模型；阻塞事件用于构造一个 P 忙、另一个 P 空闲的状态。"""
    def __init__(self, role, index):
        self.role, self.index = role, index
        self.requests = []
        self.entered = threading.Event()
        self.release = threading.Event()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, payload):
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.send({'status': 'ok', 'model_id': MODEL, 'model_path': MODEL,
                    'object': 'list', 'data': [{'id': MODEL, 'object': 'model'}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append({'path': self.path, 'body': body})
                text = json.dumps(body, ensure_ascii=False)
                if owner.role == 'prefill' and 'HOLD_WORKER' in text:
                    owner.entered.set()
                    if not owner.release.wait(20):
                        raise TimeoutError('Held test request was not released')
                tag = f'{owner.role}-{owner.index}'
                if self.path.endswith('/responses'):
                    result = {'id': 'resp_' + secrets.token_hex(8), 'object': 'response', 'model': MODEL,
                        'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant',
                            'content': [{'type': 'output_text', 'text': tag}]}],
                        'usage': {'input_tokens': 64, 'output_tokens': 1, 'total_tokens': 65}}
                else:
                    result = {'id': 'chatcmpl_' + secrets.token_hex(8), 'object': 'chat.completion',
                        'model': MODEL, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': tag},
                            'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 64, 'completion_tokens': 1, 'total_tokens': 65}}
                if owner.role == 'prefill':
                    result['kv_transfer_params'] = {'do_remote_prefill': True, 'remote_engine_id': tag,
                        'remote_request_id': tag, 'remote_host': '127.0.0.1', 'remote_port': 1,
                        'remote_block_ids': [[0]], 'tp_size': 1, 'remote_num_tokens': 63}
                self.send(result)

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


def unused_port():
    with socket.socket() as connection:
        connection.bind(('127.0.0.1', 0))
        return connection.getsockname()[1]


@contextlib.contextmanager
def router(site, output, workers, prefill_policy, decode_policy):
    port = unused_port()
    base = 'http://127.0.0.1:' + str(port)
    configuration = {'worker_urls': [], 'enable_igw': True, 'vllm_pd_disaggregation': True,
        'prefill_urls': [(worker.url, None) for worker in workers if worker.role == 'prefill'],
        'decode_urls': [worker.url for worker in workers if worker.role == 'decode'],
        'host': '127.0.0.1', 'port': port, 'policy': 'cache_aware',
        'prefill_policy': prefill_policy, 'decode_policy': decode_policy, 'kv_connector': 'nixl'}
    # 直接调用固定 wheel 的原生入口，避免测试依赖 setproctitle 或 Python MiniLB。
    code = ('import json; from vllm_router.router import Router; from vllm_router.router_args import RouterArgs; '
        'configuration=json.loads(' + repr(json.dumps(configuration)) + '); '
        'configuration["prefill_urls"]=[tuple(item) for item in configuration["prefill_urls"]]; '
        'args=RouterArgs(**configuration); '
        'args._validate_router_args(); Router.from_args(args).start()')
    environment = {**os.environ, 'PYTHONPATH': str(site), 'RUST_LOG': 'info'}
    with output.open('w') as logs:
        process = subprocess.Popen([sys.executable, '-c', code], stdout=logs, stderr=subprocess.STDOUT, env=environment)
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Native Router exited before becoming healthy')
                try:
                    with urllib.request.urlopen(base + '/health', timeout=1) as response:
                        assert response.status == 200
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Native Router did not become healthy')
                    time.sleep(.1)
            # 与 GPUStack 相同，从空表通过原生接口登记各个角色。
            for worker in workers:
                result = api(base, '/workers', 'POST', {'url': worker.url, 'model_id': MODEL, 'worker_type': worker.role})
                assert result['success']
            yield base
        finally:
            for worker in workers:
                worker.release.set()
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def body(marker, salt, protocol='chat', hold=False):
    # 长共同前缀足以触发缓存优先策略；末尾标记只控制模拟 Worker 的阻塞。
    text = marker + ' shared-prefix-record' * 600 + (' HOLD_WORKER' if hold else ' READY')
    result = {'model': MODEL, 'stream': False, 'cache_salt': salt}
    if protocol == 'chat':
        result.update(messages=[{'role': 'user', 'content': text}], max_completion_tokens=32)
    else:
        result.update(input=text, max_output_tokens=32)
    return result


def counts(workers, role):
    return [len(worker.requests) for worker in workers if worker.role == role]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expect-legacy-failure', action='store_true')
    args = parser.parse_args()
    destination = Path(args.output)
    reports = []
    for run, (p_policy, d_policy, replicas) in enumerate((('cache_aware', 'round_robin', 2),
                                                        ('round_robin', 'cache_aware', 2),
                                                        ('cache_aware', 'round_robin', 1))):
        workers = [DummyWorker(role, index) for role in ('prefill', 'decode') for index in range(replicas)]
        logs = destination.with_name(destination.stem + '-' + str(run) + '.log')
        try:
            with router(Path(args.site), logs, workers, p_policy, d_policy) as base:
                marker = secrets.token_hex(8)
                warm = body(marker, 'namespace-a')
                for _ in range(24):
                    api(base, '/v1/chat/completions', 'POST', warm)
                p_counts, d_counts = counts(workers, 'prefill'), counts(workers, 'decode')
                expected_p = max(p_counts) == 24 if p_policy == 'cache_aware' else max(p_counts) - min(p_counts) <= 1
                expected_d = max(d_counts) == 24 if d_policy == 'cache_aware' else max(d_counts) - min(d_counts) <= 1
                record = {'prefill_policy': p_policy, 'decode_policy': d_policy, 'replicas': replicas,
                    'prefill_counts': p_counts, 'decode_counts': d_counts, 'distribution_correct': expected_p and expected_d}
                if args.expect_legacy_failure:
                    record['expected_legacy_problem_reproduced'] = not record['distribution_correct']
                    reports.append(record)
                    break
                assert record['distribution_correct'], 'Declared stage policies were not applied'
                if replicas == 2 and p_policy == 'cache_aware':
                    cached = next(worker for worker in workers if worker.role == 'prefill' and len(worker.requests) == 24)
                    other = next(worker for worker in workers if worker.role == 'prefill' and worker is not cached)
                    before_other = len(other.requests)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        held = pool.submit(api, base, '/v1/chat/completions', 'POST', body(marker, 'namespace-a', hold=True))
                        assert cached.entered.wait(5), 'Cached request was not placed on the known worker'
                        try:
                            api(base, '/v1/chat/completions', 'POST', body(marker, 'namespace-b'))
                            assert len(other.requests) == before_other + 1, 'Cache namespaces were merged by routing prediction'
                        finally:
                            cached.release.set()
                        held.result()
                    record['routing_cache_namespace_isolation'] = True
                membership = []
                for worker in workers:
                    result = api(base, '/workers/' + urllib.parse.quote(worker.url, safe=''), 'DELETE')
                    assert result['success']
                    assert all(item['url'] != worker.url for item in api(base, '/workers')['workers'])
                    result = api(base, '/workers', 'POST', {'url': worker.url, 'model_id': MODEL, 'worker_type': worker.role})
                    assert result['success']
                    membership.append({'role': worker.role, 'index': worker.index, 'removed_and_readded': True})
                record['membership'] = membership
                for protocol in ('chat', 'responses'):
                    request = body(marker, 'namespace-after-membership', protocol)
                    result = api(base, '/v1/chat/completions' if protocol == 'chat' else '/v1/responses', 'POST', request)
                    assert result.get('choices') or result.get('status') == 'completed'
                    decoded = next(worker.requests[-1]['body'] for worker in workers if worker.role == 'decode' and worker.requests
                        and worker.requests[-1]['body'].get('cache_salt') == request['cache_salt']
                        and ('input' in worker.requests[-1]['body']) == (protocol == 'responses'))
                    for key in request:
                        assert decoded[key] == request[key], 'The decode request was altered'
                record['chat_and_responses_payload_preserved'] = True
                reports.append(record)
            text = logs.read_text()
            assert 'Multi-router mode enabled (enable_igw=true)' in text
            assert f'PD stage policies active: prefill={p_policy}, decode={d_policy}' in text
            record['actual_policy_log_verified'] = True
        finally:
            for worker in workers:
                worker.close()
        destination.write_text(json.dumps({'passed': False, 'runs': reports}, indent=2))
    passed = bool(reports) and all(row.get('expected_legacy_problem_reproduced') if args.expect_legacy_failure
        else row['distribution_correct'] and row.get('actual_policy_log_verified') for row in reports)
    destination.write_text(json.dumps({'passed': passed, 'legacy_reference': args.expect_legacy_failure, 'runs': reports}, indent=2))
    print(json.dumps({'passed': passed, 'runs': len(reports), 'legacy_reference': args.expect_legacy_failure}))
    if not passed:
        raise RuntimeError('Native Router policy verification failed')


if __name__ == '__main__':
    main()
