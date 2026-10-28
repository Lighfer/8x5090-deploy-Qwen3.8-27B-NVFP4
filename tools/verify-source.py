#!/usr/bin/env python3
"""在仓库外重放固定Worker及Router源码；可选核验提取的官方推理基础源码。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

sys.dont_write_bytecode = True
from repository_common import ROOT, outside_repository, digest, safe_path, read_json, output_json


def run(*arguments):
    """保留失败退出码和私有日志，不将输出复制进源码树。"""
    return subprocess.run([sys.executable, '-B', *map(str, arguments)], check=True,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True, help='仓库外、尚不存在的核验目录')
    parser.add_argument('--router-archive', help='可选的仓库外公开源码包；未提供时按固定来源下载')
    parser.add_argument('--base-source', help='可选的仓库外基础源码目录，包含base_scheduler.py、page.py及page.cuh')
    parser.add_argument('--optimization-source', help='可选的仓库外原始软件目录，包含优化清单登记的vllm与flashinfer文件')
    args = parser.parse_args()
    work = outside_repository(args.work, may_exist=False)
    work.mkdir(parents=True)
    rows = []
    result = run(ROOT / 'tools/apply-memory-patch.py', '--input',
                 ROOT / 'sources/worker-memory/fastfetch.original.py', '--output', work / 'fastfetch.py', '--execute')
    (work / 'memory-output.txt').write_text(result.stdout, encoding='utf-8')
    rows.append({'check': 'worker-memory-source-and-patch', 'passed':
                 digest(work / 'fastfetch.py') == digest(ROOT / 'templates/runtime/worker/fastfetch.py')})
    result = run(ROOT / 'sources/worker-runtime/patch-docker-nixl-pid.py', '--input',
                 ROOT / 'sources/worker-runtime/docker.original.py', '--output', work / 'docker.py',
                 '--verification', work / 'runtime-verification.json')
    (work / 'runtime-output.txt').write_text(result.stdout, encoding='utf-8')
    rows.append({'check': 'worker-runtime-source-and-eight-boundaries', 'passed':
                 read_json(work / 'runtime-verification.json')['passed'] and
                 digest(work / 'docker.py') == digest(ROOT / 'templates/runtime/worker/docker.py')})
    lock = read_json(ROOT / 'locks/rebuild-inputs.json')
    archive = work / 'router.tar.gz'
    if args.router_archive:
        shutil.copyfile(outside_repository(args.router_archive), archive)
    else:
        with urllib.request.urlopen('https://pypi.org/pypi/vllm-router/0.1.16/json', timeout=30) as response:
            metadata = json.load(response)
        record = next(row for row in metadata['urls'] if row['packagetype'] == 'sdist')
        if record['digests']['sha256'] != lock['router_sdist_sha256'] or not record['url'].startswith('https://files.pythonhosted.org/'):
            raise ValueError('公开Router源码身份不符')
        with urllib.request.urlopen(record['url'], timeout=60) as response:
            archive.write_bytes(response.read(16 * 1024 * 1024))
    if digest(archive) != lock['router_sdist_sha256']:
        raise ValueError('Router源码包散列不符')
    # 不接受源码包中的符号链接、设备或越界文件；所有输出都限制在本次外部核验目录。
    total = 0
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            path = safe_path(work, member.name)
            total += member.size
            if total > 64 * 1024 * 1024 or not (member.isfile() or member.isdir()):
                raise ValueError('源码包存在未允许成员或体积异常')
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, path.open('wb') as target:
                    shutil.copyfileobj(source, target)
    router = work / 'vllm_router-0.1.16'
    for script in ('patch-router-pd-type.py', 'patch-router-pd-routing.py'):
        result = run(ROOT / 'sources/router' / script, '--root', router)
        (work / (script + '.txt')).write_text(result.stdout, encoding='utf-8')
    checked = 0
    for line in (ROOT / 'sources/router/patched-source.sha256').read_text(encoding='utf-8').splitlines():
        checksum, name = line.split(maxsplit=1)
        if digest(safe_path(router, name.strip())) != checksum:
            raise ValueError('Router修正后源码或Cargo.lock散列不符')
        checked += 1
    rows.append({'check': 'router-source-patches-and-cargo-lock', 'passed': True, 'files': checked})
    if args.base_source:
        base = outside_repository(args.base_source)
        for name in ('base_scheduler.py', 'page.py', 'page.cuh'):
            shutil.copyfile(safe_path(base, name), work / name)
        result = run(ROOT / 'sources/vllm/nightly-patches/patch-nixl-handshake.py', work / 'base_scheduler.py')
        (work / 'nixl-output.txt').write_text(result.stdout, encoding='utf-8')
        rows.append({'check': 'nixl-fixed-source-replay', 'passed':
                     digest(work / 'base_scheduler.py') == read_json(ROOT / 'STACK.json')['patched_nixl_sha256']})
        result = run(ROOT / 'sources/vllm/flashinfer6204/patch-flashinfer-nvfp4.py', '--input',
                     work / 'page.py', '--kernel', work / 'page.cuh', '--output', work / 'page-patched.py', '--execute')
        (work / 'flashinfer-output.txt').write_text(result.stdout, encoding='utf-8')
        rows.append({'check': 'flashinfer-official-diff-and-kernel-guards', 'passed':
                     digest(work / 'page-patched.py') == read_json(ROOT / 'STACK.json')['flashinfer6204']['page_patched_sha256']})
    if args.optimization_source:
        base = outside_repository(args.optimization_source)
        destination = work / 'optimization-source'
        manifest = read_json(ROOT / 'sources/vllm/optimizations/manifest.json')
        for name in manifest['original_files']:
            source, target = safe_path(base, name), safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        result = run(ROOT / 'sources/vllm/optimizations/apply-patches.py', '--root', destination, '--execute')
        (work / 'optimization-output.txt').write_text(result.stdout, encoding='utf-8')
        if any(digest(safe_path(destination, name)) != checksum for name, checksum in manifest['final_files'].items()):
            raise ValueError('候选优化重放末态不符')
        # 第二次执行必须确认幂等，不得在已经修正的源码上继续叠加差异。
        second = run(ROOT / 'sources/vllm/optimizations/apply-patches.py', '--root', destination, '--execute')
        if not json.loads(second.stdout)['already_applied']:
            raise ValueError('候选优化重放不是幂等操作')
        rows.append({'check': 'optimization-patches-and-idempotence', 'passed': True,
                     'patches': len(manifest['patches']), 'files': len(manifest['final_files'])})
    report = {'passed': all(row['passed'] for row in rows), 'checks': rows,
              'gpu_inference_performed': False, 'base_source_checked': bool(args.base_source)}
    output_json(work / 'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
