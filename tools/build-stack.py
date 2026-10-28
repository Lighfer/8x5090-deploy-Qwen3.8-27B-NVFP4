#!/usr/bin/env python3
"""按固定软件输入构建新标签；构建目录、镜像名称与资源限额均由本次维护提供。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
from urllib.parse import urlsplit

sys.dont_write_bytecode = True
from repository_common import ROOT, read_json, outside_repository


def run(arguments, **options):
    """子命令使用参数数组；不通过Shell拼接执行用户输入。"""
    subprocess.run(arguments, check=True, **options)


def source_archive(directory, expected, opener):
    """从固定PyPI版本获取源码，内容必须匹配独立保存的SHA256。"""
    metadata_url = 'https://pypi.org/pypi/vllm-router/0.1.16/json'
    with opener.open(metadata_url, timeout=30) as response:
        metadata = json.load(response)
    record = next(row for row in metadata['urls'] if row['packagetype'] == 'sdist')
    if record['digests']['sha256'] != expected or not record['url'].startswith('https://files.pythonhosted.org/'):
        raise RuntimeError('Router公开源码身份与固定输入不符')
    target = directory / 'vllm_router-0.1.16.tar.gz'
    with opener.open(record['url'], timeout=60) as response:
        content = response.read(16 * 1024 * 1024)
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError('Router源码下载散列不符')
    target.write_bytes(content)


def lmcache_archive(directory, record, opener, local=None):
    """取得固定LMCache源码；支持仓库外预下载输入，不保存源码包到公开仓库。"""
    if local is not None:
        content = outside_repository(local).read_bytes()
    else:
        address = urlsplit(record['url'])
        if address.scheme != 'https' or address.hostname != 'files.pythonhosted.org':
            raise RuntimeError('LMCache源码地址不是固定官方发布源')
        with opener.open(record['url'], timeout=60) as response:
            content = response.read(32 * 1024 * 1024)
    if len(content) != record['size'] or hashlib.sha256(content).hexdigest() != record['sha256']:
        raise RuntimeError('LMCache源码体积或散列不符')
    (directory / 'lmcache-0.5.6.tar.gz').write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True, help='不存在且位于仓库之外的本次构建目录')
    parser.add_argument('--prefix', required=True, help='用户选择的镜像仓库前缀')
    parser.add_argument('--release', required=True, help='本次新标签的版本后缀')
    parser.add_argument('--cpus', type=int, required=True, choices=range(1, 9))
    parser.add_argument('--memory-gib', type=int, required=True, choices=range(4, 17))
    parser.add_argument('--variant', choices=('current', 'legacy'), default='current')
    parser.add_argument('--runtime-only', action='store_true')
    parser.add_argument('--proxy', help='可选回环HTTP下载代理；不改主机网络或写入镜像配置')
    parser.add_argument('--lmcache-archive', help='可选的仓库外LMCache固定源码包，用于已预下载场景')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    work = outside_repository(args.work, may_exist=False)
    if not re.fullmatch(r'[a-z0-9][a-z0-9./_-]*', args.prefix) or not re.fullmatch(r'[a-z0-9][a-z0-9._-]*', args.release):
        parser.error('镜像前缀或版本后缀格式不合法')
    if args.variant == 'legacy' and not args.runtime_only:
        parser.error('legacy仅用于独立兼容运行镜像；Router沿用其单独固定的配方')
    if args.variant == 'current' and args.memory_gib < 8:
        parser.error('当前版本需要重建LMCache CUDA扩展，构建内存限额至少8GiB')
    tags = [args.prefix + ':' + args.release + '-runtime', args.prefix + ':' + args.release + '-router']
    if args.runtime_only:
        tags = tags[:1]
    print(json.dumps({'dry_run': not args.execute, 'output_tags': tags, 'gpu_models_started': False,
                      'production_modified': False, 'variant': args.variant}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    if available < (args.memory_gib + 10) * 1024 ** 3:
        raise RuntimeError('构建限额之外没有足够的运行内存余量')
    aliases = [args.prefix + ':' + args.release + '-base']
    if not args.runtime_only:
        aliases.append(args.prefix + ':' + args.release + '-router-base')
    for tag in [*tags, *aliases]:
        if subprocess.run(['docker', 'image', 'inspect', tag], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError('输出标签已存在，禁止覆盖')
    network = []
    handlers = []
    if args.proxy:
        proxy = urlsplit(args.proxy)
        if proxy.scheme != 'http' or proxy.hostname not in ('localhost', '127.0.0.1') or not proxy.port or proxy.username or proxy.password:
            parser.error('构建代理仅接受无凭据的回环HTTP地址')
        network = ['--network=host']
        for key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
            network.extend(['-e', key + '=' + args.proxy])
        handlers.append(urllib.request.ProxyHandler({'http': args.proxy, 'https': args.proxy}))
    opener = urllib.request.build_opener(*handlers)
    lock = read_json(ROOT / 'locks/rebuild-inputs.json')
    base = lock['vllm_base_digest'] if args.variant == 'current' else lock['router_base_digest']
    run(['docker', 'pull', base])
    actual_base = json.loads(subprocess.check_output(['docker', 'image', 'inspect', base]))[0]
    # 不同Docker存储后端的本地ID表示可能不同；公共摘要及架构才是跨机器输入身份。
    if base not in actual_base.get('RepoDigests', []) or actual_base['Os'] != 'linux' or actual_base['Architecture'] != 'amd64':
        raise RuntimeError('官方基础身份不符，不能应用固定源码补丁')
    work.mkdir(parents=True)
    runtime, router = work / 'runtime', work / 'router'
    shutil.copytree(ROOT / 'sources/vllm', runtime)
    if args.variant == 'legacy':
        shutil.copyfile(runtime / 'Dockerfile.legacy', runtime / 'Dockerfile')
    else:
        lmcache_archive(runtime, lock['lmcache_cuda_source'], opener, args.lmcache_archive)
    # 构建器只读取已经核对的本地别名；引用名由本次发布选择，内容仍锁定公开摘要。
    base_alias = args.prefix + ':' + args.release + '-base'
    run(['docker', 'tag', base, base_alias])
    environment = dict(os.environ, DOCKER_BUILDKIT='0')
    run(['docker', 'build', '--pull=false', '--network=none', '--memory=' + str(args.memory_gib) + 'g',
         '--cpu-period=100000', '--cpu-quota=' + str(args.cpus * 100000), '--build-arg',
         'BASE_IMAGE=' + base_alias, '-t', tags[0], str(runtime)], env=environment)
    if not args.runtime_only:
        shutil.copytree(ROOT / 'sources/router', router)
        source_archive(router, lock['router_sdist_sha256'], opener)
        run(['docker', 'pull', lock['router_base_digest']])
        router_base = json.loads(subprocess.check_output(['docker', 'image', 'inspect', lock['router_base_digest']]))[0]
        if lock['router_base_digest'] not in router_base.get('RepoDigests', []) or router_base['Os'] != 'linux' or router_base['Architecture'] != 'amd64':
            raise RuntimeError('Router基础身份不符')
        # Router派生层同样使用已经核对的本次别名，不让构建器重新解析远端引用。
        run(['docker', 'tag', lock['router_base_digest'], aliases[1]])
        run(['docker', 'pull', lock['rust_sdk']])
        run(['docker', 'run', '--rm', '--runtime=runc', '--memory=' + str(args.memory_gib) + 'g',
             '--memory-swap=' + str(args.memory_gib) + 'g', '--cpus=' + str(args.cpus), '--pids-limit=4096',
             '-e', 'CARGO_BUILD_JOBS=' + str(args.cpus), *network,
             '-v', str(router) + ':/build', '--entrypoint', 'sh', lock['rust_sdk'], '/build/build-native-router.sh'])
        wheels = router / 'wheels'
        wheels.mkdir()
        shutil.copyfile(router / 'router-requirements.txt', wheels / 'router-requirements.txt')
        run(['docker', 'run', '--rm', '--runtime=runc', '--memory=1g', '--cpus=1', *network,
             '-v', str(wheels) + ':/wheels', '--entrypoint', 'python3', tags[0],
             '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:', '--require-hashes',
             '-r', '/wheels/router-requirements.txt', '--dest', '/wheels'])
        (router / '.dockerignore').write_text('*\n!Dockerfile\n!LICENSE\n!libvllm_router_rs.so\n!verify-native-router.py\n!verify-routing-mock.py\n!wheels/\n!wheels/**\n')
        run(['docker', 'build', '--pull=false', '--network=none', '--memory=' + str(args.memory_gib) + 'g',
             '--cpu-period=100000', '--cpu-quota=' + str(args.cpus * 100000), '--build-arg',
             'BASE_IMAGE=' + aliases[1], '-t', tags[1], str(router)], env=environment)
    result = {'images': {tag: subprocess.check_output(['docker', 'image', 'inspect', '-f', '{{.Id}}', tag], text=True).strip()
                         for tag in tags}, 'cpu_build_checks_passed': True,
              'gpu_inference_verified': False, 'production_modified': False}
    (work / 'build-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
