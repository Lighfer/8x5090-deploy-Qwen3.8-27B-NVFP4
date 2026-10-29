#!/usr/bin/env python3
"""从仓库外的明确环境输入生成私有部署目录；默认只核对，不写文件。"""
import argparse
import copy
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

sys.dont_write_bytecode = True
from repository_common import ROOT, read_json, outside_repository, safe_path

FP8_MODE = '1p1d-cache-fp8'
MODES = ('1p1d-cache', '1p1d-no-cache', '2p2d-cache', '1p2d-6gpu-cache', FP8_MODE)
INTERNAL_PATHS = {
    'server_data': '/var/lib/gpustack',
    'docker_socket': '/var/run/docker.sock',
    'meminfo': '/proc/meminfo',
    'cgroup': '/sys/fs/cgroup',
}
TOP_KEYS = {'format', 'host', 'paths', 'ports', 'names', 'gpus', 'resources', 'images', 'identity', 'admin_user', 'default_mode'}


def enabled_modes(site):
    """只有明确提供FP8目录才生成新模式，旧环境输入继续生成原有四种方案。"""
    return MODES if site['paths'].get('fp8_model_directory') else tuple(m for m in MODES if m != FP8_MODE)


def require_keys(value, expected, label):
    """拒绝缺项和不认识的键，避免拼写错误后悄悄套用某台机器的默认值。"""
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError('字段集合不符合要求：' + label)


def checked_site(path):
    """只读取仓库外配置；所有目标机身份均为必填，资源编号初次安装可以明确填写0。"""
    source = outside_repository(path)
    site = read_json(source)
    require_keys(site, TOP_KEYS, 'site')
    if site['format'] != 1:
        raise ValueError('环境文件格式不支持')
    require_keys(site['host'], {'address', 'interface', 'kind'}, 'host')
    ipaddress.IPv4Address(site['host']['address'])
    if site['host']['kind'] not in ('lxc', 'baremetal'):
        raise ValueError('host.kind必须明确为lxc或baremetal')
    if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,32}', site['host']['interface'] or ''):
        raise ValueError('必须提供实际网络接口名称')
    path_keys = {'deployment', 'model_parent', 'model_directory'}
    if 'fp8_model_directory' in site['paths']:
        path_keys.add('fp8_model_directory')
    require_keys(site['paths'], path_keys, 'paths')
    for key, value in site['paths'].items():
        if key == 'fp8_model_directory' and value is None:
            continue
        if not isinstance(value, str) or not PurePosixPath(value).is_absolute() or any(c in value for c in '\n\r\x00'):
            raise ValueError('目标目录必须是明确的Linux绝对路径')
    parent = PurePosixPath(site['paths']['model_parent'])
    model = PurePosixPath(site['paths']['model_directory'])
    if not model.is_relative_to(parent) or model == parent:
        raise ValueError('模型目录必须位于明确提供的模型父目录下')
    if site['paths'].get('fp8_model_directory'):
        fp8 = PurePosixPath(site['paths']['fp8_model_directory'])
        if not fp8.is_relative_to(parent) or fp8 in (parent, model):
            raise ValueError('FP8必须使用同一模型父目录下的独立目录，不能覆盖NVFP4权重')
    target = outside_repository(site['paths']['deployment'])
    # systemd会对ExecStart中的美元变量做自己的展开，安装目录明确拒绝这种歧义。
    # 模型目录不进入ExecStart，仍通过JSON与Compose环境文件保留其原始字符。
    if '$' in str(target):
        raise ValueError('部署目录不能包含美元符号；该字符会触发systemd环境展开')
    if source.is_relative_to(target):
        raise ValueError('环境输入文件应独立于将生成的部署目录')
    require_keys(site['ports'], {'management', 'inference', 'worker_api', 'worker_metrics', 'service_start', 'service_end'}, 'ports')
    ports = site['ports']
    if any(type(v) is not int or not 1024 <= v <= 65535 for v in ports.values()):
        raise ValueError('必须明确提供1024至65535的端口')
    fixed = [ports[k] for k in ('management', 'inference', 'worker_api', 'worker_metrics')]
    if len(set(fixed)) != len(fixed) or ports['service_end'] - ports['service_start'] < 63:
        raise ValueError('固定端口必须互不重复，动态端口段至少64个连续端口')
    if any(ports['service_start'] <= v <= ports['service_end'] for v in fixed):
        raise ValueError('动态端口段不能覆盖固定服务端口')
    name_keys = {'project', 'server', 'worker', 'proxy', 'cluster', 'node', 'worker_label', 'model', 'cache', 'server_volume', 'worker_volume'}
    require_keys(site['names'], name_keys, 'names')
    for key, value in site['names'].items():
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,62}', value):
            raise ValueError('必须明确提供安全的资源名称：' + key)
    if len({site['names'][k] for k in ('server', 'worker', 'proxy')}) != 3:
        raise ValueError('容器名称必须互不重复')
    # NUMA映射可以不提供；没有可靠拓扑时保持不绑定，不让受限CPU集合触发自动检测失败。
    gpu_keys = {'eight_card_order', 'six_card_order'}
    if 'numa_nodes' in site['gpus']:
        gpu_keys.add('numa_nodes')
    require_keys(site['gpus'], gpu_keys, 'gpus')
    for key, count in [('eight_card_order', 8), ('six_card_order', 6)]:
        values = site['gpus'][key]
        if not isinstance(values, list) or len(values) != count or len(set(values)) != count or any(type(v) is not int or v < 0 for v in values):
            raise ValueError('必须提供数量正确且无重复的实际GPU序号：' + key)
    if not set(site['gpus']['six_card_order']) <= set(site['gpus']['eight_card_order']):
        raise ValueError('六卡集合必须来自已授权的八卡集合')
    numa = site['gpus'].get('numa_nodes')
    if numa is not None and (not isinstance(numa, dict)
            or set(numa) != {str(i) for i in site['gpus']['eight_card_order']}
            or any(type(v) is not int or v < 0 for v in numa.values())):
        raise ValueError('NUMA映射必须完整覆盖授权八卡，键为GPU序号字符串，值为已核验的节点整数')
    require_keys(site['resources'], {'cache_gib', 'minimum_free_gib'}, 'resources')
    if any(type(v) is not int or v <= 0 for v in site['resources'].values()):
        raise ValueError('缓存容量和最低空闲内存必须由用户明确提供')
    require_keys(site['images'], {'runtime', 'cache', 'router'}, 'images')
    for value in site['images'].values():
        if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._/:-]*(?:@sha256:[0-9a-f]{64})?', value):
            raise ValueError('必须提供实际构建的安全镜像引用')
    require_keys(site['identity'], {'model_id', 'worker_id', 'cluster_id'}, 'identity')
    if any(type(v) is not int or v < 0 for v in site['identity'].values()):
        raise ValueError('资源编号必须是实际非负整数，初次准备统一使用0')
    if not isinstance(site['admin_user'], str) or not re.fullmatch(r'[A-Za-z0-9_.@-]{1,128}', site['admin_user']):
        raise ValueError('必须提供实际管理员用户名；不包含密码')
    if site['default_mode'] not in enabled_modes(site):
        raise ValueError('必须明确选择默认方案')
    return site, target


def inspect_images(site):
    """登记目标机实际构建的镜像ID，不把另一台机器的自制镜像ID作为公共下载地址。"""
    refs = set(site['images'].values())
    refs.add(next(i['RepoDigests'][0] for i in read_json(ROOT / 'locks/platform-images.json')['images']
                  if i['reference'].startswith('gpustack/gpustack')))
    result = {}
    for reference in refs:
        value = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', reference], text=True, timeout=30).strip()
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', value):
            raise RuntimeError('Docker没有返回合法镜像身份')
        result[reference] = value
    return result


def tokens(site, target):
    """Python字面量用repr生成，服务单元路径单独转义，不拼接用户可执行的Shell文本。"""
    names, ports = site['names'], site['ports']
    address = site['host']['address']
    manager = 'http://' + address + ':' + str(ports['management'])
    entry = 'http://' + address + ':' + str(ports['inference'])
    systemd_root = ''.join('%%' if c == '%' else '\\x' + format(ord(c), '02x')
                           if ord(c) in (32, 34, 39, 92, 9) else c for c in str(target))
    return {
        'ROOT_PY': repr(str(target)), 'ENTRY_ROOT_PY': repr(str(target / 'entrypoint')),
        'HOST_PY': repr(address), 'SERVER_PY': repr(names['server']), 'WORKER_PY': repr(names['worker']),
        'PROXY_PY': repr(names['proxy']), 'NODE_PY': repr(names['node']), 'MODEL_NAME_PY': repr(names['model']),
        'MANAGEMENT_URL_PY': repr(manager), 'ENTRY_URL_PY': repr(entry),
        'WORKER_METRICS_URL_PY': repr('http://' + address + ':' + str(ports['worker_metrics']) + '/metrics'),
        'ADMIN_PY': repr(site['admin_user']), 'MODEL_ID': str(site['identity']['model_id']),
        'WORKER_ID': str(site['identity']['worker_id']), 'ENTRY_PORT': str(ports['inference']),
        'RESERVED_PORTS_PY': repr(tuple(ports[k] for k in ('management', 'inference', 'worker_api', 'worker_metrics'))),
        'ROOT_SYSTEMD': systemd_root,
    }


def render_text(text, values):
    """只展开明确命名的标记，不处理Nginx变量、Compose变量或Shell表达式。"""
    def replace(match):
        if match[1] not in values:
            raise ValueError('模板标记没有对应输入：' + match[1])
        return str(values[match[1]])
    return re.sub(r'@@([A-Z_]+)@@', replace, text)


def model_values(site, mode):
    order = site['gpus']['six_card_order' if mode == '1p2d-6gpu-cache' else 'eight_card_order']
    split = 2 if mode == '1p2d-6gpu-cache' else 4
    prefix = site['names']['node'] + ':cuda:'
    directory = site['paths']['fp8_model_directory'] if mode == FP8_MODE else site['paths']['model_directory']
    return {'MODEL_NAME': site['names']['model'], 'MODEL_DIR': directory,
            'RUNTIME_IMAGE': site['images']['runtime'], 'ROUTER_IMAGE': site['images']['router'],
            'WORKER_NAME': site['names']['node'], 'WORKER_LABEL': site['names']['worker_label'], 'CLUSTER_NAME': site['names']['cluster'],
            'PREFILL_GPU_IDS': [prefix + str(i) for i in order[:split]],
            'DECODE_GPU_IDS': [prefix + str(i) for i in order[split:]]}


def replace_values(value, values):
    """整字段标记保留列表和数值类型，避免GPU列表或缓存容量变成字符串。"""
    if isinstance(value, dict):
        return {key: replace_values(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_values(item, values) for item in value]
    if isinstance(value, str):
        match = re.fullmatch(r'@@([A-Z_]+)@@', value)
        if match:
            if match[1] not in values:
                raise ValueError('JSON模板缺少输入：' + match[1])
            return copy.deepcopy(values[match[1]])
    return value


def bytes_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def apply_numa_binding(profile, nodes):
    """仅为同一NUMA节点内的角色显式绑定，避免多副本被分到不同节点后套用错误映射。

    vLLM按实例内可见GPU的逻辑序号读取节点列表。角色候选池跨节点时，静态列表无法
    表达不同副本的实际分配，因而不绑定该角色；不猜测调度器将如何切分候选池。
    """
    for scope in [profile, *profile['roles']]:
        if scope.get('name') == 'router':
            continue
        scope['backend_parameters'] = [v for v in scope.get('backend_parameters', []) if v != '--numa-bind']
    if nodes is None:
        return profile
    for role in profile['roles']:
        if role['name'] not in ('prefill', 'decode'):
            continue
        selected = role['gpu_selector']['gpu_ids']
        placement = {nodes[g.rsplit(':', 1)[1]] for g in selected}
        if len(placement) == 1:
            count = role['gpu_selector']['gpus_per_replica']
            role['backend_parameters'].extend(['--numa-bind', '--numa-bind-nodes', *([str(next(iter(placement)))] * count)])
    return profile


def generated_files(site, target, image_ids):
    """构造全部输出后再落盘；所有路径和身份只存在于输出，不反写源码模板。"""
    result = {}
    values = tokens(site, target)
    template_root = ROOT / 'templates/runtime'
    for path in template_root.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(template_root).as_posix()
        if relative.endswith('.in'):
            relative = relative[:-3]
            result[relative] = render_text(path.read_text(encoding='utf-8'), values).encode()
        else:
            result[relative] = path.read_bytes()
    cache = replace_values(read_json(ROOT / 'recipes/cache-service.json'), {
        'CACHE_NAME': site['names']['cache'], 'WORKER_NAME': site['names']['node'], 'WORKER_LABEL': site['names']['worker_label'],
        'CACHE_IMAGE': site['images']['cache'], 'CACHE_GIB': site['resources']['cache_gib'],
        'CACHE_GPU_IDS': ','.join(map(str, sorted(site['gpus']['eight_card_order'])))})
    cache['cluster_id'] = site['identity']['cluster_id']
    result['cache-service.json'] = bytes_json(cache)
    hashes, mode_gpus, mode_models, model_locks, mode_chunks = {}, {}, {}, {}, {}
    names = {'1p1d-cache': 'pd-model-stopped', '1p1d-no-cache': 'pd-model-1p1d-stopped',
             '2p2d-cache': 'pd-model-2p2d-stopped', '1p2d-6gpu-cache': 'pd-model-1p2d-6gpu-stopped',
             FP8_MODE: 'pd-model-1p1d-fp8-stopped'}
    for precision, lock_name in [('nvfp4', 'model-downloads.json'), ('fp8', 'model-downloads-fp8.json')]:
        if precision == 'fp8' and FP8_MODE not in enabled_modes(site):
            continue
        relative = 'model-inputs/' + precision + '.json'
        result[relative] = (ROOT / 'locks' / lock_name).read_bytes()
        model_locks[precision] = hashlib.sha256(result[relative]).hexdigest()
    for mode in enabled_modes(site):
        profile = replace_values(read_json(ROOT / 'recipes' / (mode + '.json')), model_values(site, mode))
        apply_numa_binding(profile, site['gpus'].get('numa_nodes'))
        data = bytes_json(profile)
        result['switch-profiles/' + mode + '.json'] = data
        hashes[mode] = hashlib.sha256(data).hexdigest()
        mode_gpus[mode] = list(site['gpus']['six_card_order' if mode == '1p2d-6gpu-cache' else 'eight_card_order'])
        mode_models[mode] = profile['local_path']
        # 分块来自各模式的正式配方。FP8与NVFP4可以采用不同页面大小，不能
        # 沿用一个全局chunk，否则重建CPU池后仍会与目标P的缓存块不整除。
        p_role = next(role for role in profile['roles'] if role['name'] == 'prefill')
        p_blocks = [int(arg.split('=', 1)[1]) for arg in p_role['backend_parameters']
                    if arg.startswith('--block-size=')]
        if len(p_blocks) != 1 or p_blocks[0] <= 0:
            raise ValueError('方案必须明确提供唯一正整数缓存块大小：' + mode)
        mode_chunks[mode] = p_blocks[0]
        result[names[mode] + '.json'] = data
    result['pd-model-new-stopped.json'] = result['switch-profiles/' + site['default_mode'] + '.json']
    configuration = {'format': 1, 'model_id': site['identity']['model_id'], 'model_name': site['names']['model'],
        'model_path': site['paths']['model_directory'], 'cluster_name': site['names']['cluster'],
        'entry_url': 'http://' + site['host']['address'] + ':' + str(site['ports']['inference']),
        'proxy_container': site['names']['proxy'], 'server_container': site['names']['server'],
        'worker_container': site['names']['worker'], 'worker_name': site['names']['node'],
        'cache_name': site['names']['cache'], 'cache_gib': site['resources']['cache_gib'],
        'chunk_size': 1536, 'mode_chunk_sizes': mode_chunks,
        'minimum_free_gib': site['resources']['minimum_free_gib'],
        'stop_timeout': 300, 'cache_timeout': 600, 'start_timeout': 1200,
        'profile_sha256': hashes, 'cache_definition_sha256': hashlib.sha256(result['cache-service.json']).hexdigest(),
        'image_ids': image_ids, 'production_root': str(target), 'source_repository': str(ROOT),
        'mode_gpu_ids': mode_gpus, 'hardware_gpu_ids': sorted(site['gpus']['eight_card_order']),
        'mode_model_paths': mode_models, 'model_lock_sha256': model_locks,
        'previous_runtime_images': {}, 'control_plane_images': {}, 'helper_sha256': {},
        'initial_mode': site['default_mode']}
    for name in ('gpustack-local-api.py', 'entrypoint/sync-upstream.py', 'verify-live-parameters.py', 'verify-v032-migration.py', 'verify-model-features.py'):
        configuration['helper_sha256'][name] = hashlib.sha256(result[name]).hexdigest()
    result['switch-config.json'] = bytes_json(configuration)
    composition = read_json(ROOT / 'templates' / ('compose-' + site['host']['kind'] + '.json'))
    result['compose.yaml'] = bytes_json(composition)
    result['compose.legacy-api.yaml'] = bytes_json({'services': {'legacy-api': {
        'image': next(i['RepoDigests'][0] for i in read_json(ROOT / 'locks/platform-images.json')['images'] if i['reference'].startswith('nginx')),
        'pull_policy': 'never', 'container_name': site['names']['proxy'], 'restart': 'unless-stopped',
        'runtime': 'runc', 'network_mode': 'host', 'user': '101:101', 'read_only': True,
        'cap_drop': ['ALL'], 'security_opt': ['no-new-privileges:true'], 'mem_limit': '128m', 'cpus': 1,
        'tmpfs': ['/tmp:rw,noexec,nosuid,size=32m,mode=1777'], 'volumes': ['./entrypoint:/config:ro'],
        'entrypoint': ['nginx'], 'command': ['-c', '/config/nginx.conf', '-g', 'daemon off;']}}})
    platform = read_json(ROOT / 'locks/platform-images.json')['images']
    gpustack = next(i['RepoDigests'][0] for i in platform if i['reference'].startswith('gpustack/gpustack'))
    configuration['control_plane_images'] = {site['names'][key]: image_ids[gpustack] for key in ('server', 'worker')}
    result['switch-config.json'] = bytes_json(configuration)
    env = {'COMPOSE_PROJECT_NAME': site['names']['project'], 'GPUSTACK_IMAGE': gpustack,
           'GPUSTACK_SERVER_CONTAINER': site['names']['server'], 'GPUSTACK_WORKER_CONTAINER': site['names']['worker'],
           'GPUSTACK_HOST_IP': site['host']['address'], 'GPUSTACK_HTTP_PORT': str(site['ports']['management']),
           'GPUSTACK_WORKER_NAME': site['names']['node'], 'GPUSTACK_KV_TRANSFER_IFNAME': site['host']['interface'],
           'GPUSTACK_MODEL_PARENT': site['paths']['model_parent'], 'GPUSTACK_SERVER_VOLUME': site['names']['server_volume'],
           'GPUSTACK_WORKER_VOLUME': site['names']['worker_volume'], 'GPUSTACK_TOKEN': '',
           'GPUSTACK_WORKER_PORT': str(site['ports']['worker_api']),
           'GPUSTACK_WORKER_METRICS_PORT': str(site['ports']['worker_metrics']),
           'GPUSTACK_SERVICE_PORT_RANGE': str(site['ports']['service_start']) + '-' + str(site['ports']['service_end'])}
    # 按摘要拉取不会自动生成可变标签；运行库必须直接引用已取得的固定摘要。
    env['GPUSTACK_PAUSE_IMAGE'] = next(i['RepoDigests'][0] for i in platform if i['reference'] == 'gpustack/runtime:pause')
    env['GPUSTACK_HEALTH_IMAGE'] = next(i['RepoDigests'][0] for i in platform if i['reference'] == 'gpustack/runtime:health')
    # Compose的单引号值不展开美元变量；保留路径中的空格、井号和单引号。
    result['.env'] = ''.join(key + "='" + value.replace('\\', '\\\\').replace("'", "\\'") + "'\n"
                             for key, value in env.items()).encode()
    result['entrypoint/upstream.conf'] = b'upstream legacy_model { server 127.0.0.1:9; keepalive 64; }\nupstream legacy_metadata { server 127.0.0.1:9; keepalive 8; }\n'
    result['entrypoint/metrics.prom'] = b'# No samples collected yet.\n'
    return result


def prepare(site_path, *, execute=False, configure_only=False, image_reader=inspect_images):
    site, target = checked_site(site_path)
    if not execute:
        return {'dry_run': True, 'environment_valid': True, 'will_modify_running_services': False}
    image_ids = image_reader(site)
    # 先逐一检查所有受管路径，拒绝借现有子目录或文件链接写入其他目录。
    # 登记阶段保留的令牌文件也检查路径，但不读取或覆盖其内容。
    files = generated_files(site, target, image_ids)
    for name in (*files, 'install-state.json'):
        safe_path(target, name)
    marker = safe_path(target, 'install-state.json')
    previous = read_json(marker) if marker.exists() else None
    if configure_only:
        if not previous or previous.get('generator') != 'gpustack-deployment-guide-v1':
            raise ValueError('不能适配没有生成记录的目录')
        if any(v <= 0 for v in site['identity'].values()):
            raise ValueError('登记阶段必须提供网页产生的真实正整数编号')
        for name, checksum in previous['generated_files'].items():
            path = safe_path(target, name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError('已生成文件被手工修改，拒绝部分覆盖：' + name)
    elif target.exists() and any(target.iterdir()):
        raise ValueError('初次准备只允许空目录')
    elif any(v != 0 for v in site['identity'].values()):
        raise ValueError('初次准备不能猜测资源编号；请明确填写0')
    # 后续登记仅更新本工具负责的配置，不覆盖已经填写的注册令牌或采集器的运行文件。
    mutable = {'.env', 'entrypoint/upstream.conf', 'entrypoint/metrics.prom'}
    preserved = mutable if configure_only else set()
    target.mkdir(parents=True, exist_ok=True)
    generated = {}
    for name, data in files.items():
        if name in preserved:
            continue
        path = safe_path(target, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o755 if name.endswith('.sh') else 0o600 if name == '.env' else 0o644)
        if name not in mutable:
            generated[name] = hashlib.sha256(data).hexdigest()
    marker.write_bytes(bytes_json({'format': 1, 'generator': 'gpustack-deployment-guide-v1',
                                  'generated_files': generated}))
    marker.chmod(0o600)
    return {'generated': True, 'target': str(target), 'private_output': True, 'running_services_modified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', required=True, help='仓库之外的本次环境文件')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--configure-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.site, execute=args.execute, configure_only=args.configure_only), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
