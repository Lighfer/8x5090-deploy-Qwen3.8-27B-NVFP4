"""合成环境夹具；所有值只存在于内存或系统临时目录，不来自任何真实服务器。"""
import hashlib
import importlib.util
import ipaddress
from pathlib import Path
import secrets
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('configure', ROOT / 'tools/configure.py')
configure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configure)


def fake_site(base, *, shifted=False):
    """使用不同名称、非连续GPU序号与不同端口，防止只在某台原始主机上通过。"""
    base = Path(base)
    identity = 'test-' + secrets.token_hex(4)
    host = str(ipaddress.IPv4Address(secrets.randbits(32)))
    start = 20000 + secrets.randbelow(10000)
    gpus = [19, 3, 11, 7, 23, 5, 29, 13]
    if shifted:
        gpus = [v + 30 for v in reversed(gpus)]
    names = {key: identity + '-' + key.replace('_', '-') for key in (
        'project', 'server', 'worker', 'proxy', 'cluster', 'node', 'worker_label', 'model', 'cache', 'server_volume', 'worker_volume')}
    return {'format': 1, 'host': {'address': host, 'interface': identity, 'kind': 'lxc'},
        'paths': {'deployment': str(base / 'private deployment'), 'model_parent': str(base / 'models'),
                  'model_directory': str(base / 'models/model')},
        'ports': {'management': start, 'inference': start + 1, 'worker_api': start + 2,
                  'worker_metrics': start + 3, 'service_start': start + 100, 'service_end': start + 199},
        'names': names, 'gpus': {'eight_card_order': gpus, 'six_card_order': gpus[:6]},
        'resources': {'cache_gib': 32 if shifted else 24, 'minimum_free_gib': 8},
        'images': {key: 'testing/' + identity + ':' + key for key in ('runtime', 'cache', 'router')},
        'identity': {'model_id': 0, 'worker_id': 0, 'cluster_id': 0},
        'admin_user': identity, 'default_mode': '1p1d-no-cache'}


def fake_images(site):
    """内容身份来自合成引用，不依赖本机Docker是否安装或是否已有镜像。"""
    platform = configure.read_json(ROOT / 'locks/platform-images.json')['images']
    refs = {*site['images'].values(), next(i['RepoDigests'][0] for i in platform if i['reference'].startswith('gpustack/gpustack'))}
    return {value: 'sha256:' + hashlib.sha256(value.encode()).hexdigest() for value in refs}


def generated_fixture(root):
    """生成隔离管理器夹具；这里的编号是测试协议中的合成资源，不是现场编号。"""
    site = fake_site(Path(root).parent)
    site['paths']['deployment'] = str(root)
    site['identity'] = {'model_id': 1, 'worker_id': 1, 'cluster_id': 1}
    files = configure.generated_files(site, Path(root), fake_images(site))
    for name, content in files.items():
        path = Path(root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return site
