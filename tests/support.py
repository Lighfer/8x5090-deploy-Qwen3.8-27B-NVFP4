"""合成环境夹具；所有值只存在于内存或系统临时目录，不来自任何真实服务器。"""
import hashlib
import importlib.util
import ipaddress
import json
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
                  'model_directory': str(base / 'models/model'),
                  'fp8_model_directory': str(base / 'models/model-fp8')},
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
    # 模型文件只使用当前临时安装目录中的合成内容，不创建真实权重或共享临时目录。
    site['paths'].update(model_parent=str(Path(root) / 'fixture-models'),
                         model_directory=str(Path(root) / 'fixture-models/nvfp4'),
                         fp8_model_directory=str(Path(root) / 'fixture-models/fp8'))
    site['identity'] = {'model_id': 1, 'worker_id': 1, 'cluster_id': 1}
    files = configure.generated_files(site, Path(root), fake_images(site))
    for name, content in files.items():
        path = Path(root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    configuration = configure.read_json(Path(root) / 'switch-config.json')
    for precision in ('nvfp4', 'fp8'):
        directory = Path(site['paths']['fp8_model_directory' if precision == 'fp8' else 'model_directory'])
        directory.mkdir(parents=True)
        quant = {'quant_method': 'fp8', 'weight_block_size': [128, 128]} if precision == 'fp8' else {'quant_method': 'modelopt'}
        config = {'quantization_config': quant, 'text_config': {'intermediate_size': 17408,
                  'num_key_value_heads': 4, 'max_position_embeddings': 262144}}
        metadata = {'config.json': config, 'model.safetensors.index.json': {'weight_map': {'model.weight': 'model.safetensors', 'mtp.weight': 'model.safetensors'}}}
        for name in ('generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                     'merges.txt', 'preprocessor_config.json', 'video_preprocessor_config.json', 'chat_template.jinja'):
            metadata[name] = {}
        rows = []
        for name, value in metadata.items():
            data = json.dumps(value).encode()
            (directory / name).write_bytes(data)
            rows.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        data = b'synthetic-weights'
        (directory / 'model.safetensors').write_bytes(data)
        rows.append({'path': 'model.safetensors', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        lock = (json.dumps({'files': rows}) + '\n').encode()
        (Path(root) / 'model-inputs' / (precision + '.json')).write_bytes(lock)
        configuration['model_lock_sha256'][precision] = hashlib.sha256(lock).hexdigest()
    (Path(root) / 'switch-config.json').write_text(json.dumps(configuration))
    return site
