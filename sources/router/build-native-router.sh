#!/bin/sh
# 所有依赖只安装到本次临时容器；宿主网关及其配置、数据卷均不挂载。
set -eu
export DEBIAN_FRONTEND=noninteractive
# 只在临时容器内把官方源改为 HTTPS，仍由 APT 校验原有签名。
for source_file in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; do
  if [ -f "$source_file" ]; then
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' "$source_file"
  fi
done
apt-get -o Acquire::Retries=1 -o Acquire::https::Timeout=30 update
# 不安装系统 ZeroMQ 开发包，让原锁定的 zmq-sys 使用其自带静态构建，避免给运行镜像新增 libzmq 依赖。
apt-get install -y --no-install-recommends python3-dev protobuf-compiler libclang-dev cmake pkg-config libssl-dev
cd /build
printf '%s  %s\n' b0f953025ca81694f5212d77e1e1201557175d1e45c54e3b3d41e6015324379c vllm_router-0.1.16.tar.gz | sha256sum -c -
test ! -d vllm_router-0.1.16
tar -xzf vllm_router-0.1.16.tar.gz
python3 /build/patch-router-pd-type.py
python3 /build/patch-router-pd-routing.py
cd /build/vllm_router-0.1.16
# 补丁源码和 Cargo.lock 必须与已经验收的固定输入一致，防止上下文替换意外漏项。
sha256sum -c /build/patched-source.sha256
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export PYO3_PYTHON=/usr/bin/python3
# ABI3 保持与原 wheel 的 CPython 3.8+ 扩展接口一致；全部 Rust 依赖固定于原 Cargo.lock。
# 从刚修正的实际源码提取分类函数和回归模块，用 rustc 运行；避免 Python 扩展测试
# 额外链接特定 Python 版本，也避免将整个 crate 分别做一遍调试和发布构建。
python3 - <<'PY'
from pathlib import Path
source=Path('src/routers/mod.rs').read_text()
start=source.index('// 路由类型来自各原生实现')
Path('/build/pd-type-test.rs').write_text(source[start:])
PY
rustc --test /build/pd-type-test.rs -o /build/pd-type-test
/build/pd-type-test
cargo build --locked --release --lib --features pyo3/abi3-py38
sha256sum target/release/libvllm_router_rs.so
ldd target/release/libvllm_router_rs.so
cp target/release/libvllm_router_rs.so /build/libvllm_router_rs.so
