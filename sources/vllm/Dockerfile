# 固定官方 nightly 基础；此候选不改变 Torch、CUDA、NIXL 或 LMCache 的版本。
ARG BASE_IMAGE=vllm/vllm-openai@sha256:6b90105076216e861cd30fae8828efef9263a0399137b894289f3b6a685bf4ff
FROM ${BASE_IMAGE}
ENTRYPOINT []
CMD []
# 构建主机可能以nvidia作为Docker默认运行时；屏蔽设备直到所有CPU检查结束。
ENV NVIDIA_VISIBLE_DEVICES=void

# 保留 GPUStack 的异常握手保护，并只重放 FlashInfer 官方的 Python 层差异。
RUN test -x /usr/bin/python3 && if ! command -v python >/dev/null 2>&1; then ln -s /usr/bin/python3 /usr/local/bin/python; fi
COPY nightly-patches/ /opt/gpustack-v032/patches/
COPY flashinfer6204/ /opt/gpustack-v032/flashinfer6204/
RUN python3 /opt/gpustack-v032/patches/patch-nixl-handshake.py && \
    python3 /opt/gpustack-v032/patches/verify-nixl-handshake.py && \
    python3 /opt/gpustack-v032/flashinfer6204/patch-flashinfer-nvfp4.py --execute && \
    python3 /opt/gpustack-v032/flashinfer6204/verify-flashinfer-nvfp4.py

# 只回移已登记的上游修复；逐步校验源码身份，不能在未知版本上模糊套用差异。
# 构建中的检查只使用CPU，不访问正在运行的模型或GPU。
COPY optimizations/ /opt/gpustack-v032/optimizations/
RUN NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES=-1 python3 /opt/gpustack-v032/optimizations/apply-patches.py --execute && \
    NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES=-1 python3 /opt/gpustack-v032/optimizations/verify-cpu.py && \
    python3 -c 'import pathlib,shutil,sysconfig; p=pathlib.Path(sysconfig.get_paths()["purelib"]); shutil.copyfile("/opt/gpustack-v032/optimizations/cache-numa.py",p/"gpustack_cache_numa.py"); (p/"gpustack_cache_numa.pth").write_text("import gpustack_cache_numa; gpustack_cache_numa.activate()\n")'

# 官方LMCache预编译包使用另一Torch版本；在原工具链内重建同版本CUDA扩展，
# 不能让导入失败后的Torch回退静默承担不支持的blocks-first缓存布局。
COPY lmcache-cuda/ /opt/gpustack-v032/lmcache-cuda/
COPY lmcache-0.5.6.tar.gz /opt/gpustack-v032/lmcache-cuda/
RUN NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES=-1 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=1 NVCC_THREADS=1 ENABLE_CXX11_ABI=1 python3 /opt/gpustack-v032/lmcache-cuda/build-lmcache-cuda.py

# LMCache源码安装完成后再应用纯Python查询推进修复，避免重建扩展时覆盖补丁。
# 元数据回调不受本轮计算预算限制；CPU验证不连接生产缓存或初始化CUDA。
COPY lookup-progress/ /opt/gpustack-v032/lookup-progress/
RUN python3 -B /opt/gpustack-v032/lookup-progress/apply-patch.py --execute && \
    python3 -B /opt/gpustack-v032/lookup-progress/verify-cpu.py

# 最终镜像由GPUStack按角色提供设备限制；不把构建时的屏蔽值带入运行配置。
ENV NVIDIA_VISIBLE_DEVICES=all
LABEL local.gpustack.purpose="Qwen3.8-27B SM120 native NVFP4 KV + MTP + NIXL/LMCache" \
      local.gpustack.vllm-version="0.32.1rc1.dev466+ge4a5b5537" \
      local.gpustack.patch-scope="nixl-malformed-handshake-guard,flashinfer-6204,upstream-backports-20261028" \
      local.gpustack.optimization-release="e4a5b5537-opt-20261028" \
      local.gpustack.lmcache-cuda-abi="torch-2.14.0-cu130-sm120" \
      local.gpustack.lookup-progress="lmcache-result-progress-20261028" \
      local.gpustack.flashinfer-commit="3821440929ee342fe9f939dbf5186c3e667a851c"
