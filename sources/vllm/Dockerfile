# 固定官方 nightly 基础；此候选不改变 Torch、CUDA、NIXL 或 LMCache 的版本。
ARG BASE_IMAGE=vllm/vllm-openai@sha256:6b90105076216e861cd30fae8828efef9263a0399137b894289f3b6a685bf4ff
FROM ${BASE_IMAGE}
ENTRYPOINT []
CMD []

# 保留 GPUStack 的异常握手保护，并只重放 FlashInfer 官方的 Python 层差异。
RUN test -x /usr/bin/python3 && if ! command -v python >/dev/null 2>&1; then ln -s /usr/bin/python3 /usr/local/bin/python; fi
COPY nightly-patches/ /opt/gpustack-v032/patches/
COPY flashinfer6204/ /opt/gpustack-v032/flashinfer6204/
RUN python3 /opt/gpustack-v032/patches/patch-nixl-handshake.py && \
    python3 /opt/gpustack-v032/patches/verify-nixl-handshake.py && \
    python3 /opt/gpustack-v032/flashinfer6204/patch-flashinfer-nvfp4.py --execute && \
    python3 /opt/gpustack-v032/flashinfer6204/verify-flashinfer-nvfp4.py

LABEL local.gpustack.purpose="Qwen3.8-27B SM120 native NVFP4 KV + MTP + NIXL/LMCache" \
      local.gpustack.vllm-version="0.32.1rc1.dev466+ge4a5b5537" \
      local.gpustack.patch-scope="nixl-malformed-handshake-guard,flashinfer-6204" \
      local.gpustack.flashinfer-commit="3821440929ee342fe9f939dbf5186c3e667a851c"
