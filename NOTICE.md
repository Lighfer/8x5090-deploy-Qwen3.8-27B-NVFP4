# 第三方来源与许可

本仓库自行编写的工具和文档采用[MIT许可证](LICENSE)。引用、修改和构建的第三方软件保留其原有许可证、版权与署名；本仓库的MIT许可不改变第三方材料的许可。

- [GPUStack](https://github.com/gpustack/gpustack)：固定2.3.0；内存采集原始模块、相应差异及上游LICENSE位于`sources/worker-memory/`，Copyright (c) 2024 The GPUStack authors。
- [GPUStack Runtime](https://github.com/gpustack/runtime)：固定0.2.6；原始Docker部署模块及其许可位于`sources/worker-runtime/`，仅增加显式选择的PID命名空间处理。
- [vLLM](https://github.com/vllm-project/vllm)：固定提交和公开镜像摘要见`STACK.json`、`locks/rebuild-inputs.json`。源码片段与补丁用于NIXL异常握手保护。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：`sources/vllm/flashinfer6204/upstream.patch`保存官方#6204的Python差异，其来源提交在同目录`inputs.json`中。
- [vLLM Router](https://pypi.org/project/vllm-router/0.1.16/)：固定0.1.16，发布元数据署名Byron Hsu、声明Apache-2.0，并明确感谢其分支来源SGLang Model Gateway的作者。该源码包未打包LICENSE；本目录补充上游仓库公布的Apache-2.0文本，并在镜像中保留。原始源码与Cargo.lock的内容身份由散列核对。
- 模型文件包括模型自己的LICENSE。下载工具按锁定版本获取该文件；模型权重与模型许可不由本仓库的MIT许可替代。

源码、镜像、模型与SDK的公开取得方式见[构建与固定输入](docs/04-补丁与构建.md)。发行包不包含权重、镜像层、编译库或SDK。

修改声明：`templates/runtime/worker/fastfetch.py`是在GPUStack原始模块上应用LXC内存修正的版本；同目录`docker.py`是在Runtime原始模块上应用显式PID命名空间修正的版本。Router与NIXL补丁脚本内含修改后的源码片段；FlashInfer差异来自上述官方修复，未把第三方文件重新声明为MIT。
