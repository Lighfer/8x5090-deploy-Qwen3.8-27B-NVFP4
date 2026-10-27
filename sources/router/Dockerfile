# Router的重新构建使用当前官方推理环境，避免必须保留历史0.31自定义基础镜像。
# Rust扩展先按锁定的源码、SDK和Cargo.lock编译；此处只安装同版Python启动器并替换扩展。
ARG BASE_IMAGE=vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
FROM ${BASE_IMAGE}
ENTRYPOINT []
CMD []
COPY LICENSE /opt/router-build/LICENSE
COPY wheels/ /opt/router-build/wheels/
RUN python3 -m pip install --no-index --no-deps --require-hashes \
    --find-links=/opt/router-build/wheels -r /opt/router-build/wheels/router-requirements.txt && \
    rm -rf /opt/router-build/wheels
COPY libvllm_router_rs.so verify-native-router.py verify-routing-mock.py /opt/router-build/
# 通过实际导入位置确定扩展目录，不把旧虚拟环境的安装路径套到新官方镜像。
RUN python3 -c 'import importlib.util,shutil; p=importlib.util.find_spec("vllm_router_rs").origin; shutil.copyfile("/opt/router-build/libvllm_router_rs.so",p)' && \
    python3 /opt/router-build/verify-native-router.py && \
    python3 /opt/router-build/verify-routing-mock.py \
      --site /usr/local/lib/python3.12/dist-packages --output /opt/router-build/verification.json && \
    rm /opt/router-build/libvllm_router_rs.so
LABEL local.gpustack.purpose="Router0.1.16 with explicit PD stage routing and cache salt isolation"
