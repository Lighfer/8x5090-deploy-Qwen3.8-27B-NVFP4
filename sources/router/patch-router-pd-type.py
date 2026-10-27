"""对固定版本 Router 做最小类型识别修正，保留多路由与成员管理。"""
import argparse
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path('/build/vllm_router-0.1.16'),
                    help='构建容器内源码约定目录，或外部独立核验目录')
root = parser.parse_args().root
path=root/'src/routers/mod.rs'
source=path.read_text()
anchor='''    fn is_pd_mode(&self) -> bool {
        self.router_type() == "pd"
    }'''
replacement='''    fn is_pd_mode(&self) -> bool {
        // vLLM 专用路由返回 vllm_pd；与通用 pd 一样需要参与 PD 选择和观测。
        is_pd_router_type(self.router_type())
    }'''
helper='''
// 路由类型来自各原生实现；普通路由必须继续保持非 PD，避免改变其默认选择。
fn is_pd_router_type(router_type: &str) -> bool {
    matches!(router_type, "pd" | "vllm_pd")
}

#[cfg(test)]
mod pd_type_regression {
    use super::is_pd_router_type;

    #[test]
    fn native_vllm_and_generic_pd_are_recognized() {
        assert!(is_pd_router_type("vllm_pd"));
        assert!(is_pd_router_type("pd"));
    }

    #[test]
    fn regular_and_unknown_types_remain_non_pd() {
        assert!(!is_pd_router_type("regular"));
        assert!(!is_pd_router_type("openai"));
        assert!(!is_pd_router_type("unknown"));
    }
}
'''
if source.count(anchor)!=1:raise RuntimeError('Unexpected Router trait source')
before=hashlib.sha256(source.encode()).hexdigest()
source=source.replace(anchor,replacement)+helper
path.write_text(source)
print('Router trait SHA256 before='+before+' after='+hashlib.sha256(source.encode()).hexdigest(),flush=True)
