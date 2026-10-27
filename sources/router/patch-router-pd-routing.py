"""修正固定 Router 0.1.16 的多路由阶段策略和动态缓存成员，保留原生请求及管理接口。"""
import argparse
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path('/build/vllm_router-0.1.16'),
                    help='构建容器内源码约定目录，或外部独立核验目录')
root = parser.parse_args().root


def replace(relative, before, after, count=1):
    path = root / relative
    text = path.read_text()
    if text.count(before) != count:
        raise RuntimeError('Unexpected source anchor in ' + relative)
    original = hashlib.sha256(text.encode()).hexdigest()
    text = text.replace(before, after)
    path.write_text(text)
    print(relative + ' before=' + original + ' after=' + hashlib.sha256(text.encode()).hexdigest(), flush=True)


# 多路由只是成员管理方式；启用它时仍须保留 vLLM PD 的阶段策略配置。
replace('src/lib.rs', '''        let mode = if self.enable_igw {
            // IGW mode - routing mode is not used in IGW, but we need to provide a placeholder
            RoutingMode::Regular {
                worker_urls: vec![],
            }
        } else if self.vllm_pd_disaggregation {''', '''        let mode = if self.vllm_pd_disaggregation {''')
replace('src/lib.rs', '''                discovery_address: self.vllm_discovery_address.clone(),
            }
        } else {
            RoutingMode::Regular {''', '''                discovery_address: self.vllm_discovery_address.clone(),
            }
        } else if self.enable_igw {
            // 普通多路由仍从空成员表启动，随后通过原生成员接口注册。
            RoutingMode::Regular { worker_urls: vec![] }
        } else {
            RoutingMode::Regular {''')

# IGW 仍从空成员表创建 PD Router，只补回被遗漏的两侧策略，不预先重复注册地址。
replace('src/server.rs', '''            // 2. HTTP vLLM PD Router
            match RouterFactory::create_vllm_pd_router(
                &[],
                &[],
                None,
                None,
                None,''', '''            // 多路由保留原生动态成员管理，同时应用配置中的 P/D 独立策略。
            let (prefill_policy, decode_policy) = match &config.router_config.mode {
                crate::config::RoutingMode::VllmPrefillDecode { prefill_policy, decode_policy, .. } =>
                    (prefill_policy.as_ref(), decode_policy.as_ref()),
                _ => (None, None),
            };
            // 2. HTTP vLLM PD Router
            match RouterFactory::create_vllm_pd_router(
                &[],
                &[],
                None,
                prefill_policy,
                decode_policy,''')

replace('src/routers/factory.rs', '''        ctx.policy_registry.set_decode_policy(decode_policy);
''', '''        ctx.policy_registry.set_decode_policy(decode_policy);
        // 输出真正安装到注册表的阶段策略，便于区分声明参数与实际运行行为。
        tracing::info!("PD stage policies active: prefill={}, decode={}",
            ctx.policy_registry.get_prefill_policy().name(),
            ctx.policy_registry.get_decode_policy().name());
''')

replace('src/routers/router_manager.rs', '''        let policy = self.policy_registry.on_worker_added(&model_id, policy_hint);
''', '''        let policy = self.policy_registry.on_worker_added(&model_id, policy_hint);
        // 成员动态加入时为有状态策略建立对应模型的记录；不能一直退回随机选择。
        if policy.requires_initialization() {
            policy.init_workers(&[worker_arc.clone()]);
        }
        let stage_policy = match worker_arc.worker_type() {
            WorkerType::Prefill { .. } => Some(self.policy_registry.get_prefill_policy()),
            WorkerType::Decode => Some(self.policy_registry.get_decode_policy()),
            _ => None,
        };
        if let Some(stage_policy) = stage_policy {
            if stage_policy.requires_initialization() {
                stage_policy.init_workers(&[worker_arc.clone()]);
            }
        }
''')
replace('src/routers/router_manager.rs', '''        if let Some(_worker) = self.worker_registry.remove_by_url(url) {
            // Notify PolicyRegistry about worker removal
            if let Some(ref model_id) = model_id {
                self.policy_registry.on_worker_removed(model_id);''', '''        if let Some(worker) = self.worker_registry.remove_by_url(url) {
            // 同步移除缓存策略中的失效成员，保留其他实例已经学习到的前缀记录。
            let stage_policy = match worker.worker_type() {
                WorkerType::Prefill { .. } => Some(self.policy_registry.get_prefill_policy()),
                WorkerType::Decode => Some(self.policy_registry.get_decode_policy()),
                _ => None,
            };
            if let Some(stage_policy) = stage_policy {
                if let Some(cache) = stage_policy.as_any().downcast_ref::<crate::policies::CacheAwarePolicy>() {
                    cache.remove_worker(worker.as_ref());
                }
            }
            // Notify PolicyRegistry about worker removal
            if let Some(ref model_id) = model_id {
                let policy = self.policy_registry.get_policy_or_default(model_id);
                if let Some(cache) = policy.as_any().downcast_ref::<crate::policies::CacheAwarePolicy>() {
                    cache.remove_worker(worker.as_ref());
                }
                self.policy_registry.on_worker_removed(model_id);''')

# 缓存隔离值必须参与路由前缀的最前部；只改变路由估计键，不改发给 P/D 的请求正文。
replace('src/routers/http/vllm_pd_router.rs', 'impl VllmPDRouter {', '''impl VllmPDRouter {
    fn routing_request_text(request: &Value) -> Option<String> {
        match request.get("cache_salt").filter(|value| !value.is_null()) {
            Some(salt) => serde_json::to_string(&(salt, request)).ok(),
            None => serde_json::to_string(request).ok(),
        }
    }
''')
replace('src/routers/http/vllm_pd_router.rs', 'let request_text = serde_json::to_string(&request_json).ok();',
    'let request_text = Self::routing_request_text(&request_json);', count=4)
