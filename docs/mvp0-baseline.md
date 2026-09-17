# MVP-0：已验证基线与部署参考

[返回首页](../README.md) · [MVP-1 验证手册](mvp1-validation.md)

你当前的目标环境已完成 MVP-0 验证。本页保留安装、Ray/NPU 分配、固定权重生成和扩缩容操作，
用于重建环境或排查基础服务；开展 MVP-1 验证时可沿用已有配置，无需重跑整套 MVP-0 验收。
四节点 TP8 等其他拓扑的示例不代表已经在这些环境完成验收。

MVP-0 的范围是固定权重生成、租约路由、`2 → 4 → 2` 扩缩容、故障补建和资源清理。
本页的 AReaL workflow / verl AgentLoop 示例都不验证训练权重同步。

## 结构与边界

```text
AReaL ElasticInferenceEngine       verl ElasticLLMClient
               |                       |
               +--- Acquire/Release ---+----> Controller HTTP API
               |                       |       TrafficManager / RouteLease
               +--- tokens/logprobs ---+----> vLLM HTTP endpoints
                                               ^
Controller -> Reconciler -> ReplicaManager      |
                   -> StaticResourceProvider   |
                   -> RayProcessLauncher ------+
                   -> VLLMServiceDriver
```

- Core 不 import `areal`、`verl`；适配器在 `elastic_rollout/integrations/`，按需导入。
- Controller 不处理 prompt、token 或 logprob，没有 `/elastic/generate`。
- `GenerationClient` 是框架进程内的 HTTP 客户端和 token 编解码工具，不是服务端代理。
- Ray actor 按 slot 申请 `resources={"NPU": slot.num_devices}` 并固定到指定节点，vLLM 用本地 `mp` 启动 TP workers。
- 一个 replica 对应单节点逻辑 capacity slot，启动前要求 `num_devices == TP`。同节点 serving port 不得冲突，实际 NPU 集合由 Ray 分配。
- `spec.replica.model/policy_version` 是不可变 bootstrap 配置；当前已提交模型和版本由 `status.policy` 描述。

### 路由和缩容

新 key 只进入 READY；已有 key 在 drain grace 内可以继续进入其 DRAINING 副本。
首次选择使用 least-inflight；释放租约不删除 sticky mapping；mapping 为有界 LRU。
每次请求有独立租约，Release 幂等，客户端续租，Controller 自动回收过期租约。

grace 结束后移除 sticky mapping，等待租约归零，**再要求 vLLM `/metrics` 的
`num_requests_running` 和 `num_requests_waiting` 均为零**才停进程。
缺失/无效指标不会被当成空闲。这样客户端掉线或 HTTP 超时后的后台生成也受到保护。
发生主动取消时客户端等待有界 HTTP 操作结束，然后释放租约；因此取消不保证立即返回。

启动必须经过 health 和一次真实 completion warmup 才进入 READY。进程退出或连续三次
health 失败会撤出路由并替换。停止失败的 slot 保留占用，避免重复使用 NPU。
Ray actor 无法联系时不能证明物理进程已清理，slot 保持隔离并在状态里报告错误。

同一 deployment 的生命周期调和串行执行，desired state 可以重复设置。
扩缩方向变化时先完成已有启动/排空，再向最新目标收敛；不会反复创建重复副本。

## 安装与启动

运行时、所有 Ray worker 节点、框架环境都需要安装本包。Core 零第三方依赖；
Ray 是可选依赖，使用现有 Ascend/vLLM 环境，不在此安装或重建驱动栈。

```bash
cd elastic-rollout
uv pip install -e .
# 仅在运行时环境尚未安装 Ray 时：
uv pip install -e '.[ray]'
```

1. 准备模型可见的 Linux NPU 节点，按下面的 NOSET 步骤启动 Ray，节点公布真实 NPU 资源。原 TP=8 示例需四台八卡节点；单机 TP=2 示例只需一台八卡节点。
2. 从现有集群 `ray.nodes()` 取得 `NodeID` 和网络地址，填写
   [runtime.example.json](../examples/runtime.example.json) 的所有 `REPLACE_*`，保存为 `runtime.json`。
3. 确认模型路径、Python/vLLM 版本、可见 NPU ID、端口均与实际环境一致；运行时与框架必须能直连各节点的 vLLM 端口。
4. Controller 只启动一个进程，默认监听 localhost；跨节点访问时显式指定绑定地址。

```bash
python -m elastic_rollout.cli serve --config runtime.json --host 0.0.0.0 --port 8080
```

可通过 `ELASTIC_ROLLOUT_TOKEN` 为 control API 设置 Bearer token；框架环境使用同一变量。
MVP 服务面向可信集群网络，不包含 TLS、HA 或 Controller 状态持久化。vLLM 端点也需限制在集群网络内。

```bash
elastic-rollout status --endpoint http://CONTROL_HOST:8080 --deployment mvp0
elastic-rollout scale --endpoint http://CONTROL_HOST:8080 --deployment mvp0 4 --wait
elastic-rollout scale --endpoint http://CONTROL_HOST:8080 --deployment mvp0 2 --wait
elastic-rollout scale --endpoint http://CONTROL_HOST:8080 --deployment mvp0 0 --wait
```

Ctrl-C 会等待排空和停止。正常停止时进程 guardian 会清理 vLLM 进程组；Ray actor 异常退出
时 guardian 通过父进程管道 EOF 触发清理。不可达节点仍需人工确认残留进程后恢复资源。
进程组清理机制仅支持 Linux，未在本次 Windows 环境进行实机验证。

### 单台 8-NPU：四个 TP=2 副本（MVP-0.1）

使用 [runtime.single-node-tp2.json](../examples/runtime.single-node-tp2.json)：四个 slot 共用真实
NodeID/IP，各声明 `num_devices: 2`，端口为 8000–8003。
每个 actor 申请 2 NPU / 2 CPU，节点应有足够的 Ray CPU 资源；TP 在 deployment 内固定，PP=1。
填好占位符后保存为 `runtime.json`，按上面的 serve/scale 命令验证 `2 → 4 → 2`。
四节点 TP=8 示例也已改为 `num_devices: 8`，框架接入方式不变。

**配置迁移：删除旧 `device_ids` 字段，改为 `num_devices`。** 不再接受静态物理设备绑定；
slot 只描述节点、NPU 数量和服务端口，Ray 负责实际设备分配与总容量 accounting。
四个 slot 不预分配设备集合；正确公布 8 NPU 的 Ray 节点最多同时容纳四个 TP=2 actor。

### Ray NOSET 模式与设备归属

先正常停止旧 ElasticRollout 服务，再在启动 Ray 的环境中设置：

```bash
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray stop --force
ray start --head --resources='{"NPU":8}'
```

这是单机八卡示例；保留实际环境所需的网络参数，公布的 NPU 数量必须与可用设备一致。
多节点部署时，每个 Ray 节点启动前都设置对应节点的环境，Replica 本身仍只运行在单节点。
**修改这些变量后必须重新启动 Ray cluster；只在运行中的 actor 或 serve shell 设置不会更新已有 Ray worker。**
重启后重新读取 `ray.nodes()` 的 NodeID/IP 并更新 `runtime.json`；所有 worker 使用同一版组件。
这个启动时机要求也见 [vLLM-Ascend 官方 Ray 启动说明](https://docs.vllm.ai/projects/ascend/en/v0.8.4rc1/tutorials/multi_node.html#start-ray-cluster)。

启动链路为：

```text
Ray get_runtime_context().get_accelerator_ids()["NPU"] -> ["5", "4"]
ElasticRollout sorted(assigned_ids, key=int)           -> ["4", "5"]
guardian / vLLM subprocess ASCEND_RT_VISIBLE_DEVICES  -> 4,5
```

NOSET 下 actor 可能继承整机可见设备；它不是实际分配的依据。
launcher 检查 Ray 返回的数量、数字 ID 和唯一性，只给子进程设置排序后的同一设备集合；
不修改 actor 环境、不在 actor 内导入 torch_npu 或初始化 NPU。排序仅作用于当前节点，
不设计跨节点设备顺序。guardian 原样继承环境并传给 vLLM，不改变进程组清理机制。
无需依赖 actor 启动顺序匹配 slot 编号；重试时同一 slot 得到不同卡组是合法的。

迁移 Ray/CANN/vLLM 版本或部署环境时，需要重新检查 NOSET 配置和首次启动日志。
不能将重试后 READY 视为首次初始化故障已经消失。

### MVP-0 基础控制 API

所有路径均以 `/v1/deployments/{deployment_id}` 开头。

| 方法 | 后缀 | 含义 |
|---|---|---|
| PUT | 空 | EnsureDeployment，body 为 DeploymentSpec |
| GET | 空 | spec、desired/ready/available/total、inflight、last_error |
| PUT | `/desired` | `{"desired_replicas": 4}` |
| GET | `/replicas` | 当前及历史副本、状态迁移、资源和错误 |
| POST | `/routes` | `{"routing_key": "opaque-key"}`，返回 RouteLease |
| DELETE | `/leases/{lease_id}` | 幂等释放 |
| PUT | `/leases/{lease_id}` | 续租；过期租约不能复活 |

`ready` 只计 READY；`available` 计 READY+DRAINING；`total` 计所有非 FAILED/TERMINATED 副本。
历史失败副本的资源是否已释放，要查看 `/replicas` 中的 `resource`；它可能仍处于隔离状态。
每个副本的 `allocation` 保留最近获知的 Ray 分配，例如：

```json
{"slot_id": "slot0", "num_devices": 2, "actual_device_ids": ["4", "5"]}
```

即使 FAILED/TERMINATED 副本的 `resource` 已变成 null，这份 `allocation` 仍保留。
尚未调度或 actor 在回传分配前失联时，`actual_device_ids` 可能为空；不根据 slot 猜测设备。
历史仅存在当前 Controller 内存中，重启不恢复。
Ensure 不重置已有 desired state，不允许原地修改模型、policy 或 replica shape。

## AReaL 接入（固定权重生成）

目标源码：相邻 AReaL `feat/elastic-rollout`，基于此前 Ascend fork 的外部推理适配。
已在 `areal/infra/controller/rollout_controller.py` 增加 capability hook。

```python
from areal.api.cli_args import InferenceEngineConfig
from elastic_rollout.integrations.areal.inference_engine import ElasticInferenceEngine

config = InferenceEngineConfig(
    backend="vllm:d1",
    consumer_batch_size=2,
    max_concurrent_rollouts=8,
    max_head_offpolicyness=10**12,  # 固定权重离线 workflow，沿用框架 eval 的做法
)
engine = ElasticInferenceEngine(config)
engine.initialize(addr="http://CONTROL_HOST:8080", deployment_id="mvp0")
# await engine.agenerate(ModelRequest(...))，或原有 submit/rollout_batch(workflow=...)
# engine.destroy() 只清理框架 workflow executor。
```

多个固定 CPU workflow worker：

```python
controller = ElasticInferenceEngine.as_controller(config, scheduler)
controller.initialize(
    "rollout", workflow_worker_count=2,
    addr="http://CONTROL_HOST:8080", deployment_id="mvp0",
)
# scheduler 仍由框架提供；不再传 server_args/server_infos。
```

`req.rid` 原样作为 routing key；每 token 的 `output_versions` 来自租约。
保留 RemoteInfEngine 的 workflow/submit/callback 实现，但重新绑定 WorkflowExecutor 到
Elastic 子类，重写 initialize/agenerate，永不调用继承的物理 server 路由或启动。
普通 AgentConfig 默认值可保留；真正的 Agent proxy、VLM、LoRA、expert replay 显式报不支持。

**固定权重的离线 workflow 要显式放宽 max_head_offpolicyness。** 原框架的 staleness
capacity 会扣除累计 accepted 样本数，默认配置没有训练版本推进时会在容量耗尽后等待。
此限制属于框架，本组件不伪造新版本绕过它。

完整磁盘更新训练入口见 [MVP-1 验证手册](mvp1-validation.md)；裸 AReaL `PPOTrainer` 仍选择其原生推理引擎。

## verl 接入（固定权重生成）

相邻 verl 为 `feat/elastic-rollout`，基于 v0.9.0 的既有外部 AgentLoop 适配。
本次不修改 verl 仓库或原有路由架构。

```python
from elastic_rollout.integrations.verl.rollout_service import ElasticRolloutService
from verl.experimental.agent_loop import AgentLoopManager

# config 是框架的完整配置，额外信息放进已有 rollout.custom 字段：
# actor_rollout_ref.rollout.custom.elastic = {
#   "endpoint": "http://CONTROL_HOST:8080",
#   "deployment_id": "mvp0", "fixed_policy": True,
# }
# 在 async 函数中调用：
service = await ElasticRolloutService.create(config=config)
manager = await AgentLoopManager.create(config=config, llm_client=service.get_client())
# manager.generate_sequences(...) 沿用现有 AgentLoop。
```

不创建 native RolloutReplica、原生负载均衡器或 CheckpointEngine。
客户端保留 TokenOutput 和 global_steps/min/max 元数据；真实 request ID 为 opaque key。
纯文本 AgentLoop 传入的空 multimodal kwargs 可正常处理。Runtime 启动 vLLM 时使用
priority scheduling，以支持 AgentLoop 发来的请求优先级。

框架新增通用配置项 `actor_rollout_ref.rollout.llm_server_manager_class`，默认 null，
与原 CheckpointEngine class hook 形式一致；配置 dataclass 和 YAML 同步补齐。
未来的完整训练 provider 可使用它。**当前 ElasticRolloutService 仅供独立生成入口使用**，
传入 trainer worker_group/resource_pool 或索取原生 replicas 会直接报错，防止误用原生权重路径。

## MVP-0 验收脚本（已验证流程参考）

以下脚本会改变指定测试 deployment 的副本数，最后缩到零并检查清理。请使用专用 deployment。
`PROMPT_IDS` 必须是当前模型 tokenizer 生成的有效 JSON token 列表。

```bash
# 先独立 Core，再分别在 AReaL / verl 环境运行。
python -m examples.cluster_smoke --framework core --endpoint "$ENDPOINT" --deployment mvp0 --prompt-ids "$PROMPT_IDS"
python -m examples.cluster_smoke --framework areal --endpoint "$ENDPOINT" --deployment mvp0 --prompt-ids "$PROMPT_IDS"
python -m examples.cluster_smoke --framework verl --endpoint "$ENDPOINT" --deployment mvp0 --prompt-ids "$PROMPT_IDS"

# 真正的固定 CPU workflow workers / AgentLoop 路径：
python -m examples.areal_workflow --endpoint "$ENDPOINT" --deployment mvp0 --prompt-ids "$PROMPT_IDS"
python -m examples.verl_agent_loop --endpoint "$ENDPOINT" --deployment mvp0 --model "$MODEL_PATH"
```

也可设置 `ELASTIC_E2E=1`、`ELASTIC_ROLLOUT_ENDPOINT`、`ELASTIC_ROLLOUT_DEPLOYMENT`、
`ELASTIC_E2E_PROMPT_IDS` 后执行 `pytest -m cluster -q`。默认明确 skip 三个集群测试。
脚本检查 token/logprob/version 对齐与 token budget，并输出生成 token 供比对；
没有把固定 temperature 当成跨硬件必然逐 token 一致的保证。

保留已有 MVP-0 运行日志，以及 NPU 进程清理与 Ray 资源恢复的记录，作为 MVP-1 回归基线。
MVP-1 的 CPU contract 测试额外验证 prepare/commit、快照追赶及动态版本传播；HCCL 留给 MVP-2。
