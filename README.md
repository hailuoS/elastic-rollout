# ElasticRollout MVP-0

根据相邻 `elastic_rollout_v1_2_design_rfc.md` 实现的独立 Python 公共组件。
公共运行时管理 external standalone vLLM 副本；框架只保留 workflow/AgentLoop 和生成语义。

**当前范围：固定模型、固定 policy、单节点正整数 TP / PP=1、副本数手动 `2 → 4 → 2`。**
已实现代码与 CPU/HTTP 模拟测试；真实 Ascend 集群和完整框架 workflow 验收仍需运行下述脚本。
按 RFC 第 32 节，尚未进入 MVP-1，不提供权重同步、HCCL、自动扩缩策略或完整 PPO 训练接入。

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
- Ray actor 按 slot 申请 `resources={"NPU": len(slot.device_ids)}` 并固定到指定节点，vLLM 用本地 `mp` 启动 TP workers。
- 一个 replica 对应单节点 slot，启动前要求设备数等于 TP。同节点 slot 的设备和 serving port 不得冲突；不同节点可复用设备编号和端口。
- `policy_version` 是启动时固定模型的标签，MVP-0 不执行权重内容校验或更新。

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

1. 准备模型可见的 Linux NPU 节点，Ray 节点公布真实 NPU 资源。原 TP=8 示例需四台八卡节点；单机 TP=2 示例只需一台八卡节点。
2. 从现有集群 `ray.nodes()` 取得 `NodeID` 和网络地址，填写
   [runtime.example.json](examples/runtime.example.json) 的所有 `REPLACE_*`，保存为 `runtime.json`。
3. 确认模型路径、Python/vLLM 版本、可见 NPU ID、端口均与实际环境一致；运行时与框架必须能直连各节点的 vLLM 端口。
4. Controller 只启动一个进程，默认监听 localhost；跨节点访问时显式指定绑定地址。

```bash
python -m elastic_rollout.cli serve --config runtime.json --host 0.0.0.0 --port 8080
```

可通过 `ELASTIC_ROLLOUT_TOKEN` 为 control API 设置 Bearer token；框架环境使用同一变量。
MVP 服务面向可信集群网络，不包含 TLS、HA 或持久化。vLLM 端点也需限制在集群网络内。

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

使用 [runtime.single-node-tp2.json](examples/runtime.single-node-tp2.json)：四个 slot 共用真实
NodeID/IP，设备分别为 `[0,1]`、`[2,3]`、`[4,5]`、`[6,7]`，端口为 8000–8003。
每个 actor 申请 2 NPU / 2 CPU，节点应有足够的 Ray CPU 资源；TP 在 deployment 内固定，PP=1。
填好占位符后保存为 `runtime.json`，按上面的 serve/scale 命令验证 `2 → 4 → 2`。
原四节点 TP=8 配置仍然支持，框架接入方式不变。

**需要实机验证的假设：Ray 按数量申请 2 NPU，未承诺恰好返回静态 slot 的物理设备编号。**
同节点 actor 的调度顺序、缩容后重新分配都可能使编号不同。
launcher 保留 `ASCEND_RT_VISIBLE_DEVICES` 严格检查；不一致时在创建 vLLM 进程前报错，
显示实际/预期 device IDs，绝不覆盖环境变量。此时该副本失败并记录错误，现有调和逻辑会重试；
不能把配置支持视为 Ray 静态编号匹配已经通过集群验收。模型本身也必须能在 TP=2 下加载运行。

### 控制 API

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
Ensure 不重置已有 desired state，不允许原地修改模型、policy 或 replica shape。

## AReaL 接入

目标源码：相邻 AReaL `ascend-v1.0.5`，本次读取 HEAD `2262610b`。
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

完整 PPOTrainer 仍会 `connect_engine/update_weights`，不属于 MVP-0；这些调用明确报错。

## verl 接入

相邻 verl 当前是 detached HEAD `483b8a00`（tag `v0.9.0`），不是一个本地命名 release 分支。
实现基于此实际源码，没有切换分支。未修改 `experimental/fully_async_policy`。

```python
from elastic_rollout.integrations.verl.rollout_service import ElasticRolloutService
from verl.experimental.agent_loop import AgentLoopManager

# config 是框架的完整配置，额外信息放进已有 rollout.custom 字段：
# actor_rollout_ref.rollout.custom.elastic = {
#   "endpoint": "http://CONTROL_HOST:8080",
#   "deployment_id": "mvp0", "fixed_policy": True,
# }
service = ElasticRolloutService.create(config=config)
manager = AgentLoopManager.create(config=config, llm_client=service.get_client())
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

## 验证

```bash
uv pip install -e '.[test]'
python -m pytest -q
ruff check .
ruff format --check .
```

CPU 测试覆盖状态机、并发路由、租约、drain、故障、资源隔离、HTTP API、直接 generation、
中断续传、Ray 资源申请参数，以及框架适配 contract。AReaL contract 测试直接执行相邻目标分支
的 Controller 和 payload builder 方法体，仅用 stub 隔离 torch/Ray 等重依赖。
这不是完整框架集成测试，也不能证明 Ascend 通信/算子兼容。

### 真实集群验收

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

MVP-0 的最终验收需保留这些真实运行日志，并确认 NPU 进程与 Ray 资源恢复。
两框架均完成后再进入 snapshot/HCCL 阶段。

## 当前限制

- 单 Controller、内存状态、无重启恢复；不要同时启动两个 Controller 管同一批 slot。
- 静态独占单节点 slot，不能与训练共享/切换同一设备，不能动态改变 TP/PP，不支持 multi-node replica。
- vLLM 必须提供 OpenAI completions 的 token-ID/logprobs 和运行/排队 metrics；自定义旧版服务
  可替换 `VLLMControlPlane`，MVP-0 不猜测其私有 pause/reload API。
- 停止失败/节点网络分区时优先保留资源占用，需要人工处理隔离 slot；不自动重用不确定资源。
- 不含完整 PPO、weight update、fully-async policy-update resume、Agent proxy、多模态或 LoRA。

实现参考：本地 RFC/目标源码；Ray [NPU 资源](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)、
[节点绑定](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy.html)
和 vLLM [completion 协议](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/completion/protocol/)。
