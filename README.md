# ElasticRollout MVP-1

根据相邻 `elastic_rollout_v1_2_design_rfc.md` 实现的独立 Python 公共组件。
公共运行时管理 external standalone vLLM 副本；框架只保留 workflow/AgentLoop 和生成语义。

**当前范围：共享文件系统持久快照、prepare/commit 权重切换、单节点正整数 TP / PP=1、副本数手动 `2 → 4 → 2`。**
已实现代码与 CPU/HTTP 模拟测试；真实 Ascend 集群和完整框架 workflow 验收仍需运行下述脚本。
MVP-1 基于相邻 `ElasticRollout MVP-1 Weight Sync Implementation Spec.md`；保留 MVP-0 路由和资源分配。
通过重启 stock vLLM 应用快照，HCCL 是 MVP-2，不实现自动扩缩策略。

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
   [runtime.example.json](examples/runtime.example.json) 的所有 `REPLACE_*`，保存为 `runtime.json`。
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

使用 [runtime.single-node-tp2.json](examples/runtime.single-node-tp2.json)：四个 slot 共用真实
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

本 patch 的 CPU 测试验证分配传递与历史保留；NOSET 配置和 Ascend 首次启动仍需在目标
Ray/CANN/vLLM 环境验证，不能将重试后 READY 视为首次初始化故障已经消失。

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
| POST | `/policy-updates` | `{"snapshot_path":"/shared/policy-1"}`，暂停新路由并准备更新 |
| GET | `/policy-updates/{update_id}` | 更新状态、目标副本、错误和状态历史 |
| POST | `/policy-updates/{update_id}/commit` | `{"policy_version":1}`，原子提交 PREPARED 更新 |

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

## AReaL 接入

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

完整磁盘更新训练入口见下方 `ElasticPPOTrainer`；裸 AReaL `PPOTrainer` 仍选择其原生推理引擎。

## MVP-1：共享磁盘快照与两阶段提交

### Core 流程

```text
REQUESTED -> QUIESCING -> RESTARTING -> VERIFYING -> PREPARED -> COMMITTED
                       unrecoverable error -> FAILED (routing remains paused)
```

Begin 要求 deployment 稳定且 READY 数等于 desired；扩缩容与更新互斥，冲突返回 HTTP 409。
HTTP handler 只记录控制状态；Reconciler 执行排空、stop/start、health 和 completion warmup。
更新立即阻止所有新租约（包括已有 sticky key），已有租约可以续租、释放和到期。
只有租约数为零且各 backend `/metrics` 确认空闲后才重启，TTL 到期不代替 backend idle。

Core 将候选路径作为不可变、不透明的模型目录，既不检查 Controller 本地存在性，也不复制或转换文件。
每个目标 replica 保留逻辑 ResourceLease、slot 和 endpoint，stop 后以 `--model <snapshot_path>` 启动；
Ray actor 重新申请资源时实际卡号可能变化，仍由 runtime context 决定。
这是进程重启，不是热加载；没有 AReaL vLLM extension 或 `/areal_update_weights` 依赖。

PREPARED 的候选副本仍不可路由，`serving_policy_version` 尚未变化。
显式 commit 要求更大的非负整数版本，在同一同步边界更新 committed policy、全部候选副本版本、
清除 sticky 并重新接收请求。同一 update/version 重复 commit 幂等。
提交后的扩容和 crash replacement 直接从最新 committed snapshot 启动。
候选启动故障可替换重试，仍使用候选快照；PREPARED 后检测到故障会回到验证流程。
不可恢复故障保持路由关闭、不推进版本、不自动回滚。FAILED 后也锁定拓扑；可正常关闭 Controller
清理进程，再人工确定恢复模型。Controller 重启恢复不在本次范围。

```python
from elastic_rollout.client import ControlClient

client = ControlClient("http://CONTROL_HOST:8080", "mvp0")
update = client.begin_policy_update("/shared/immutable-policy-1")
client.wait_policy_update(update["update_id"])  # bounded polling, fails immediately on FAILED
client.commit_policy_update(update["update_id"], 1)
```

常规 health probe 正持有生命周期锁时，begin/commit 也可能短暂返回 409，可稍后重试。
`status.policy` 是当前版本；`status.policy_update` 是当前/最近更新；`accepting_routes` 表示准入开关。
`/replicas.loaded_model_path` 记录实际 launch 使用的模型路径。READY 数不代表路由开关已经打开。

### AReaL 训练入口与磁盘契约

AReaL 是本次首个完整训练接入目标，使用 [ElasticPPOTrainer](elastic_rollout/integrations/areal/trainer.py)
复用原 PPOTrainer 训练循环，仅选择外部 CPU workflow workers 和持久磁盘导出。
需要同步更新本包与相邻 AReaL `feat/elastic-rollout` 的小范围补丁。

```bash
export ELASTIC_ROLLOUT_ENDPOINT=http://CONTROL_HOST:8080
export ELASTIC_ROLLOUT_DEPLOYMENT=mvp0
```

```python
from elastic_rollout.integrations.areal.trainer import ElasticPPOTrainer

# config 是完整 AReaL PPOConfig，按现有训练脚本准备 datasets/workflow。
config.actor.weight_update_mode = "disk"
config.cluster.fileroot = "/shared/areal"  # 所有训练 rank、workflow worker、vLLM 节点共同可见
config.recover.mode = "disabled"
trainer = ElasticPPOTrainer(config, train_dataset, valid_dataset, workflow_worker_count=2)
try:
    trainer.train(workflow=workflow)
finally:
    trainer.close()
```

运行条件：AReaL single-controller/v1 engines、完整模型、非 colocate、训练和 rollout 使用独占资源。
首次训练从可路由的 committed v0 开始，训练模型与 bootstrap 模型应为同一权重。
训练恢复/连接非零版本后继续训练未纳入 MVP-1，入口会明确拒绝，避免训练端版本 0 与推理端版本 N 的
导出完成通知不一致。独立推理 adapter 仍能读取非零 committed 版本。
评估 workers 使用 `ElasticEvaluationEngine`，只观察训练已经提交的版本，不重复提交。
训练时保留正常 off-policy/staleness 设置；上方 `10**12` 仅用于固定权重离线 workflow。

本分支真实契约是 `WeightUpdateMeta.path` 与 `update_weights_from_disk(meta) -> Future[None]`。
FSDP/Megatron 在调用 update 方法之后才导出，adapter 后台等待
`names.update_weights_from_disk(experiment, trial, 当前旧版本)` 完成标记，再 begin 并等到 PREPARED。
Future 完成时不 commit；训练循环随后 `set_version(new_version)` 才 commit。
`continue_generation()` 在原训练循环中早于 `set_version()`，因此它不会打开 Core 路由。
多个 workflow workers 加入同一路径/基础版本的更新；训练 callback 超时按 1200 秒配置，覆盖导出通知
120 秒与 prepare 900 秒预算。更新失败保留异常和关闭的准入开关。

导出复用现有 HF save_pretrained / Megatron HF exporter，包含权重、config 和 tokenizer。
Megatron 持久导出补齐 base model 配置资源，供新 vLLM 进程独立加载；这仍需真实模型验证。
metadata hook 使用 run-unique 父目录，`with_version(N)` 生成类似：

```text
<原 AReaL save root>/elastic-rollout/<run-uuid>/weight_update_vN/
```

`clear_checkpoint_after_load=False`；RolloutController 尊重该标志。
当前和之前快照都会保留，不做 GC。**只要当前快照还可能用于扩容或故障补建，就不能删除或覆盖。**
共享文件系统必须在导出 barrier/完成通知后让推理节点看到完整不可变目录。

### 人工验证

准备一份完整 HF 候选快照后，在专用 deployment 执行：

```bash
python -m examples.policy_update_smoke --endpoint "$ENDPOINT" --deployment mvp0 \
  --snapshot "$SHARED_SNAPSHOT" --version 1 --prompt-ids "$PROMPT_IDS"
```

脚本执行旧版生成、prepare、PREPARED、commit、新版生成、扩容到四副本并校验模型路径和版本；
结束时保留四副本。它不训练、不产生快照、不证明模型参数数值正确。
真正的 AReaL train/export/vLLM load/生成验证必须在 Ascend 环境另行完成。

## verl 接入

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
MVP-1 的 CPU contract 测试额外验证 prepare/commit、快照追赶及动态版本传播；HCCL 留给 MVP-2。

## 当前限制

- 单 Controller、内存状态、无重启恢复；不要同时启动两个 Controller 管同一批 slot。
- 静态单节点 capacity slot，Ray actor 持有独占 NPU 资源；不能与训练共享/切换同一设备，不能动态改变 TP/PP，不支持 multi-node replica。
- vLLM 必须提供 OpenAI completions 的 token-ID/logprobs 和运行/排队 metrics；自定义旧版服务
  可替换 `VLLMControlPlane`，MVP-0 不猜测其私有 pause/reload API。
- 停止失败/节点网络分区时优先保留资源占用，需要人工处理隔离 slot；不自动重用不确定资源。
- verl Trainer/CheckpointEngine 权重同步仍不支持；不含 HCCL、异步权重更新、自动回滚、Agent proxy、多模态或 LoRA。

实现参考：本地 RFC/目标源码；Ray [NPU 资源](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)、
[节点绑定](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy.html)
和 vLLM [completion 协议](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/completion/protocol/)。
