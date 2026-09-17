# MVP-1：行为、接口与配置参考

[返回首页](../README.md) · [先按验证手册执行](mvp1-validation.md)

本页解释代码行为和框架契约，操作顺序见验证手册。实现已通过 CPU/HTTP contract 测试，
Ascend 真机验收仍待完成。下方 `mvp0` 是 deployment 名称，与实现版本无关。

## Core 流程

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
更新要求至少一个 READY 副本，`desired_replicas=0` 时 begin 返回 HTTP 409；空副本集不能通过验证。
每个更新事务有绝对截止时间和累计候选失败预算，默认从 begin 起 1800 秒、最多 8 次失败。
失败次数按整个事务的所有候选副本累计，包含启动异常、启动/健康超时，以及提交前被检测到的进程故障；
同一个失败副本只计一次。第 8 次失败或到达截止时间即进入终态 FAILED，停止创建/重试候选。
截止时间覆盖 REQUESTED、QUIESCING、RESTARTING、VERIFYING 和 PREPARED（等待显式 commit）。
commit 也校验截止时间，不能抢在下一次 reconcile 前提交过期事务。
status/update API 暴露 `deadline_at`、`candidate_failure_count` 和失败原因。
FAILED 不推进已提交策略/版本、不恢复路由，也不接受扩缩容或新更新；需要显式关闭并重新启动部署。
时间预算在 reconcile/commit 操作边界检查，不能强行中断已在执行的底层进程/HTTP 调用。

**PREPARED/COMMIT 窗口：** 健康检查后、commit 前若进程死亡，描述符可能暂时仍为 READY。
本补丁保留此窗口：commit 不执行同步网络健康检查；下一次 reconcile 检测到故障后补建最新已提交策略。
若故障在 commit 前已被 reconcile 检测到，则 PREPARED 失效并重新验证。
此窗口可能短暂路由到失效端点，不会暴露混合策略版本；不保证进程在健康探测之后持续存活。
不可恢复故障保持路由关闭、不推进版本、不自动回滚。FAILED 后也锁定拓扑；可正常关闭 Controller
清理进程，再人工确定恢复模型。Controller 重启恢复不在本次范围。

```python
import os
from elastic_rollout.client import ControlClient

client = ControlClient("http://CONTROL_HOST:8080", "mvp0", os.environ.get("ELASTIC_ROLLOUT_TOKEN"))
update = client.begin_policy_update("/shared/immutable-policy-1")
client.wait_policy_update(update["update_id"])  # bounded polling, fails immediately on FAILED
client.commit_policy_update(update["update_id"], 1)
```

常规 health probe 正持有生命周期锁时，begin/commit 也可能短暂返回 409，可稍后重试。
`status.policy` 是当前版本；`status.policy_update` 是当前/最近更新；`accepting_routes` 表示准入开关。
`/replicas.loaded_model_path` 记录实际 launch 使用的模型路径。READY 数不代表路由开关已经打开。

## AReaL 训练入口与磁盘契约

AReaL 是本次首个完整训练接入目标，使用 [ElasticPPOTrainer](../elastic_rollout/integrations/areal/trainer.py)
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
训练时保留正常 off-policy/staleness 设置；MVP-0 示例中的 `10**12` 仅用于固定权重离线 workflow。

本分支真实契约是 `WeightUpdateMeta.path` 与 `update_weights_from_disk(meta) -> Future[None]`。
FSDP/Megatron 在调用 update 方法之后才导出，adapter 后台等待
`names.update_weights_from_disk(experiment, trial, 当前旧版本)` 完成标记，再 begin 并等到 PREPARED。
Future 完成时不 commit；训练循环随后 `set_version(new_version)` 才 commit。
`continue_generation()` 在原训练循环中早于 `set_version()`，因此它不会打开 Core 路由。
多个 workflow workers 加入同一路径/基础版本的更新。adapter 在 PREPARED 后绑定 `meta.version`；
后续 `set_version()` 必须与该目标版本相同，否则拒绝 commit 并保留 pending 状态。
成功提交后同时清空 pending update ID 和版本。对当前已提交版本的重复调用保持幂等。
更新失败保留异常和关闭的准入开关。

<a id="timeouts"></a>

## 时间和重试配置

Core 参数写在 `runtime.json` 的 `deployments[]` 对象中，旧配置省略新字段仍可读取：

```json
{
  "deployment_id": "mvp0",
  "replica": {"model": "/shared/bootstrap", "tensor_parallel_size": 2},
  "desired_replicas": 2,
  "policy_update_runtime_timeout_s": 1800,
  "candidate_start_max_failures": 8
}
```

AReaL 时间预算通过 `ElasticPPOTrainer(..., weight_sync_config=...)` 显式传入。
同一配置被序列化传给 workflow workers，并用于训练 callback，避免 worker 环境变量不一致：

```python
from elastic_rollout.integrations.areal.weight_sync import WeightSyncConfig

sync = WeightSyncConfig(
    snapshot_export_timeout_s=1800,
    policy_prepare_timeout_s=1800,
    weight_update_callback_timeout_s=3660,
)
trainer = ElasticPPOTrainer(config, train_dataset=train_dataset, weight_sync_config=sync)
```

| 参数 | 默认值 | 计时范围 |
| --- | --- | --- |
| `snapshot_export_timeout_s` | 1800 秒 | 从接到 disk update 到 AReaL 导出完成通知 |
| `policy_prepare_timeout_s` | 1800 秒 | 导出后等待部署稳定、begin/join 和 PREPARED |
| `weight_update_callback_timeout_s` | 3660 秒 | 外层权重更新 callback 请求 |
| `policy_update_runtime_timeout_s` | 1800 秒 | Core 从接受 begin 到 commit 的整个事务 |
| `candidate_start_max_failures` | 8 次 | 事务内候选副本累计失败上限 |

callback 必须至少覆盖 export + prepare + 60 秒，且不超过 7140 秒，为现有 AReaL 7200 秒
RPC/分布式等待上限留出余量；配置不符合时立即报错。更长的外层训练超时调整不在此补丁范围。
Core runtime 和客户端默认 prepare 使用同一个默认值常量；调大 Core 预算时也应调大 prepare/callback，
并为等待部署稳定和网络开销留出余量。客户端超时不会取消 Core 事务，Core 仍受自己的截止时间约束。
Core 的运行时间不包含快照导出。每个副本仍沿用 `replica.startup_timeout`（默认 600 秒），
如大模型单次加载超过该值，需显式增大该参数。
`set_version` commit 重试仍为 45 秒，受现有 AReaL 60 秒版本 RPC 约束；单次控制 HTTP 超时仍为 10 秒。

导出复用现有 HF save_pretrained / Megatron HF exporter，包含权重、config 和 tokenizer。
Megatron 持久导出补齐 base model 配置资源，供新 vLLM 进程独立加载；这仍需真实模型验证。
metadata hook 使用 run-unique 父目录，`with_version(N)` 生成类似：

```text
<原 AReaL save root>/elastic-rollout/<run-uuid>/weight_update_vN/
```

`clear_checkpoint_after_load=False`；RolloutController 尊重该标志。
当前和之前快照都会保留，不做 GC。**只要当前快照还可能用于扩容或故障补建，就不能删除或覆盖。**
共享文件系统必须在导出 barrier/完成通知后让推理节点看到完整不可变目录。

## MVP-1 控制 API

所有路径以 `/v1/deployments/{deployment_id}` 开头，使用与原 control API 相同的 Bearer token。

| 方法 | 后缀 | 用途 |
| --- | --- | --- |
| POST | `/policy-updates` | `{"snapshot_path":"/shared/policy-1"}`，开始准备 |
| GET | `/policy-updates/{update_id}` | 状态、截止时间、失败次数和错误 |
| POST | `/policy-updates/{update_id}/commit` | `{"policy_version":1}`，显式提交 |

HTTP 400 表示参数无效；409 表示生命周期/版本提交冲突；更新暂停新路由时 acquire 返回 503。
Core 不校验 Controller 本地路径是否存在，快照必须在实际 vLLM 节点可读。
