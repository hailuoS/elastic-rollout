# MVP-1 真机验证手册

[返回首页](../README.md) · [行为与配置参考](mvp1-reference.md) · [MVP-0 基线](mvp0-baseline.md)

**前提：MVP-0 已在当前目标环境跑通。** 本手册聚焦新增的权重更新能力。
先完成 A：Core 快照切换，再完成 B：AReaL 真实训练导出；两部分分别记录结果。

## 1. 验证前准备

### 代码和基础环境

- ElasticRollout 使用 `main`，至少包含修复提交 `e9c74a6`。
- AReaL 使用 `feat/elastic-rollout`，至少包含提交 `fa4bcfde`。
- 沿用已经验证过的 Ray / CANN / torch_npu / vLLM 环境和 `runtime.json`。
- Controller、Ray workers 和框架进程均安装同一版 ElasticRollout。
- 当前推理形态仍为单台 8-NPU、TP=2、PP=1，准备四个可用 capacity slots 验证 Core 扩容。

如需要重建 Ray/NPU 环境，查阅 [MVP-0 基线](mvp0-baseline.md)。已有正常运行的 Ray 集群无需仅因验证 MVP-1 而重建。
更新代码后应正常重启 Controller 和相关框架进程，使新配置与适配代码生效。

### 模型和快照

准备两个目录：bootstrap 模型和一份**已完整导出的候选 HF 模型快照**。候选目录必须包含：

- 完整权重：safetensors 或受当前 vLLM 支持的 HF 权重文件；分片权重需有对应索引。
- `config.json`，以及模型加载所需的 tokenizer/config 资源。
- 全部训练 rank、workflow workers 和 vLLM 节点可读的共享路径；单机时可用共同挂载目录。

使用相同架构、兼容 tokenizer、同一训练任务的不同版本权重。目录应在写入完成后再用于更新，
使用过程中不可覆盖或删除；当前和历史快照均不会自动清理。
复制同一份权重只能验证重启和版本标签流程，不能证明训练产生的新权重已生效。

### 两个验证阶段的资源差别

| 阶段 | 推理资源 | 训练资源 |
| --- | --- | --- |
| A：Core | 从 2 个 TP2 扩为 4 个 TP2，最多占满 8 NPU | 不启动训练 |
| B：AReaL | 至少 1 个 READY 推理副本，训练期间保持稳定拓扑 | 必须另有独占设备 |

**4×TP2 已占满单机 8 NPU，不能同时在这 8 张卡上启动训练。** 阶段 B 需要额外训练设备，
或先在你的环境验证训练/推理的资源隔离后重新划分容量；两个调度器不能重复使用同一组卡。

## 2. A：Core 快照切换和扩容验证

### A1. 启动两个 v0 副本

沿用已有 `runtime.json`，确认其中目标 deployment：

```json
{
  "deployment_id": "mvp0",
  "desired_replicas": 2,
  "policy_update_runtime_timeout_s": 1800,
  "candidate_start_max_failures": 8,
  "replica": {
    "model": "/home/shl/models/Qwen3-4B",
    "served_model_name": "policy",
    "policy_version": 0,
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "startup_timeout": 600,
    "vllm_args": ["--dtype", "bfloat16", "--max-model-len", "4096", "--enforce-eager"]
  }
}
```

上面只是 `runtime.json` 中 `deployments[]` 的一个元素；保留原来的 `slots`、真实 NodeID/IP 和端口配置。
模型路径按实际目录填写。旧配置不写两个新增字段也会使用以上默认值。

在专用终端启动 Controller（若已运行且满足下面的 v0 条件，可直接使用）：

```bash
python -m elastic_rollout.cli serve --config runtime.json --host 0.0.0.0 --port 8080
```

在另一个终端进入 ElasticRollout 仓库根目录，设置后续命令共用变量：

```bash
export ENDPOINT=http://127.0.0.1:8080       # 远程访问时填 Controller 实际 IP
export DEPLOYMENT=mvp0
export BOOTSTRAP_MODEL=/home/shl/models/Qwen3-4B
export SHARED_SNAPSHOT=/shared/policy-000001  # 改为已存在的候选快照目录
# 若 Controller 设置了 ELASTIC_ROLLOUT_TOKEN，本终端也设置相同值。

elastic-rollout status --endpoint "$ENDPOINT" --deployment "$DEPLOYMENT"
```

进入下一步前，应看到 `policy.version=0`、`desired_replicas=ready_replicas=2`、
`inflight=0`、`accepting_routes=true`。若 `policy_update.state=FAILED`，先按后面的恢复说明处理。

### A2. 生成合法 prompt token IDs

在已有模型 tokenizer 环境中执行；bootstrap 与候选快照应使用兼容 tokenizer：

```bash
export PROMPT_IDS="$(python - <<'PY'
import json
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ["BOOTSTRAP_MODEL"])
print(json.dumps(tokenizer.encode("Explain reinforcement learning briefly.")))
PY
)"
```

若已经有 MVP-0 验证时使用的合法 `PROMPT_IDS`，可直接复用。

### A3. 执行策略切换脚本

```bash
set -o pipefail
python -m examples.policy_update_smoke \
  --endpoint "$ENDPOINT" \
  --deployment "$DEPLOYMENT" \
  --snapshot "$SHARED_SNAPSHOT" \
  --version 1 \
  --prompt-ids "$PROMPT_IDS" 2>&1 | tee mvp1-core-smoke.log
```

该脚本自动执行：

```text
2 个 v0 副本生成
  → begin，关闭新路由
  → 排空、重启加载候选快照、health + warmup
  → PREPARED（仍暂停新路由）
  → commit v1
  → v1 生成
  → 扩容到 4 个副本
  → 检查 4 个副本的加载路径和版本，再生成
```

正常输出包含以下信息，token 数量取决于模型：

```text
generation policy=0, tokens=...
generation policy=1, tokens=...
generation policy=1, tokens=...
all four replicas serve committed policy 1 from /shared/policy-000001
```

脚本遇到 PREPARED 后会自动 commit，因此状态查询可能看不到短暂的 PREPARED；查看 Controller 的状态迁移日志。
脚本结束时保留 **4 个 v1 副本**，不会缩到零，也不会创建/导出候选快照。
重复执行时 `--version` 必须大于当前 `policy.version`；正式多轮权重验证应使用下一份独立快照。

### A4. 核对状态和副本

```bash
python - <<'PY'
import json
import os
from elastic_rollout.client import ControlClient

c = ControlClient(os.environ["ENDPOINT"], os.environ["DEPLOYMENT"],
                  os.environ.get("ELASTIC_ROLLOUT_TOKEN"))
print(json.dumps(c.status(), indent=2))
print(json.dumps(c.replicas(), indent=2))
PY
```

| 检查项 | 预期 |
| --- | --- |
| `policy.version` | `1` |
| `policy.model_path` | 候选快照路径 |
| `policy_update.state` | `COMMITTED` |
| `accepting_routes` | `true` |
| desired / ready / total | 均为 `4` |
| 每个 READY 副本的 `loaded_model_path` | 候选快照路径 |
| 每个 READY 副本的 `serving_policy_version` | `1` |
| `spec.replica.model` / `policy_version` | 仍是 bootstrap 配置和 `0`，这是正常行为 |

`/replicas` 会包含历史失败副本；检查当前 READY 副本，并确认历史资源已释放或明确隔离。
`last_error` 可能保留历史错误，应结合当前更新状态、失败次数和副本记录判断。

本脚本验证真实启动、生成和控制层版本/路径信息，**不比较模型参数数值，也不要求文本输出一定发生变化**。
如需验证新权重数值效果，使用同一候选快照建立独立推理对照，保留固定输入及生成/logprob 结果。

### A5. 补副本与排空回归

在专用测试环境，按 MVP-0 已验证的故障注入方式终止一个当前 vLLM 服务进程，保留 Controller。
等待补建后应恢复 4 个 READY 副本；新 replica ID 必须从候选快照启动，版本仍为 1，随后再次生成应成功。
注入故障前核对进程与 replica 的对应关系，避免终止 Controller 或整个 Ray 集群。

然后执行：

```bash
elastic-rollout scale --endpoint "$ENDPOINT" --deployment "$DEPLOYMENT" 2 --wait
```

确认缩掉的两个副本已释放 NPU，剩余两个仍为 v1。基础缩容机制属于 MVP-0，这里验证的是更新后的回归。

## 3. B：AReaL 真实训练导出验证

### B1. 为训练重新准备 v0

**不能把阶段 A 已提交到 v1 的部署直接交给新的 `ElasticPPOTrainer`。** 当前训练入口只接受
可路由的 committed v0，且不支持训练恢复。单纯 scale 到零再扩容不会重置 committed policy。

正常关闭阶段 A 的 Controller，确认 vLLM 进程和资源清理完成，再以 bootstrap v0 配置重新启动。
Controller 没有状态持久化；重启使用配置中的 bootstrap 模型，不会自动恢复上次 v1。
重新分配训练和推理的独占资源后，确认至少一个推理副本 READY、`policy.version=0`、`accepting_routes=true`。

### B2. 接入现有 AReaL 训练脚本

使用你已经准备好的 AReaL 训练配置、数据集、tokenizer 和 workflow，替换 Trainer 入口：

```bash
export ELASTIC_ROLLOUT_ENDPOINT="$ENDPOINT"
export ELASTIC_ROLLOUT_DEPLOYMENT="$DEPLOYMENT"
```

```python
from elastic_rollout.integrations.areal.trainer import ElasticPPOTrainer
from elastic_rollout.integrations.areal.weight_sync import WeightSyncConfig

# config / train_dataset / valid_dataset / workflow 沿用现有训练脚本。
config.actor.weight_update_mode = "disk"
config.cluster.fileroot = "/shared/areal"
config.recover.mode = "disabled"

trainer = ElasticPPOTrainer(
    config,
    train_dataset,
    valid_dataset,
    workflow_worker_count=2,
    weight_sync_config=WeightSyncConfig(
        snapshot_export_timeout_s=1800,
        policy_prepare_timeout_s=1800,
        weight_update_callback_timeout_s=3660,
    ),
)
try:
    trainer.train(workflow=workflow)
finally:
    trainer.close()
```

这是对现有训练脚本的接入片段，不是无需配置即可独立执行的训练程序。
要求 AReaL single-controller / v1 engines、disk 更新、完整模型、无 LoRA、非 colocate；
训练初始模型应与 bootstrap 权重一致。保持正常训练的 staleness 配置，不套用 MVP-0 固定权重示例中的放宽值。

先运行足够的小规模训练，确保发生一次更新并进入下一轮 rollout。按现有训练脚本正常启动，
不用手动调用 Core commit；AReaL 的 `rollout.set_version()` 会提交 pending 更新。

### B3. 验收一次真实更新

| 阶段 | 需要保留的证据 |
| --- | --- |
| 训练/导出 | FSDP 或 Megatron 更新成功；HF 导出完成，目录可从 vLLM 节点读取 |
| 持久快照 | 新的 `elastic-rollout/<run-uuid>/weight_update_vN/` 目录，权重/config/tokenizer 完整 |
| Prepare | Controller 日志显示候选路径重启、候选 READY、PREPARED |
| Commit | `policy.version=N`、`policy_update.state=COMMITTED`、路由恢复 |
| 下一轮 rollout | 请求成功，AReaL token 的 `output_versions` 来自新版本租约 |
| 后续可用性 | 已提交快照没有被训练端删除；推理仍可从该目录重启/补建 |

条件允许时再运行第二次更新，确认版本递增、路径不同、先前快照保留。
第一轮真实导出/切换成功后，才能将 Core 验证和 AReaL 训练验证共同记录为 MVP-1 真机验收结果。
若 actor 已推进版本但 rollout commit 最终失败，应停止本轮训练；没有自动回滚或续训恢复。

## 4. 遇到问题先看这里

| 现象 | 含义与处理 |
| --- | --- |
| begin 返回 409 | 检查副本是否稳定、desired 是否大于零、是否有活动/失败更新；短暂 lifecycle 锁冲突可重试 |
| 卡在 QUIESCING | 检查 inflight 和 vLLM running/waiting metrics；租约到期不代表后台生成结束 |
| 候选反复启动失败 | 查看每个副本日志、快照完整性及设备初始化；达到失败预算后会终止更新 |
| FAILED / deadline exceeded | 保留错误、截止时间和失败次数；修正原因后正常关闭并重新启动，不会自动恢复旧策略路由 |
| PREPARED 但生成 acquire 返回 503 | 提交前暂停新路由是预期行为；检查 AReaL 是否已进入 set_version，或手动验证是否执行 commit |
| `spec.replica.policy_version` 仍是 0 | 查看 `status.policy.version`；spec 是不可变 bootstrap 配置 |
| snapshot version mismatch | 检查 `meta.version` 与 set_version 是否相同，不可用错误版本标签提交 |
| 新 Trainer 拒绝非零版本 | 阶段 A 改变了 committed policy；按 B1 重新准备 v0 |
| callback 配置报错 | 必须覆盖 export + prepare + 60 秒，且不超过 7140 秒 |

默认 Core 更新预算是 1800 秒、累计最多 8 次候选失败；单副本 startup_timeout 仍为 600 秒。
大模型的导出、加载时间可能更长，应在测试前根据模型和存储配置预算。
完整参数及约束见 [时间和重试配置](mvp1-reference.md#timeouts)。

PREPARED 与 COMMIT 之间仍存在物理进程死亡的短窗口：下一次健康检测前可能路由到失效端点，
不构成混合策略服务。不要把这项已记录的限制误判为已实现进程存活的原子保证。

## 5. 这次需要保存的结果

- 两仓 commit ID、`runtime.json`（移除访问令牌）、训练配置及模型/快照路径。
- Controller、候选 vLLM、AReaL 训练日志，以及 `mvp1-core-smoke.log`。
- 更新前后 status 与 replicas 输出；记录失败重试、最终策略版本、加载路径和 NPU 释放情况。
- 分别标注“Core 已通过”“AReaL 真实训练已通过”或具体失败阶段。

本轮不包含 verl PPO/CheckpointEngine 权重同步；已有 verl 生成接入属于 MVP-0 基线。
