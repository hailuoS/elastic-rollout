# 框架接入

在框架环境安装 `pip install -e '.[async-http]'`。AReaL 和 verl 默认使用共享异步 HTTP 传输，不再让长生成请求占用默认线程池。连接池在同一事件循环的并发生成间复用，最后一个逻辑请求结束后关闭；取消请求仍等待在途请求结束并释放租约。推理请求直连控制器返回的副本 endpoint，不继承环境 HTTP 代理、不跟随重定向，也不自动重试；单请求超时覆盖整个 HTTP 调用。

## verl

填写 `examples/verl_launch.json`。`deployment.core_template` 相对于 LaunchSpec 文件解析；模型路径、数据、控制器地址、Ray GCS 地址及原生 Hydra overrides 需与实际部署匹配。

```bash
elastic-rollout verl run -c examples/verl_launch.json --report run.json
```

默认 NPU/FSDP2。GPU 设置 `runtime.accelerator="gpu"`，并将原生 override `trainer.device` 改为 `cuda`。启动器解析一个确定的 Ray 集群地址，执行配置预检，等待 bootstrap 副本 ready，再启动原生 `verl.trainer.main_ppo`。

Megatron 设置 `verl.strategy="megatron"`，移除示例中显式的 `actor_rollout_ref.actor.strategy=fsdp2`。其余 TP/PP、训练 batch、优化器等参数使用原生框架配置。适配器选择原生 Megatron Bridge 导出，支持 dense BF16；不支持 EP/CP、LoRA、FP8 组合。

## AReaL

### 原始脚本入口（NPU，当前验收候选）

工作区中的 AReaL 已接入可选 `elastic` 配置。保持原始 Python 训练脚本、
workflow 和奖励代码，在原生配置后追加参数：

```bash
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN_PATH=/path/to/private/ray/auth_token
export RAY_ADDRESS=<专用Ray集群GCS地址>
python examples/math/gsm8k_rl.py --config <原生训练配置.yaml> \
  scheduler.type=ray actor.weight_update_mode=xccl \
  +elastic.enabled=true +elastic.min_replicas=1 +elastic.max_replicas=4 \
  +elastic.ray_address="${RAY_ADDRESS}"
```

原生 YAML 没有 `elastic` 节点时使用 Hydra 的 `+` 语法；已有该字段时直接
覆盖。关闭功能只需将 `enabled` 改为 `false`，仍使用原生 `PPOTrainer`。
此入口需要安装本仓与包含对应接入修改的 AReaL，不能用于未修改的上游版本。

启动前准备一个空闲、启用 token 认证的专用 Ray NPU 集群；其节点数及每节点
`NPU` 资源必须与原生 `cluster` 配置一致。actor 与 rollout 使用独立的设备预算，
例如 8 卡节点分配 4 卡训练和 4 卡推理预算；每个 actor worker 的 scheduling spec
声明 1 张卡。FSDP 与 Megatron 均从原生 backend/并行配置推导布局，无需另写 Core JSON。
推理 TP 组必须在单节点内放下，最大副本数不能超过推理预算。

当前入口要求 v1、dense BF16、预训练本地/共享模型、`recover.mode=disabled`、
vLLM rollout。Megatron 使用 Megatron Bridge 和 Bridge 权重更新；EP/CP 为 1，
不启用 LoRA/FP8。启用 KL reference 时，reference 必须与 actor 采用相同后端和
并行布局，并通过非 fork 的 colocation 复用 actor workers。critic/teacher 尚未接入
该入口的资源规划，会在启动前拒绝。

父进程先启动 Core bootstrap 副本，再重新执行同一个 Python 脚本或模块；训练子进程
绑定同一 Ray 集群，由 `ElasticPPOTrainer` 执行权重更新和 rollout。默认开启自动策略；
独立手动功能验收可追加 `+elastic.autoscale=false`。控制器默认端口 28080、
推理端口起点 28100，可通过对应 elastic 参数覆盖。退出报告写入原生 stats logger
目录下的 `elastic/native-entry.json` 和 `elastic/cleanup.json`，不记录认证密钥值。

该入口的配置、原始命令重执行和 CPU 契约测试已通过；AReaL 两后端的完整 NPU
训练及扩缩验收仍在进行，不能将这些测试视为实机扩缩已通过。

### 显式外部 Core 入口

先启动 Core，设置训练进程环境：

```bash
export ELASTIC_ROLLOUT_ENDPOINT=http://<控制器>:8080
export ELASTIC_ROLLOUT_DEPLOYMENT=elastic-rollout
export RAY_ADDRESS=<GCS地址>
python -m examples.areal_agent_train --weight-sync nccl --master-address <训练rank0地址> <原生AReaL配置参数>
```

示例从源码目录运行，原生配置参数按 AReaL 的 `load_expr_config` 约定传入。NPU 改用 `--weight-sync hccl`；快照模式用 `--weight-sync snapshot`。Direct 模式需配置原生 `actor.weight_update_mode=xccl`，初始模型与 Core bootstrap v0 一致，并禁用训练恢复。

`ElasticPPOTrainer` 支持原生 FSDP2 和 Megatron actor。Megatron 要求 `megatron-bridge`、启用 Bridge 权重更新并使用 dense BF16，EP/CP 均为 1，关闭 FP8/LoRA。Agent workflow、奖励和优化仍由 AReaL 原生实现。

`--agent` 可选 `math`、`multi-turn`、`tools`。部署 Agent Proxy 时配置 `AREAL_PROXY_ADMIN_KEY`；依赖按 AReaL 自身锁文件安装。

## Megatron 原生兼容补丁

补丁位于 `patches/megatron/`，应用前检查目标源码版本：

| 源码 | 基线提交 |
| --- | --- |
| verl | `961dcbc2620dbdac4ae24fbaaff9fa1fde29f65a` |
| AReaL | `1029c93bdc0b3c91b0f72251abaa39b1e716fc31` |

| 补丁 | 作用 |
| --- | --- |
| `verl-export-offload.patch` | 原生导出期间的参数驻留及退出处理 |
| `areal-cuda.patch` | CUDA 与原生 NPU 路径选择 |
| `areal-bridge-gradient-fusion.patch` | Bridge 梯度融合配置兼容 |
| `areal-tms-runtime.patch` | TMS 运行时选择与加载 |
| `areal-tms-state.patch` | TMS 状态及内存迁移兼容 |

在对应原生仓先执行 `git apply --check <补丁>`，再执行 `git apply <补丁>`。不要对不同版本盲目应用。补丁保留原有 MindSpeed/NPU 分支，当前源码测试覆盖相应结构和行为契约。

此前 Megatron GPU 验证环境使用 vLLM 0.23.0、Torch 2.11.0+cu130、Megatron Core 0.17.0、Bridge 0.4.0、mbridge `310e8fb`、Transformer Engine 2.20.2、TMS 0.0.10。该组合是已验证基线，并非对任意依赖版本的保证。

训练与推理 Python 环境可分离，但通信库必须兼容。必要时设置 `VLLM_NCCL_SO_PATH` 为推理环境使用的 NCCL 动态库路径，并传递到训练 worker。

## 测试范围

本仓保留核心行为测试、HTTP 事务测试、框架导出契约和原生补丁检查。框架契约测试要求工作区同级存在对应固定版本的 `AReaL/` 和 `verl/` 源码。一次性集群验收驱动与历史阶段报告已退出当前源码；旧结果可从 Git 历史查看。

本地 stub/CPU 测试不能证明跨机器 collective 或 NPU 实机行为。重构后的运行验证结果应与此前端到端训练基线分别记录。
