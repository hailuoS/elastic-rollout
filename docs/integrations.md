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
