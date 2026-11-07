# ElasticRollout

独立管理 RL 训练所用的外部 vLLM 推理副本。AReaL、verl 保留原生训练、Agent 和参数导出，公共运行时负责副本生命周期、路由、策略版本事务和弹性扩缩。

- 支持 AReaL / verl FSDP2 和原生 Megatron Bridge 的 dense BF16 权重更新。
- GPU 使用 NCCL，NPU 使用 HCCL；保留 MindSpeed 路径。
- 支持手动扩缩、流水线压力驱动的自动扩缩、跨版本 partial rollout。
- 当前资源来自预先配置的静态 slot 池；平台资源申请和节点扩容尚需接入。

## 安装与启动

在对应框架及 vLLM 环境中安装本组件：

```bash
pip install -e '.[ray,async-http]'
```

填写 `examples/runtime.elastic.json` 的节点、地址、共享模型路径。默认配置是 NPU/HCCL；GPU 需设置 `replica.accelerator="gpu"`、`replica.weight_sync="nccl"`。

```bash
elastic-rollout serve --config examples/runtime.elastic.json --host 0.0.0.0 --ray-address <GCS地址>
elastic-rollout status --endpoint http://<控制器>:8080 --deployment elastic-rollout
elastic-rollout scale --endpoint http://<控制器>:8080 --deployment elastic-rollout 3 --wait
```

verl 的 V1 separate_async 训练可保留原生命令，在原训练参数后追加 `+elastic.enabled=true` 及副本上下限；不需要维护额外 JSON，详见[原生入口指南](docs/verl-native.md)。需要配套的 verl 入口与资源池补丁；训练模式、dtype 和资源预算沿用原生配置，不自动切换。已在单机 5090 上验证 FSDP2 真实训练、手动扩缩及信号清理；自动弹性收益、双物理机和 NPU 仍待验收。

显式管理配置的场景仍可使用原有统一 launcher：

```bash
elastic-rollout verl run -c examples/verl_launch.json --report run.json
```

AReaL 的训练入口是 `examples/areal_agent_train.py`，详见[框架接入](docs/integrations.md)。

## 自动扩缩

```bash
elastic-rollout autoscale --endpoint http://<控制器>:8080 --deployment elastic-rollout --config examples/autoscaling.json --once
```

独立 CLI 默认仅给出建议，加 `--activate` 执行扩缩；原生 verl 入口默认自动启动策略。默认采用 Relax 风格的推理负载策略（`policy_mode=service`），不再要求库存换代 trial。手动设置 desired 会切回 MANUAL 模式。规则、来源和边界见[自动扩缩说明](docs/autoscaling.md)。

## 使用边界

弹性模式保留初始基础副本组，新增副本使用独立持久通信组。新副本完成当前策略同步后才加入路由；缩容必须确认通信组退出后才释放资源。基础副本数是最小容量，无法通过弹性接口缩到更低。

固定拓扑模式仍支持共享快照重启加载和 Direct 更新；弹性模式要求 Direct HCCL/NCCL。控制状态保存在单个控制器进程内，控制器故障恢复和训练断点恢复尚未实现。

此前 GPU 基线已验证双机 Megatron TP2/PP2/DP2、40 次更新、手动/自动扩缩及同版本冷加载数值对齐。NPU/MindSpeed 路径保留，新增 Megatron 和弹性组合仍需 NPU 实机补验。该基线不代表本次重构已完成同规模 GPU 复验。

## 代码与验证

- `elastic_rollout/`：公共运行时、权重传输、框架适配器。
- `patches/megatron/`：原生框架兼容补丁。
- `examples/`：最小运行配置和 AReaL 训练入口。
- `tests/`：生命周期、事务、传输、生成、框架契约回归。

```bash
pip install -e '.[test]'
python -m pytest -q
python -m ruff check elastic_rollout tests examples
```

历史阶段报告和一次性验收程序不再随当前源码发布，可从 Git 历史恢复。

[架构与不变量](docs/architecture.md) · [配置与运行](docs/configuration.md) · [框架接入](docs/integrations.md)
