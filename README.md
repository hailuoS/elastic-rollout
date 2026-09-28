# ElasticRollout

独立的 RL 推理弹性组件：公共运行时管理外部 vLLM 副本，AReaL / verl 保留各自的 workflow、
AgentLoop 和生成语义。Core 不依赖这两个框架。

## 当前阶段

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| MVP-0 | 固定权重生成、租约路由、手动扩缩容、故障补建、NPU 资源清理 | 已在你的目标环境验证 |
| MVP-1 | 持久 HF 快照、重启加载、prepare/commit、更新后扩容和补副本追赶新权重 | 已在你的目标环境验证 |
| **MVP-1.5A** | AReaL Agent Proxy、Proxy 跟随 committed policy、保守磁盘更新 | **当前需要做真机验证** |
| **MVP-1.5B** | planned abort、跨版本 partial continuation、组合暂停原因 | **用户已在 NPU 环境验证** |
| MVP-2A | verl 同步 Direct HCCL 权重传输 | 用户已在 NPU 环境验证 |
| MVP-2B | verl fully-async + Direct HCCL + partial rollout | 本地接缝与组合测试已实现，待 NPU 验收 |

本地 CPU/contract 回归测试不能替代 NPU Agent 和权重更新验收。
MVP-1 使用共享文件系统和 vLLM 进程重启，不是热加载。AReaL 是首个完整训练接入目标；
verl 新增 MVP-2A 同步 HCCL 接入，使用独立训练/推理卡；原固定权重生成入口保留。

## 验证 MVP-2A

**[MVP-2A 验证指南](docs/mvp2a-validation.md)**：依次运行原生 HCCL、Core 手工 sender、
verl FSDP2 单/双副本和同步多步训练。此入口不包含 MVP-2B async partial rollout。

**[MVP-2B 验证指南](docs/mvp2b-validation.md)**：沿用固定 HCCL 拓扑，验证 fully-async 训练、活动请求更新和跨策略版本续接。真机验收尚未执行。

> 示例中的 `deployment_id: "mvp0"` 只是部署名称。使用该名称的现有部署也可以验证 MVP-1，
> 不需要为了阶段名称改动已经跑通的配置。

## 现在验证 MVP-1.5B，从这里开始

**[打开 MVP-1.5B 实现与验收报告](docs/mvp15b-validation.md)**：先验证 stock vLLM abort/resume，
再验证跨版本 continuation，最后运行真实 AReaL RLVR / Agent 训练。
报告包含 Rev2 逐项核对、并发语义、自动化测试和待真机确认的边界。

Serving 基线为 **vLLM / vLLM-Ascend 0.23.x**。最新暂停排空语义、180 秒控制 RPC、
迟到请求与 AcquireRoute 竞态修复见 [1.5B Hardening 报告](docs/mvp15b-hardening.md)。

### MVP-1.5A Agent 基线回归

**[打开 MVP-1.5A 验证手册](docs/mvp15a-validation.md)**，按阶段进行：

1. **固定权重 Agent**：验证单轮、多轮、工具调用、trajectory 导出及 2→4→2 扩缩容。
2. **真实训练更新**：Agent v0 → 训练与持久快照 → drain/restart/commit → Proxy 跟随 v1。
3. **故障补建回归**：确认最新策略副本恢复后 Agent 仍可工作。

1.5A 手册保留原阶段基线；当前代码在训练更新时已经接入 1.5B planned abort，按新手册检查续接。

| 文档 | 何时阅读 |
| --- | --- |
| [MVP-1.5B 实现与验收报告](docs/mvp15b-validation.md) | 当前入口：planned abort、跨版本续接、Rev2 核对 |
| [MVP-1.5A 验证手册](docs/mvp15a-validation.md) | Agent/Proxy 基线回归 |
| [MVP-1 验证手册](docs/mvp1-validation.md) | 已验收的权重同步基线与回归 |
| [MVP-1 行为与配置参考](docs/mvp1-reference.md) | 查询状态机、HTTP API、超时和 AReaL 磁盘契约 |
| [MVP-0 已验证基线](docs/mvp0-baseline.md) | 重建基础环境、查询 Ray/NPU 配置或回归固定权重生成 |

## 代码入口

- [runtime.py](elastic_rollout/runtime.py)：副本生命周期、策略更新与故障处理。
- [ElasticPPOTrainer](elastic_rollout/integrations/areal/trainer.py)：AReaL 完整训练接入。
- [areal_agent.py](examples/areal_agent.py)：固定权重 Agent Proxy 真机验证。
- [areal_agent_train.py](examples/areal_agent_train.py)：原生 Math Agent 的 Elastic 训练入口。
- [policy_update_smoke.py](examples/policy_update_smoke.py)：MVP-1 Core 真机验证工具。
- [partial_rollout_smoke.py](examples/partial_rollout_smoke.py)：中断、恢复和跨版本续接验收。
- [单机 TP2 配置](examples/runtime.single-node-tp2.json)：单台 8-NPU、最多四个 TP2 副本。

## 本地开发检查

```bash
uv pip install -e '.[test]'
python -m pytest -q
ruff check .
ruff format --check .
```

测试覆盖 Core、HTTP、资源管理及 AReaL/verl contract；框架 contract 测试使用相邻目标分支源码，
以 stub 隔离 torch/Ray 等依赖。真实集群测试默认跳过。

## 当前边界

- 单 Controller、内存状态；无 Controller 重启恢复、自动回滚或快照 GC。
- 单个推理副本只跨一个节点，TP 固定、PP=1；训练与推理必须使用独占设备。
- 副本数手动调整，不含自动扩缩策略。
- 已接入 verl Direct HCCL；LoRA、多模态与 MVP-2B fully-async 跨版本续接尚未完成相应真机验收。
- 更新失败保持路由关闭，停止失败的资源保持隔离，需显式关闭和人工恢复。
