# ElasticRollout

独立的 RL 推理弹性组件：公共运行时管理外部 vLLM 副本，AReaL / verl 保留各自的 workflow、
AgentLoop 和生成语义。Core 不依赖这两个框架。

## 当前阶段

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| MVP-0 | 固定权重生成、租约路由、手动扩缩容、故障补建、NPU 资源清理 | 已在你的目标环境验证 |
| **MVP-1** | 持久 HF 快照、重启加载、prepare/commit、更新后扩容和补副本追赶新权重 | **当前需要做真机验证** |
| MVP-2 | HCCL/XCCL 直接权重传输 | 未实现 |

最近一轮代码回归结果为 **176 passed、3 个集群测试 skipped**，不能替代 NPU 权重更新验收。
MVP-1 使用共享文件系统和 vLLM 进程重启，不是热加载。AReaL 是首个完整训练接入目标；
verl 目前保留生成接入，Trainer/CheckpointEngine 权重同步尚不支持。

> 示例中的 `deployment_id: "mvp0"` 只是部署名称。使用该名称的现有部署也可以验证 MVP-1，
> 不需要为了阶段名称改动已经跑通的配置。

## 现在验证 MVP-1，从这里开始

**[打开 MVP-1 验证手册](docs/mvp1-validation.md)**，按两步进行：

1. **先验证 Core**：准备已有 HF 候选快照，执行旧版生成 → prepare → commit → 新版生成 → 扩容到四副本。
2. **再验证 AReaL 训练**：使用全新的 v0 部署和独立训练资源，检查真实训练导出 → prepare → set_version/commit → 下一轮 rollout。

只通过第一步，表示 Core 更新链路已跑通；完整 MVP-1 验收还需要第二步。

| 文档 | 何时阅读 |
| --- | --- |
| [MVP-1 验证手册](docs/mvp1-validation.md) | 当前入口：准备、命令、预期结果、故障定位 |
| [MVP-1 行为与配置参考](docs/mvp1-reference.md) | 查询状态机、HTTP API、超时和 AReaL 磁盘契约 |
| [MVP-0 已验证基线](docs/mvp0-baseline.md) | 重建基础环境、查询 Ray/NPU 配置或回归固定权重生成 |

## 代码入口

- [runtime.py](elastic_rollout/runtime.py)：副本生命周期、策略更新与故障处理。
- [ElasticPPOTrainer](elastic_rollout/integrations/areal/trainer.py)：AReaL 完整训练接入。
- [policy_update_smoke.py](examples/policy_update_smoke.py)：MVP-1 Core 真机验证工具。
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
- 不支持 verl Trainer 权重同步、HCCL、LoRA、多模态或 Agent proxy。
- 更新失败保持路由关闭，停止失败的资源保持隔离，需显式关闭和人工恢复。
