# MVP-1.5A：AReaL Agent Proxy 验证

## 文档审查与本轮范围

基线为 `b500ac6`，MVP-1 已由用户在 Ascend 真机验收。本轮只实现 MVP-1.5A，
保留现有 drain → restart → prepare → commit，不启用 partial rollout。
原规格要求 A 完成真机验收后再开始 B；本地 CPU 测试通过不能代替这一阶段门槛。

对照源码补充的细节：

- `MathAgent` 是单轮请求；多轮用 `MultiTurnMathAgent`，工具调用用 `MathToolAgent`。
- Proxy 需要有效 `rollout.tokenizer_path`，非回环监听需要非默认 `rollout.agent.admin_api_key`。
- Proxy 引擎是 policy observer：只验证 Runtime 已提交版本并更新本地版本，不发起 prepare/commit。
- 正常 rollout workers 的 commit RPC 完成后，Controller 才通知 Proxy 跟随版本。
- 多轮会话不保证固定物理副本。AReaL 保留会话和工具状态，逐次生成由 Elastic 获取路由。

### B 阶段进入前仍需核实

1. 目标 vLLM/Ascend 版本的暂停协议及 abort 的实际返回字段；不能把规格中的
   `stop_reason` 与当前 completion codec 的 `finish_reason` 不加验证地等同。
2. 暂停期间的崩溃补建与 resume 必须服从关闭路由的约束。
3. 同时验证普通 rollout 与 Proxy 的恢复门禁，保证 PREPARED 阶段没有 continuation。

这些是后续 B 的实现与测试边界，本轮没有改动其 serving profile、Runtime 状态或生成中断逻辑。

## 1. 准备

两个仓库都需要安装当前修改后的源码：

```bash
# 分别在对应仓库中执行，使用同一个 AReaL 运行环境。
pip install -e /AReaL --no-deps
pip install -e /elastic-rollout --no-deps
```

路径按实际 checkout 位置替换。沿用 MVP-1 已验证的依赖、Ray/NPU 设置和 `runtime.json`。
保留 `request_queue_size=256` 与独立 `elastic-control` 线程池修复。

先启动控制服务与推理副本，然后在 AReaL 环境配置：

```bash
export AREAL_SPMD_MODE=0
export ELASTIC_ROLLOUT_ENDPOINT=http://127.0.0.1:8080
export ELASTIC_ROLLOUT_DEPLOYMENT=mvp0
export TOKENIZER_PATH=/home/shl/models/Qwen3-4B
export AREAL_PROXY_ADMIN_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

同一运行的 workflow 与 Proxy 使用同一个 key。它与可选的 `ELASTIC_ROLLOUT_TOKEN` 是两种不同凭据。
端点必须能被 workflow/Proxy 进程访问；跨节点时使用实际控制服务地址。

## 2. 固定权重 Agent / Proxy / 扩缩容

从 **elastic-rollout 仓库**执行。此脚本用 LocalScheduler 启动两个 CPU workflow worker
及两个 colocated CPU Proxy；外部 vLLM 仍由已有的 Elastic Runtime / Ray 管理。
在专用测试部署运行，脚本会按 `2 → 4 → 2` 调整目标容量，并保留最后的两个推理副本。

```bash
python -m examples.areal_agent \
  --endpoint "$ELASTIC_ROLLOUT_ENDPOINT" \
  --deployment "$ELASTIC_ROLLOUT_DEPLOYMENT" \
  --tokenizer "$TOKENIZER_PATH" \
  --agent math --log-dir /home/shl/elastic-mvp15a/math
```

随后分别验证多轮和工具 Agent：

```bash
python -m examples.areal_agent \
  --endpoint "$ELASTIC_ROLLOUT_ENDPOINT" --deployment "$ELASTIC_ROLLOUT_DEPLOYMENT" \
  --tokenizer "$TOKENIZER_PATH" --agent multi-turn --log-dir /home/shl/elastic-mvp15a/multi

python -m examples.areal_agent \
  --endpoint "$ELASTIC_ROLLOUT_ENDPOINT" --deployment "$ELASTIC_ROLLOUT_DEPLOYMENT" \
  --tokenizer "$TOKENIZER_PATH" --agent tools --log-dir /home/shl/elastic-mvp15a/tools
```

若只检查现有四副本，追加 `--replicas 4`。日志和 `trajectories-*-r*.pt` 保留在 `--log-dir`，
成功运行还会产生 `report.json`。脚本仅销毁自己创建的 CPU workflow/Proxy，不关闭 Elastic 部署。

每个容量阶段检查：

- 生成两条有效 trajectory；生成 token 的 logprob 有限、版本均等于当前 committed policy。
- workflow / Proxy 各两个，ID 在扩缩容过程中保持不变，`server_infos` 保持空列表。
- Proxy 本地版本与 Runtime committed 版本一致；reward tensor 存在且为有限值。

脚本通过不代表工具调用和多轮分支一定发生：`MultiTurnMathAgent` 首次回答正确会提前结束，
工具使用也取决于模型输出。验收时需要另用实际任务确认至少两次模型交互、真实工具调用及会话状态，
检查 AReaL 导出的交互与 reward，不能仅凭 smoke 的退出码判定全部 Agent 能力通过。
工具模式沿用原生 `MathToolAgent` 和 Agents SDK，模型上下文上限由 Proxy 的 `engine_max_tokens` 约束。

## 3. 真实训练与版本跟随

使用已通过 MVP-1 的训练 YAML，保留 actor 的 disk/full-model/v1、staleness、设备分配等配置。
确保 `rollout.tokenizer_path` 有效，`rollout.agent.mode=inline`。Qwen3 可沿用：

```yaml
rollout:
  tokenizer_path: ${tokenizer_path}
  agent:
    mode: inline
    export_style: individual
    chat_template_kwargs:
      enable_thinking: false
```

这是对已有 YAML 的字段补充，不要覆盖整个 rollout 配置。Admin key 从上面的环境变量传入。

新的 Trainer 要求可路由的 committed v0，并与 actor 初始权重一致；已经推进版本的部署不能直接续跑
新的 Trainer。沿用 MVP-1 的全新部署流程。单机 8 NPU 可以继续 4 卡训练 + 4 个 TP1 推理副本。

```bash
set -o pipefail
python -u -m examples.areal_agent_train \
  --agent math --workflow-worker-count 2 \
  --config /AReaL/examples/math/gsm8k_grpo_npu.yaml \
  scheduler.type=ray total_train_steps=2 2>&1 | tee mvp15a-train.log
```

该入口沿用 AReaL 的 dataset 和 Agent 实现，仅选择 `ElasticPPOTrainer`；Ray 模式连接已有集群。
也可用 `--agent multi-turn` 或 `--agent tools` 运行对应训练，所有配置必须与已经验收的环境匹配。

需要保留以下证据：

```text
Agent trajectory @ v0
→ 训练及持久快照导出
→ 保守排空与候选副本重启
→ PREPARED（路由仍关闭）
→ 普通 rollout workers commit v1
→ Proxy observer 校验 committed v1 并跟随
→ workflow resume
→ 下一批 Agent token versions 为 v1
```

不得手动替 Proxy 调用第二次 commit。Observer 收到未提交版本应报错并保持原本地版本。

## 4. 补副本验收与完成门槛

按 MVP-0 已验证的方法，在专用环境中终止一个明确对应的 READY vLLM 进程，保留 Controller。
确认替代副本从最新 committed 快照启动。恢复 READY 后，执行 `examples.areal_agent --replicas 2`
（或当前目标数量）再次验证 Agent。不要终止整个 Ray 集群来模拟单副本故障。

本轮不增加透明的 generation 网络重试：故障恰好落在正在执行的请求上，该任务可能失败。
验收关注补建后的 Agent 可用性；跨更新的 partial trajectory 连续性属于 1.5B。

验收记录分别标注：固定权重、真实多轮/工具、真实训练 v0→v1、2→4→2、READY 故障补建。
全部通过后才进入 1.5B；当前本地结果不能宣称 MVP-1.5 或 1.5A 真机已完成。
