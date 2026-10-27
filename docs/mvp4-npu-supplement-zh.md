# MVP-4 NPU 补充验收

本文用于 GPU 验收后的 Ascend/HCCL 补验。GPU 结果见
[验收记录](mvp4-gpu-validation.md)；没有实际执行的 NPU 项目仍为待验收。
重点检查 HCCL 多通信组、TP 接收 rank 的同步销毁和物理资源释放。
训练后端仍限 dense BF16 FSDP2，Megatron 不在本次范围。

## 准备一次独立任务

使用 GPU 验收报告中记录的本分支代码，重新安装并核对导入路径：
生产代码验收基线为 `0fd4e22`，其后的验收资料提交不改变生产代码。

本分支尚未推送。内网拿不到分支时，在外网已有仓库执行
`git bundle create mvp4.bundle feat/mvp4-autoscaling`，把 bundle 和本文带入内网，
用 `git clone -b feat/mvp4-autoscaling /path/to/mvp4.bundle elastic-rollout-mvp4`
建立独立目录；已有仓库也可以从 bundle fetch。模型、数据、两个框架和依赖不在 bundle 内。

```bash
cd /path/to/elastic-rollout
git rev-parse HEAD
python -m pip install --no-deps --no-build-isolation -e .
python -c 'import elastic_rollout; print(elastic_rollout.__file__)'
mkdir -p local-config artifacts/mvp4-npu
```

沿用已通过 MVP-3 的 torch、Ray、vLLM-Ascend、模型、数据和奖励函数环境。
不要在验收时同时升级这些依赖。HCCL 端必须支持 worker extension 和原生
`hcclCommDestroy`；能力检查失败会阻止副本 READY，应先解决版本兼容。

单机 8 卡建议分配 **2 张训练卡 + 3 个 TP2 rollout slot**。
初始 2 个副本占 4 卡，另 2 卡供动态副本使用。基础副本不缩。
4 个 TP2 slot 再加 2 张训练卡需要 10 卡，不能放在单机 8 卡里。
以下示例只适用于已为本次任务预留的独立 Ray 资源池。
Ray 的 CPU 和逻辑 memory 也必须足够。若 AReaL 沿用每 worker 32 GiB 的默认请求，
DP2 actor 加两组各两个 workflow/eval worker 共请求约 192 GiB；可在物理内存足够的
独立 head 上预留至少 256 GiB。逻辑 memory 不足会一直等待 placement group，
即使 NPU 都空闲也无法开始。先检查资源，勿停止或重配共享 Ray 集群。

先复制你已经通过 MVP-3 的 LaunchSpec 到
`local-config/mvp4.launch.json`，保留原模型、数据、奖励和显存相关参数。
再填写以下变量；`ROLLOUT_NODE_ID` 是 Ray NodeID，不是 IP。

```bash
export CORE_IP='REPLACE_IP'
export RAY_ADDRESS='REPLACE_GCS_IP:PORT'
export ROLLOUT_NODE_ID='REPLACE_RAY_NODE_ID'
export ROLLOUT_IP='REPLACE_IP'
# 包含训练、权重更新的单 step 耗时估计，单位秒。
export STEP_SECONDS='120'
```

可用下面的只读查询核对 Ray NodeID、资源总量与剩余量：

```bash
python - <<'PY'
import os, ray
ray.init(address=os.environ['RAY_ADDRESS'])
print('total:', ray.cluster_resources())
print('available:', ray.available_resources())
for n in ray.nodes():
    if n['Alive']: print(n['NodeID'], n['NodeManagerAddress'], n['Resources'])
ray.shutdown()
PY
```

在仓库根目录执行以下配置生成命令。它保留已验证的其余 verl 参数。

```bash
python - <<'PY'
import json, os
from pathlib import Path
p = Path('local-config/mvp4.launch.json')
launch = json.loads(p.read_text())
launch['runtime'].update(ray_address=os.environ['RAY_ADDRESS'], accelerator='npu')
launch['controller'].update(advertise_host=os.environ['CORE_IP'])
launch['deployment'].update(id='mvp4-npu', core_template='mvp4.core.json')
model = Path(launch['model']['path']).resolve().as_posix()
core = json.loads(Path('examples/runtime.mvp4.example.json').read_text())
core['slots'] = [dict(slot_id=f'npu-{i}', node_id=os.environ['ROLLOUT_NODE_ID'],
    host=os.environ['ROLLOUT_IP'], num_devices=2, num_cpus=2, port=28100+i)
    for i in range(3)]
d = core['deployments'][0]
d.update(deployment_id='mvp4-npu', desired_replicas=2,
    drain_grace_period=max(60, 2*float(os.environ['STEP_SECONDS'])),
    retirement_timeout_s=max(300, 2*float(os.environ['STEP_SECONDS'])+120))
d['replica'].update(model=model, accelerator='npu', weight_sync='hccl',
    tensor_parallel_size=2)
overrides = dict(s.split('=', 1) for s in launch['verl']['overrides'])
overrides.update({
    'trainer.device': 'npu', 'trainer.nnodes': '1', 'trainer.n_gpus_per_node': '2',
    'trainer.total_training_steps': '80', 'trainer.total_epochs': '200',
    'trainer.test_freq': '-1',
    'trainer.save_freq': '-1', 'trainer.val_before_train': 'false',
    'trainer.v1.separate_async.parameter_sync_step': '1',
    'data.train_batch_size': '16', 'data.max_prompt_length': '1024',
    'data.max_response_length': '2048',
    'actor_rollout_ref.actor.strategy': 'fsdp2',
    'actor_rollout_ref.actor.ppo_mini_batch_size': '16',
    'actor_rollout_ref.rollout.nnodes': '1',
    'actor_rollout_ref.rollout.n_gpus_per_node': '6',
    'actor_rollout_ref.rollout.tensor_model_parallel_size': '2',
    'actor_rollout_ref.rollout.n': '2',
})
launch['verl']['overrides'] = [f'{k}={v}' for k,v in overrides.items()]
p.write_text(json.dumps(launch, indent=2))
Path('local-config/mvp4.core.json').write_text(json.dumps(core, indent=2))
from elastic_rollout.integrations.verl.launch_spec import LaunchSpec
LaunchSpec.from_file(p).core_config()
print('配置校验通过')
PY
```

80 步是留给人工操作和较慢冷启动的预算；验完目标后可以用下文的 SIGTERM 流程
提前结束，并如实记录中断退出码。无需为补验把 80 步重复跑满。

1024/2048 在这里是长度上限。若要测严格长度负载，需要原始数据分词后确为
1024，并启用 `rollout.ignore_eos=true`、核对实际 response length。
AReaL OpenAI Agent Proxy 的 EOS 参数行为与直接 RLVR workflow 不同，不能仅凭
`gconfig.ignore_eos` 宣称 Agent 输出了固定 2048 token。

## 用例一：两个框架各做一次 2→3→2

先运行 verl。记录实际 Launcher PID，信号测试只针对这个 PID。

```bash
elastic-rollout verl run -c local-config/mvp4.launch.json \
  > artifacts/mvp4-npu/verl.log 2>&1 &
export LAUNCHER_PID=$!
echo "$LAUNCHER_PID" > artifacts/mvp4-npu/launcher.pid
export ELASTIC_ROLLOUT_ENDPOINT="http://$CORE_IP:8080"
export ELASTIC_ROLLOUT_DEPLOYMENT='mvp4-npu'
```

若已有配置使用其他 Core 端口，请同步修改 endpoint。
下面的采集程序在另一个终端运行，继承同样的环境变量：

```bash
python -u - <<'PY' > artifacts/mvp4-npu/status.jsonl
import json, os, time
from elastic_rollout.client import ControlClient
c = ControlClient(os.environ['ELASTIC_ROLLOUT_ENDPOINT'],
                  os.environ['ELASTIC_ROLLOUT_DEPLOYMENT'],
                  os.environ.get('ELASTIC_ROLLOUT_TOKEN'))
while True:
    print(json.dumps(dict(at=time.time(), snapshot=c.scaling(),
                         replicas=c.replicas())), flush=True)
    time.sleep(2)
PY
```

等待 serving policy version 至少为 3，然后依次执行：

```bash
elastic-rollout scale --endpoint "$ELASTIC_ROLLOUT_ENDPOINT" \
  --deployment "$ELASTIC_ROLLOUT_DEPLOYMENT" 3 --wait
# 观察至少两次后续权重提交，再缩回 2。
elastic-rollout scale --endpoint "$ELASTIC_ROLLOUT_ENDPOINT" \
  --deployment "$ELASTIC_ROLLOUT_DEPLOYMENT" 2 --wait
```

每个阶段需满足：

| 阶段 | 必须看到的证据 |
| --- | --- |
| 2→3 | 新副本先 JOIN_PENDING；在正常训练更新 COMMIT 后 READY；随后能实际生成 |
| 连续更新 | 两个基础 replica ID 和 base transport ID 不变；base `init_count=1`；新副本 policy version 追上当前版本 |
| 3→2 | 先排空 lease/引擎请求，再 RETIRE_PENDING；同一 retirement ID 收齐 sender 和两个 TP receiver rank 的原生销毁确认 |
| 释放后 | 新副本 TERMINATED 且 `resource=null`；对应端口关闭、supervisor/vLLM 子树退出、所占 NPU 显存释放 |
| 缩后继续 | 至少两次新的正常权重 COMMIT；基础副本仍能生成，T0 没有重建 |

`--wait` 返回只作为控制面条件，仍需检查进程和 NPU：

```bash
npu-smi info
ps -eo user,pid,ppid,pgid,sid,lstart,args | grep -E 'elastic_rollout|vllm|verl|areal'
ss -ltnp | grep -E ':2810[012]\b'
ray status --address "$RAY_ADDRESS"
```

只核对本次任务的资源变化，保留开始前的基线，不要求整机显存归零。
不要按进程名批量 kill。缩容后允许下一次增长重用已确认释放的 slot，
建议再做一次 2→3→2，确认没有累计泄漏。

verl 完成并确认清理后，再使用同一 Core 模板测试 AReaL。
把 `AREAL_CONFIG` 设为已经通过 MVP-3 的本地配置，actor 保持 FSDP、BF16、
`weight_update_mode=xccl`，分配 2 张训练卡；endpoint/deployment 沿用上面的值。

```bash
export TRAINER_RANK0_IP='REPLACE_TRAINER_RANK0_IP'
export AREAL_CONFIG='/REPLACE_PATH/validated-areal.yaml'
elastic-rollout serve --config local-config/mvp4.core.json \
  --host 0.0.0.0 --port 8080 --ray-address "$RAY_ADDRESS" \
  > artifacts/mvp4-npu/areal-core.log 2>&1 &
export CORE_PID=$!
# 另一个终端确认初始两个 v0 副本 READY 后，再启动训练。
python -m examples.areal_agent_train \
  --weight-sync hccl --master-address "$TRAINER_RANK0_IP" \
  --config "$AREAL_CONFIG" > artifacts/mvp4-npu/areal-trainer.log 2>&1
```

复用上面的采集程序及 2→3→2 命令，改用独立的日志文件名，避免覆盖 verl 结果。
observer 会自动安装，不需改框架源码。训练退出后单独停止 Core 并等待：

```bash
kill -TERM "$CORE_PID"
wait "$CORE_PID"
```

AReaL 当前没有 verl 的统一 Launcher；下面的 Launcher 信号用例使用 verl 执行。

## 用例二：一次真实 AUTO 决策

使用仍在持续训练的全新任务，先输出影子建议：

```bash
elastic-rollout autoscale --endpoint "$ELASTIC_ROLLOUT_ENDPOINT" \
  --deployment "$ELASTIC_ROLLOUT_DEPLOYMENT" --once
```

检查 workload 有新鲜报告，`unit` 分别为 `prompt_group` / `workflow_group`，
消费单调增加，phase/两类 wait 能解释当前训练状态；未知等待不能当作缺卡。
确认后启动默认 AUTO，并保留日志：

```bash
elastic-rollout autoscale --endpoint "$ELASTIC_ROLLOUT_ENDPOINT" \
  --deployment "$ELASTIC_ROLLOUT_DEPLOYMENT" --activate \
  > artifacts/mvp4-npu/auto.log 2>&1
```

只需补一个与真实负载相符的端到端决策：持续缺数据时 OUT，或有余量时完成
TRIAL_START→COMMITTED。一直 HOLD 可能是正确结果，不能按预期强填 PASS。
缩容要求至少三个新消费周期、两次权重推进和旧库存周转；默认十周期窗口在慢 step
下可能超过 900s。若需要增大 `trial_timeout_s`，复制 autoscaling 示例 JSON 后显式
填写并把文件留在验收记录里，不要降低周期数或版本周转要求。

## 用例三：退出与清理

使用两个独立的新任务，避免故障状态相互影响：

1. **SIGTERM**：请求 2→3，在新副本 STARTING/JOIN_PENDING 或更新期间执行
   `kill -TERM "$LAUNCHER_PID"`。等待配置的 Trainer/Core 关闭期限，核对本次所有
   训练 Actor、supervisor、vLLM 子树、服务端口及 NPU 占用已释放。
2. **SIGKILL**：新副本已 READY 并经过至少一次后续更新后，执行
   `kill -KILL "$LAUNCHER_PID"`。Launcher 无法自行清理，但 Linux 父进程死亡通知
   应触发子进程 SIGTERM。按同样标准检查，不能只看 Launcher 已消失。

HCCL 原生调用卡死仍需要任务外部 watchdog。若达到配置的关闭上限仍残留，
记录 PID、PPID、PGID、完整命令行、端口、设备占用和最后日志，再处理明确属于本次
任务的进程。没有销毁/退出确认的资源不能被当成可复用 slot。

## 验收记录

| 项目 | 结果/证据路径 |
| --- | --- |
| 代码 SHA、三方版本、模型、batch、实际长度/step 耗时 | 待填写 |
| verl：TP2 2→3→2、T0 复用、销毁确认、缩后更新 | 待验收 |
| AReaL：TP2 2→3→2、T0 复用、销毁确认、缩后更新 | 待验收 |
| 一次与真实负载相符的 AUTO 决策 | 待验收 |
| Launcher SIGTERM / SIGKILL 后物理清理 | 待验收 |

只有这些 NPU 项目也通过后，才能把该 Ascend/vLLM-Ascend 版本标为可用。
GPU 测试不能代替原生 HCCL 销毁和 NPU 进程清理证明。
