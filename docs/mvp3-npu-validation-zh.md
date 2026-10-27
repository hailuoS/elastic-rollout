# MVP-3 真机验证操作手册

验证目标：AReaL FSDP 和 verl FSDP2 使用同一公共 Direct HCCL 权重路径，
完成真实训练、连续更新和更新后的实际生成。本文对应 MVP-3 提交
`1aad77ffca0bd7849c436e57efb691b7a0049b5d`，不包含 MVP-4 自动扩缩。
当前状态是待真机验收，下面的结果栏不能提前填 PASS。

## 0. 先看操作顺序

本文命令用于 **Linux / Bash / 已有 Ascend Python 环境**，不是 Windows PowerShell。
首次按下面顺序做，不要同时跑两个框架：

1. 固定 MVP-3 代码并准备环境变量。
2. AReaL 单副本：启动 Core → 确认 v0 READY → 启动状态采集 → 训练 3 步 → 实际生成 → 关闭清理。
3. AReaL 双副本：使用新的 Core 进程，从 v0 重做上一步。
4. 关闭 AReaL Core 后，再跑 verl 原 Run A 配置的共享实现回归。
5. 最后做单独的活跃生成/故障/SIGTERM 测试。

**第 2 步失败就停在该步，不要继续跑后续用例或带病重试。** 把文末列出的材料带回分析。
第 2～4 步通过表示正常路径通过；所有验收项通过才标记 MVP-3 ACCEPTED。
本文不要求现场编写 sender/commit 故障注入代码，未执行的项目保留“待测”。

## 1. 固定测试代码

当前开发目录有 MVP-4 未提交改动，不要直接切 main，也不要把它们带入 MVP-3 测试。
在测试机已有仓库中，用已存在的 MVP-3 提交单独建立目录：

```bash
git cat-file -t 1aad77ffca0bd7849c436e57efb691b7a0049b5d
git worktree add --detach ../elastic-rollout-mvp3 1aad77ffca0bd7849c436e57efb691b7a0049b5d
cd ../elastic-rollout-mvp3
git rev-parse HEAD
python -m pip install --no-deps --no-build-isolation -e .
python -c 'import elastic_rollout; print(elastic_rollout.__file__)'
mkdir -p artifacts/mvp3
```

该提交之前未推送；如果测试机还没有它，需先把该提交传到测试机。
若内网不能访问远程仓库，可在已有该提交的外网仓库确认 `git rev-parse main`
等于本文 SHA 后，执行 `git bundle create mvp3.bundle main`，将 bundle 和**这份文档**一起传入内网。
内网新目录可用 `git clone -b main /path/to/mvp3.bundle elastic-rollout-mvp3`，
进入后再次核对 SHA。bundle 不包含另两个框架、不包含模型/数据/依赖，也不包含未提交的文档。
不要用另一个同名远程分支代替 SHA。确认 Python 导入路径属于新目录。
更换到 MVP-4 测试时需要重新安装对应目录的 editable package。
安装命令不会下载依赖，但要求当前环境已具备 setuptools 构建工具和运行依赖；
若提示缺包，先从已验证环境或内网 wheel 源补齐，不要现场升级 torch/Ray/vLLM。

### 1.1 一次填写路径，所有终端共用

在 MVP-3 仓库根目录执行以下命令。先把 `REPLACE_...` 全部改成实际值。
`VALIDATED_AREAL_CONFIG` 必须是已经在该环境跑通过的 **text/Math Agent** 配置，
不是原生 Megatron、视觉模型或未修改的公共下载配置。

```bash
mkdir -p local-config artifacts/mvp3
cat > local-config/mvp3.env <<'EOF'
export MVP3_ROOT='/REPLACE_ABSOLUTE_PATH/elastic-rollout-mvp3'
export MODEL_DIR='/REPLACE_SHARED_MODEL_DIR'
export VALIDATED_AREAL_CONFIG='/REPLACE_PATH/validated-areal.yaml'
export RAY_ADDRESS='REPLACE_GCS_IP:PORT'
export CORE_IP='REPLACE_CORE_IP'
export ROLLOUT_NODE_ID='REPLACE_RAY_NODE_ID'
export ROLLOUT_IP='REPLACE_ROLLOUT_IP'
export TRAINER_RANK0_IP='REPLACE_TRAINER_RANK0_IP'
export AREAL_PROXY_ADMIN_KEY='REPLACE_WITH_YOUR_NONDEFAULT_ADMIN_KEY'
export ELASTIC_ROLLOUT_DEPLOYMENT='areal-hccl'
export ELASTIC_ROLLOUT_ENDPOINT="http://${CORE_IP}:8080"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export AREAL_SPMD_MODE=0
EOF
chmod 600 local-config/mvp3.env
source local-config/mvp3.env
```

每个新终端先激活同一个 Python 环境，然后执行：

```bash
cd /实际路径/elastic-rollout-mvp3
source local-config/mvp3.env
cd "$MVP3_ROOT"
```

不要把 env 文件、admin key 或完整环境变量打包回传。
如果你已使用 `ELASTIC_ROLLOUT_TOKEN`，将同一个 token 配到各终端；不要只给 trainer 配而漏掉 Core。
离线模式要求模型、tokenizer、数据和 reward 所需资源已在本地准备好；出现下载请求应停下检查路径。

```bash
python - <<'PY'
import os
from pathlib import Path
for key in ('MVP3_ROOT', 'MODEL_DIR', 'VALIDATED_AREAL_CONFIG', 'RAY_ADDRESS',
            'CORE_IP', 'ROLLOUT_NODE_ID', 'ROLLOUT_IP', 'TRAINER_RANK0_IP',
            'AREAL_PROXY_ADMIN_KEY'):
    value = os.environ.get(key, '')
    assert value and 'REPLACE' not in value, f'请先填写 {key}'
assert Path(os.environ['MODEL_DIR'], 'config.json').is_file(), '模型目录不完整'
assert Path(os.environ['VALIDATED_AREAL_CONFIG']).is_file(), 'AReaL 配置不存在'
import elastic_rollout, areal, verl
print('elastic_rollout:', elastic_rollout.__file__)
print('areal:', areal.__file__)
print('verl:', verl.__file__)
print('路径预检通过；尚未启动模型或训练')
PY
python -m pip freeze > artifacts/mvp3/python-packages.txt
npu-smi info > artifacts/mvp3/npu-before.txt
```

如果没有已经跑通的 AReaL 配置，先停在这里：本文不能凭空保证训练设备布局、数据格式和 reward
配置正确。`AReaL/examples/math/gsm8k_grpo_npu.yaml` 只能作字段参考，其中默认模型/数据下载地址、
设备数、batch_size、scheduler 和较长 HCCL timeout 都需要按实际环境处理，不能原样当作本次配置。

## 2. 环境与资源

- 沿用 Run A 已跑通的 Ascend、PyTorch/torch_npu、Ray、vLLM/vLLM-Ascend 0.23 环境。
- 使用已有补丁的 verl、AReaL；记录三套仓库 SHA、依赖版本、Ray 节点和设备分配。
- 各节点训练/serving worker 使用同一套可用环境，能够导入本次 elastic-rollout；
  只在一个 driver 上安装 editable package，不会自动把本地目录分发到其他节点。
- 训练 NPU 与 rollout NPU 分开；先用两个 FSDP 训练 rank、一个 TP2 rollout 副本。
- 仅支持 dense text、BF16、预训练 bootstrap、AReaL v1/single-controller。
  不启用 Megatron、MoE、LoRA、vision、训练恢复或训练/推理共卡。
- bootstrap 模型是不可变共享目录，trainer 和 rollout 使用同一绝对路径。
- trainer rank0 的 rendezvous IP/端口必须能被所有 rollout worker 访问。
  HCCL 超时沿用已验证的集群配置，不能只设置 Python HTTP timeout。
- 测试前记录 `npu-smi info`、端口占用与本任务进程，作为资源清理基线。

## 3. AReaL：公共直传主验收

复制 `examples/runtime.areal-hccl.example.json` 为本地配置，替换模型、Ray NodeID、
节点 IP、端口。保留 `weight_sync=hccl`、`desired_replicas=1`，不启用弹性扩缩。
以下命令自动填入上一节的值，输出到 `local-config/runtime.areal-hccl.json`：

```bash
python - <<'PY'
import json, os
from pathlib import Path
d = json.loads(Path('examples/runtime.areal-hccl.example.json').read_text())
s = d['slots'][0]
s.update(node_id=os.environ['ROLLOUT_NODE_ID'], host=os.environ['ROLLOUT_IP'], port=8000)
d['deployments'][0]['replica']['model'] = str(Path(os.environ['MODEL_DIR']).resolve())
Path('local-config/runtime.areal-hccl.json').write_text(json.dumps(d, indent=2) + '\n')
print(json.dumps(d, indent=2))
PY
```

检查输出的 TP=2、rollout 端口 8000、Core 端口 8080；若端口已被占用，先确认占用者，
不要直接结束不属于本任务的进程。Ray 必须已启动；这份命令不会创建新 Ray 集群。

终端一启动 Core：

```bash
set -o pipefail
elastic-rollout serve --config local-config/runtime.areal-hccl.json \
  --host 0.0.0.0 --port 8080 --ray-address "$RAY_ADDRESS" \
  2>&1 | tee artifacts/mvp3/areal-core.log
```

终端二检查并启动训练：

```bash
source local-config/mvp3.env
elastic-rollout status --endpoint "$ELASTIC_ROLLOUT_ENDPOINT" \
  --deployment "$ELASTIC_ROLLOUT_DEPLOYMENT" > artifacts/mvp3/areal-before.json
```

开始训练前必须看到 `serving_ready_for_external_update=true`、版本 0、一个 READY 副本。
使用已经跑通的数据、tokenizer、reward 和 Math Agent 配置，仅修改：

| 配置 | 要求 |
| --- | --- |
| `actor.backend` | 原有可用的 `fsdp:...` 资源形状，两训练 rank |
| `actor.dtype` | `bfloat16` |
| `actor.weight_update_mode` | `xccl` |
| `actor.path` | 与 Core 相同的 bootstrap 绝对路径 |
| `actor.init_from_scratch` / `actor.use_lora` | `false` / `false` |
| `actor._version` / `rollout._version` | `v1` / `v1` |
| `recover.mode` | `disabled` |
| `total_train_steps` | 先运行 3 步 |
| `tokenizer_path`、训练/验证数据路径 | 本地已存在资源，不能依赖外网下载 |
| `scheduler`、`actor.scheduling_spec` | 沿用可用配置，明确与 rollout 的两张 NPU 分开 |
| `rollout` 调度 | 外部 CPU workflow；不要照搬原生 vLLM server 的 NPU 占卡布局 |

`rollout.backend` 的原生 DP 数不是 external vLLM 副本数；`--workflow-worker-count 2`
也是 CPU workflow worker 数，不能用它来把副本改成两个。
配置修改后另存为 `local-config/areal-mvp3.yaml`，保留原配置备份。

### 3.1 先打开状态采集终端，再启动训练

终端三执行以下命令，直到本轮结束后 Ctrl+C；每秒保存一个独立 JSON，不覆盖早期版本：

```bash
source local-config/mvp3.env
python -u - <<'PY' 2>&1 | tee artifacts/mvp3/status-watch.log
import json, os, time
from pathlib import Path
from elastic_rollout.client import ControlClient
c = ControlClient(os.environ['ELASTIC_ROLLOUT_ENDPOINT'],
                  os.environ['ELASTIC_ROLLOUT_DEPLOYMENT'],
                  os.environ.get('ELASTIC_ROLLOUT_TOKEN'), timeout=3)
out = Path('artifacts/mvp3/status'); out.mkdir(parents=True, exist_ok=True)
try:
    while True:
        try:
            s = c.status()
            name = str(time.time_ns()) + '.json'
            (out / name).write_text(json.dumps(s, indent=2) + '\n')
            print('version=', s.get('serving_policy_version'),
                  'ready=', s.get('ready_replicas'),
                  'update=', (s.get('policy_update') or {}).get('state'), flush=True)
        except Exception as e:
            print(type(e).__name__, str(e), flush=True)
        time.sleep(1)
except KeyboardInterrupt:
    pass
PY
```

采集到 `version=0 ready=1` 且 before.json 中
`serving_ready_for_external_update=true` 才在终端二启动训练：

```bash
set -o pipefail
python -m examples.areal_agent_train \
  --agent math --workflow-worker-count 2 --weight-sync hccl \
  --hccl-master-address "$TRAINER_RANK0_IP" --hccl-bucket-size-mb 1024 \
  --config local-config/areal-mvp3.yaml \
  2>&1 | tee artifacts/mvp3/areal-trainer.log
TRAIN_RC=${PIPESTATUS[0]}
printf '%s\n' "$TRAIN_RC" > artifacts/mvp3/areal-trainer.exit
echo "trainer exit=$TRAIN_RC（正常路径应为 0）"
```

bucket 必须容纳模型最大的完整 BF16 参数，不能机械地把 1024 MB 缩小。
训练结束后保存最终状态，**此时先不要关闭 Core**：

```bash
elastic-rollout status --endpoint "$ELASTIC_ROLLOUT_ENDPOINT" \
  --deployment "$ELASTIC_ROLLOUT_DEPLOYMENT" > artifacts/mvp3/areal-after.json
```

检查版本 0→1→2→3，`policy_update.state=COMMITTED`；记录 `transport` 的
`transport_id/init_count`、`transfer_session` 的 groups/bytes、vLLM PID。
同一次任务中 communicator 只初始化一次，vLLM PID 不因更新而重建。
一秒采集可能跳过很短的中间版本，不能据此判失败；用 Core 的每次 COMMITTED 日志补足 1/2/3。

### 3.2 更新后实际生成：直接执行

在终端二运行。它只从本地 tokenizer 编码固定问题，调用真实 vLLM，
把 token、logprob、版本和结束原因写入 JSON；无需手工构造 token IDs。

```bash
python - <<'PY'
import asyncio, json, os
from dataclasses import asdict
from pathlib import Path
from transformers import AutoTokenizer
from elastic_rollout.client import ControlClient
from elastic_rollout.generation import GenerationClient
c = ControlClient(os.environ['ELASTIC_ROLLOUT_ENDPOINT'],
                  os.environ['ELASTIC_ROLLOUT_DEPLOYMENT'],
                  os.environ.get('ELASTIC_ROLLOUT_TOKEN'))
s = c.status()
assert s['serving_ready_for_external_update'], 'Core 尚未恢复稳定就绪'
assert s['serving_policy_version'] == 3, '3 步训练预期提交到 v3；先查训练日志'
assert s['policy_update']['state'] == 'COMMITTED', '最后一笔更新未提交'
tok = AutoTokenizer.from_pretrained(os.environ['MODEL_DIR'], local_files_only=True)
ids = tok.encode('What is 1 + 1?', add_special_tokens=True)
g = GenerationClient(c, s['spec']['replica']['served_model_name'], request_timeout=60)
r = asyncio.run(g.generate('mvp3-final-probe', ids, {'max_tokens': 32, 'temperature': 0}))
assert r.token_ids, '未生成 token'
assert len(r.token_ids) == len(r.log_probs) == len(r.versions), 'token/logprob/version 长度不一致'
assert all(v == 3 for v in r.versions), '生成使用了非最终 committed 版本'
body = {'input_ids': ids, 'final_version': 3, 'completion': asdict(r),
        'text': tok.decode(r.token_ids)}
Path('artifacts/mvp3/areal-generation.json').write_text(json.dumps(body, indent=2) + '\n')
print('PASS: v3 实际生成，token 数=', len(r.token_ids))
PY
```

**实际生成验证不能只看 route 的版本标签。** 保留每次更新后生成的 token/logprob、
workflow trajectory/reward；条件允许时与同一版本导出模型独立加载后的固定输入
logprob 对照，记录精度容差与差异。生成文本没有变化本身不能证明更新失败，
文本变了也不能单独证明所有权重正确。
上面的 PASS 证明公共更新后实际生成链路可用，**不等同于逐参数或数值等价性通过**；
如需该等级验收，必须另行准备同版本权重对照，不能在结果表中混写。

### 3.3 本轮关闭

先确认 trainer 已退出、实际生成检查已完成，再在 Core 所在终端按一次 Ctrl+C。
等待 Core 自己完成清理，不要马上 `kill -9`。保存 `npu-smi info` 为
`artifacts/mvp3/npu-after.txt`，对照本任务 vLLM PID/端口及显存。
Core 退出后在状态采集终端 Ctrl+C，避免反复输出连接失败。
重跑或改双副本时必须启动全新 Core，重新看到 committed v0，不能直接重启 trainer 连接 v3。

## 4. verl：共享实现兼容回归

复制并修改 `examples/verl_v1_launch.json` 及其 Core 模板。沿用已通过 Run A 的
配置、输入 token IDs 和数据，保持 3 步训练、FSDP2、V1 separate_async、固定拓扑。

```bash
set -o pipefail
elastic-rollout verl run -c /path/to/verl_v1_launch.json \
  --smoke-report artifacts/mvp3/verl-smoke.json --smoke-prompt-ids 1 2 \
  2>&1 | tee artifacts/mvp3/verl-launch.log
VERL_RC=${PIPESTATUS[0]}
printf '%s\n' "$VERL_RC" > artifacts/mvp3/verl-launch.exit
```

`1 2` 应替换为 Run A 已验证的真实模型输入 IDs。
要求 launcher 退出码 0；报告中实际 post-update generation 成功，版本推进、
Core graceful cleanup 和 rollout ports closed 均成功。共享组件抽取后不应破坏 Run A。
AReaL Core 必须先关闭，否则 verl Launcher 会因为控制端口已占用而失败。
检查报告具体字段：`final_version=3`、`generated_tokens>0`、
`post_update_token_versions` 全为 3、`trainer_exit_code=0`、`core_exit_code=0`、
`core_graceful_cleanup=true`、`rollout_ports_closed=true`、`cleanup_success=true`。

## 5. 扩展验收与异常退出

1. AReaL 改成两个固定 external 副本，再跑 3 步。验证所有 receiver 完成 apply/verify
   后才 COMMIT；更新期间副本数量保持不变。
   配置同时增加第二个 slot、把 `desired_replicas` 改为 2；每个 slot 两张 NPU。
   同节点的两个 slot 必须使用不同端口（例如 8000、8001），不要复制相同 slot_id。
   不能只改 desired 而不增加 slot，也不能靠改 workflow-worker-count 增加 serving 副本。
   第一轮 artifacts 先另存，第二轮从新 Core 的 v0 开始。
2. 在生成活跃时更新，检查 abort 后的 continuation：没有旧 lease 残留，
   token/logprob/version 长度一致，Proxy/eval 跟随 committed 版本。
3. 在独立故障测试任务中终止一个明确属于本任务的 receiver，或用测试环境注入 sender/
   commit 故障。预期 fail closed，版本不错误推进、不放行混合权重，所有 rank 在配置
   的原生超时内退出。失败后重新启动 trainer 和 Core，不复用 poisoned communicator。
4. 正常结束以及 SIGTERM 分别验证资源释放。SIGTERM 发给已核对命令行的 Launcher PID：

   ```bash
   LAUNCHER_PID=REPLACE_WITH_VERIFIED_NUMERIC_PID
   ps -p "$LAUNCHER_PID" -o pid,ppid,pgid,args
   # 确认上面的命令行属于本次 Launcher，再执行：
   kill -TERM "$LAUNCHER_PID"
   ```

   记录退出耗时、状态码、Ray Actor、guardian/supervisor/vLLM PID、API 端口及
   `npu-smi info`。AReaL 先结束 trainer，再 SIGTERM Core。只清理本任务资源。
   `kill -9` 不属于 SIGTERM 验收，不能以 SIGKILL 后遗留进程判断 graceful cleanup。

MVP-3 固定 HCCL 成员，不要求更新后扩缩或故障后热补建；这些属于 MVP-4。
原 MVP-2B Run B 的退出验收仍单独记录，不因 MVP-3 训练通过而自动关闭。

## 6. 结果记录与通过条件

| 用例 | 结果 | 必须保存的证据 |
| --- | --- | --- |
| AReaL 1 replica / 2 ranks / 3 steps | 待测 | 训练/Core 日志、0→3 状态、实际生成 |
| AReaL 2 replicas / 3 steps | 待测 | 全部 receiver 完成、稳定 PID/transport |
| 活跃生成跨更新 | 待测 | abort/continuation、版本及 lease |
| sender/receiver/commit 故障 | 待测 | fail closed、超时退出、无错误提交 |
| verl 共享路径回归 | 待测 | launcher 日志、smoke JSON、退出码 |
| 正常关闭与 SIGTERM 清理 | 待测 | PID/端口/Ray/NPU 前后对照 |

以上通过后再标记 MVP-3 ACCEPTED。CPU 测试通过、HTTP 200、版本标签推进均不能代替真机证据。

## 7. 常见问题：停在哪里、带回什么

| 现象 | 先检查；不要做什么 |
| --- | --- |
| 找不到 `1aad77f` | 提交未传入内网；先传 bundle，不要退而测试旧远程 main |
| 安装时尝试访问外网 | 确认使用 `--no-deps --no-build-isolation`，当前环境已有构建工具 |
| 提示必须 committed v0 / fresh trainer | 连接了上轮 Core；完整关闭后启动新 Core，不在 v3 上硬重试 |
| Core 长时间没有 READY | 查最早的 supervisor/vLLM 启动错误、模型路径、NodeID、NPU 与端口；先别启动 trainer |
| Proxy admin key/tokenizer 报错 | 所有终端 source 同一 env；检查本地 tokenizer 和非默认 admin key |
| HCCL 初始化卡住 | 核对 **trainer rank0** IP、通信端口连通性、各进程 HCCL 配置；Core IP 不一定是 rank0 IP |
| 最大参数超过 bucket | 增大 bucket 或用已验证的小模型；不要用改参数名/跳过参数绕过 |
| status 版本变了但生成失败 | 本轮不通过，保留生成错误和 receiver finish/apply 日志 |
| receiver 故障后很久不退出 | 记录原生 HCCL/RPC timeout 和实际等待时长，可能是小时级；不要自行认定几秒就应退出 |
| SIGTERM 后有 PID/显存残留 | 保存目标命令、进程树和日志时间线；不要先用宽泛 pkill 清场而丢失证据 |

遇错至少保留：本文测试 SHA、另两个框架 SHA、脱敏后的实际配置、`python-packages.txt`、
训练退出码、完整 Core/trainer 日志、`status/`、before/after 状态、生成 JSON（若有）、
相关节点的 NPU 前后状态和本任务进程信息。用**第一次异常前后完整上下文**，不要只截最后一行。
若错误发生在 worker，补齐该节点的 Ray worker/supervisor/vLLM 日志；driver 最后的 RPC traceback
可能只包含转发错误，并非真正根因。

脱敏后可打包 `artifacts/mvp3`；配置里的 admin key/token、内网敏感路径按要求移除。
不要打包 `local-config/mvp3.env` 或完整 `env` 输出。没有执行的项目写“未执行”，不推断为通过。
