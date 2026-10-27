# MVP-3: AReaL and verl share Direct HCCL

中文真机操作步骤、固定提交与验收记录见 [MVP-3 真机验证操作手册](mvp3-npu-validation-zh.md)。

Status: implementation and CPU contracts; **real NPU acceptance is pending**.
MVP-2B Run B remains a separate open acceptance gate. This implementation does
not claim to fix launcher SIGKILL ownership or close MVP-2B.

## Architecture and supported profile

Both framework adapters now use the same framework-independent components:

- `elastic_rollout.weight_sync.ElasticWeightSync`: begin, pause/abort and drain,
  frozen target preparation, producer initialization, transfer, finish/verify,
  commit and resume. Only the existing explicit lifecycle-lock rejection is retried.
- `elastic_rollout.hccl_sender.HCCLSender`: persistent sender identity and
  communicator, bounded named-tensor buckets, HTTP metadata/HCCL data overlap,
  version binding, transport reuse and failure propagation.
- Existing Core `TransferCoordinator`, policy transaction and generation paths
  remain authoritative. The public Core does not import either framework.

AReaL uses a custom actor controller and FSDP actor inside this package. No AReaL
source changes, config-schema changes, scheduler changes or new dependencies are
required. Keep using the modified adjacent AReaL checkout that supports external
CPU workflows and Agent Proxy. The original snapshot path remains the default.

Supported: Ascend NPU, AReaL v1 engine/controller APIs, single controller, FSDP2,
dense text models, BF16 compute weights, separate trainer/rollout devices,
fixed serving membership, pretrained bootstrap (`init_from_scratch=false`), no
trainer recovery. Megatron, Archon, MoE, vision,
LoRA, colocation, GPU/NCCL, dynamic joins and automatic scaling are excluded.

## AReaL lifecycle

```text
native PPOTrainer training/update safe point
  -> ElasticHCCLActorController.update_weights(meta)
  -> shared ElasticWeightSync transaction
  -> RPC to every ElasticHCCLFSDPActor
  -> native canonical names, full_tensor and compute-dtype conversion
  -> shared HCCLSender (only trainer rank0 sends to rollout)
  -> Core PREPARED -> COMMIT -> resume physical generation
  -> native actor/rollout/eval set_version callbacks
  -> workflow and Proxy observe the committed serving version
```

The adapter preserves the native rollout callback/coordinator but bypasses the
native AReaL XCCL receiver discovery and group initialization. `weight_update_mode`
is still the native `xccl` value; external HCCL is selected by passing an
`HCCLWeightSyncConfig`, not by adding an AReaL config field.

All ranks execute native FSDP full-tensor collectives and BF16 conversion under
`torch.no_grad()` and the native offload-aware context. Initial parameter names
and shapes are compared before FSDP export. Each parameter's post-gather cast
status is exchanged before the next gather, so a rank-local cast/OOM cannot
leave peers starting the next collective. Before each bucket send,
the CPU group compares errors and the tensor metadata manifest. After each send,
all ranks exchange status before materializing the next bucket. A rank0 HCCL
send failure therefore does not leave its peers advancing into the next FSDP
collective. Collectives that themselves hang still require finite framework/HCCL
timeouts and clean process restart; arbitrary collective repair is not supported.

The sender holds one bucket plus one lookahead tensor. The bucket size must fit
the model's largest full BF16 tensor; tensors are not split across buckets. Every
rank temporarily materializes these bounded buckets. No complete model snapshot
is serialized through the controller or written to disk for steady-state updates.

## Configure and run

1. Install current ElasticRollout and the adjacent patched AReaL in the same NPU
   Python environment. Use the same vLLM/vLLM-Ascend 0.23 serving baseline and
   Direct HCCL network/runtime prerequisites documented for MVP-2A.
2. Copy `examples/runtime.areal-hccl.example.json` and replace the model directory,
   rollout node ID/IP and port. The absolute immutable bootstrap model must be
   visible at the same canonical path to trainer and rollout workers. Start Core:

   ```bash
   elastic-rollout serve --config /path/to/runtime.areal-hccl.json \
     --host 0.0.0.0 --port 8080 --ray-address <existing-gcs-address>
   ```

3. Wait for `serving_ready_for_external_update=true`. Keep trainer devices separate
   from Core rollout slots. The standard AReaL Ray/local scheduler still owns
   trainer placement. Set the control address and deployment:

   ```bash
   export ELASTIC_ROLLOUT_ENDPOINT=http://<controller-ip>:8080
   export ELASTIC_ROLLOUT_DEPLOYMENT=areal-hccl
   ```

   Set `ELASTIC_ROLLOUT_TOKEN` if Core uses authentication. The adapter explicitly
   passes this token and sender settings to training workers; it does not rely on
   arbitrary inherited worker environment variables.

4. Use an already working AReaL native Math Agent training config/data/reward setup.
   Change `actor.weight_update_mode` to `xccl`, use an `fsdp:` actor backend and
   `actor.dtype=bfloat16`, set `actor.path` to the Core bootstrap directory, and
   disable recovery. Keep actor/rollout `_version=v1`. Start with two training
   ranks, one external rollout replica and 2–3 training steps.

   The existing example accepts the new HCCL switches:

   ```bash
   python -m examples.areal_agent_train \
     --agent math --weight-sync hccl \
     --hccl-master-address <trainer-rank0-reachable-ip> \
     --hccl-bucket-size-mb 1024 \
     --config /path/to/your-validated-areal-config.yaml
   ```

   Retain the config invocation/overrides required by your AReaL environment.
   `--hccl-master-address` names the trainer **rank0 node**, which may differ from
   the controller host. Receiver workers must reach its rendezvous port. Set
   finite HCCL connect/execute timeouts on trainer and rollout processes using
   your working NPU environment configuration. Agent Proxy also requires the
   existing non-default admin key and tokenizer configuration.

For an existing Python workflow entrypoint, use the same trainer:

```python
from elastic_rollout.integrations.areal.trainer import ElasticPPOTrainer
from elastic_rollout.integrations.areal.weight_sync import HCCLWeightSyncConfig

with ElasticPPOTrainer(
    config,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    weight_sync_config=HCCLWeightSyncConfig(master_address=trainer_rank0_ip),
) as trainer:
    trainer.train(workflow=workflow, workflow_kwargs=workflow_kwargs)
```

This is an AReaL training entry, not a new unified AReaL launcher. Core must already
be running. The existing `elastic-rollout verl run` path keeps its public API and
uses the shared components behind its CheckpointEngine facade.

## CPU regression and NPU acceptance

Local result (2026-10-23, Windows/Python 3.12): full regression **452 passed,
13 skipped**; targeted AReaL/verl/integration regression **63 passed**. Ruff lint,
format checks for the modified Python files and `git diff --check` passed.
Hardware/cluster and Linux process tests remain skipped in this workspace.

```bash
python -m pytest tests/test_areal_hccl.py tests/test_verl_hccl.py tests/test_integrations.py -q
python -m pytest -q
```

CPU contracts exercise real Core/HTTP receiver controls with two simulated
training ranks, three committed versions with one transport initialization,
unchanged receiver identities, BF16 export from FP32 master tensors, rank0 send
failure with peer exit, rank-local cast failure, commit failure, native callback/coordinator retention,
workflow version checks, trainer selection and unsupported configuration rejection.
They do not execute real torch/FSDP/HCCL or prove numerical weight equivalence.

Record all of the following on NPU before marking MVP-3 accepted:

1. v0 bootstrap matches. Update v0 -> v1 -> v2 -> v3; vLLM PIDs and transport IDs
   remain stable, with one communicator initialization and no snapshot restarts.
2. Two real FSDP training ranks participate; only rank0 sends to receiver NPUs.
   Verify actual post-update outputs/logprobs or equivalent numerical evidence,
   not only the route-derived serving version labels.
3. Update while a real rollout is active. Verify planned abort, zero physical
   inflight before transfer, committed-version continuation and equal lengths for
   tokens/logprobs/versions. Proxy and evaluation follow committed versions.
4. Inject sender, receiver and commit failures. No partial policy is admitted;
   ranks terminate within configured native timeouts; no crash replay or automatic
   rollback occurs. Restart both trainer and HCCL deployment from clean bootstrap.
5. Repeat with two external replicas, then rerun the existing verl Direct HCCL
   smoke against the extracted shared implementation.
6. Gracefully stop trainer and Core; inspect Ray actors, guardian/vLLM processes,
   API ports and NPU memory. Keep SIGTERM and SIGKILL tests separate.

HCCL members remain fixed. If an in-memory committed version has moved beyond
the disk bootstrap, stale replacement is still blocked. MVP-4 owns dynamic join
and scaling; this milestone does not remove that safety gate.
