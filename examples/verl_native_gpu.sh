#!/usr/bin/env bash
# Single-node, 8-GPU smoke recipe. Requires the paired native verl entry patch.
set -euo pipefail

MODEL="${MODEL:-/shared/models/Qwen3-8B}"
TRAIN_DATA="${TRAIN_DATA:-/shared/data/gsm8k/train.parquet}"
VAL_DATA="${VAL_DATA:-/shared/data/gsm8k/test.parquet}"
OUTPUT="${OUTPUT:-/shared/er-results/native-single}"
TP="${TP:-2}"
MAX_REPLICAS="${MAX_REPLICAS:-$((4 / TP))}"
STEPS="${STEPS:-10}"

python -m verl.trainer.main_ppo \
  actor_rollout_ref.model.path="$MODEL" \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size=32 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  trainer.device=cuda \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.parameter_sync_step=1 \
  trainer.resume_mode=disable \
  actor_rollout_ref.hybrid_engine=false \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.total_training_steps="$STEPS" \
  trainer.total_epochs=1000 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.default_local_dir="$OUTPUT" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.8 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  +elastic.enabled=true \
  +elastic.min_replicas=1 \
  +elastic.max_replicas="$MAX_REPLICAS" \
  "$@"
