"""Verl V1 separate_async observer using the supported custom-sampler seam."""

from threading import RLock

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

from elastic_rollout.client import ControlClient
from elastic_rollout.pipeline_observer import PipelineObserver, exclude_method, wrap_method


class ElasticPipelineReplayBuffer(ReplayBufferAsync):
    def __init__(self, **kwargs):
        trainer = kwargs["refill_fn"].__self__
        config = trainer.config
        if kwargs["trainer_mode"] != "separate_async" or config.actor_rollout_ref.hybrid_engine:
            raise ValueError("pipeline observer requires disaggregated V1 separate_async")
        # Verl deliberately omits built-in DAPO options for external samplers.
        # Recover the exact native setting rather than silently disabling it.
        kwargs["filter_groups_metric"] = trainer._resolve_filter_groups_metric()
        super().__init__(**kwargs)
        checkpoint = config.actor_rollout_ref.rollout.checkpoint_engine
        options = checkpoint.engine_kwargs[checkpoint.backend]
        import os

        control = ControlClient(
            options["endpoint"],
            options["deployment_id"],
            os.environ.get("ELASTIC_ROLLOUT_TOKEN"),
            timeout=2,
        )
        self._floors, self._floor_lock = {}, RLock()
        # The reporter owns this independent read-only metadata view. Never
        # refresh the training sampler from a background thread.
        self._view = ReplayBufferAsync(**kwargs)
        self._trainer = trainer
        self._batch_size = config.data.train_batch_size // trainer.parameter_sync_step
        self._sampling = False
        self.observer = PipelineObserver(control, "prompt_group", self._metadata)
        self._install(trainer)

    def _install(self, trainer):
        observer = self.observer

        def submit(original, batch):
            # A lower bound in committed policy-version space, not trainer
            # global_steps (warmup and parameter_sync_step use another clock).
            with self._floor_lock:
                self._floors.update({str(uid): observer.version for uid in batch["uid"]})
            return original(batch)

        def step(original, *args, **kwargs):
            observer.begin()
            try:
                result = original(*args, **kwargs)
            except BaseException:
                observer.finish(False)
                raise
            observer.finish()
            return result

        def sync(original, *args, **kwargs):
            with observer.excluded("sync"):
                result = original(*args, **kwargs)
                with observer.lock:
                    observer.version = trainer.global_steps
                return result

        def fit(original, *args, **kwargs):
            observer.start()
            try:
                return original(*args, **kwargs)
            finally:
                observer.close()

        wrap_method(trainer, "_submit_batch_to_rollout", submit)
        wrap_method(trainer, "_step_once", step)
        wrap_method(trainer, "on_step_end", sync)
        wrap_method(trainer, "fit", fit)
        exclude_method(trainer, "_validate", observer, "eval")
        exclude_method(trainer, "_save_checkpoint", observer, "paused")

    def _metadata(self, status):
        view = self._view
        view._sync_metadata_from_transfer_queue()
        # Use the framework's exact drop/wait and DAPO classification. This
        # read-only view never evicts, selects, refills or materializes tensors.
        step = getattr(self._trainer, "global_steps", 0)
        reasons = view._terminal_eviction_reasons(step, "train")
        keys = view._sampleable_terminal_keys("train", reasons)
        eligible = view._has_enough_samples(step, "train", self._batch_size, keys)
        inflight = view.pending_keys["train"] | view.running_keys["train"]
        with self._floor_lock:
            floors = [self._floors.get(key) for key in keys]
        oldest = min(floors) if floors and all(v is not None for v in floors) else None
        return dict(
            available_work=bool(inflight),
            # More than one outstanding prompt per serving replica is needed
            # before claiming added physical capacity can receive work.
            dispatch_unconstrained=len(inflight) > status["ready_replicas"],
            attribution_known=True,
            eligible_units=len(keys) if eligible else 0,
            next_batch_units=self._batch_size,
            eligibility_known=True,
            oldest_buffer_policy_version=oldest,
        )

    def sample(self, global_steps, partition_id, batch_size):
        if partition_id != "train":
            return super().sample(global_steps, partition_id, batch_size)
        self._sampling = True
        try:
            return super().sample(global_steps, partition_id, batch_size)
        finally:
            self._sampling = False
            self.observer.acquired()

    def _has_enough_samples(self, global_steps, partition_id, batch_size, sampleable_keys):
        result = super()._has_enough_samples(
            global_steps, partition_id, batch_size, sampleable_keys
        )
        if self._sampling and partition_id == "train":
            # A sufficient terminal inventory blocked by native wait policy is
            # staleness wait. A shortage is data wait unless stale in-flight
            # prompts independently block the native sampling predicate.
            self._wait_reason = (
                "staleness"
                if not super()._has_enough_samples(global_steps, partition_id, 0, sampleable_keys)
                else "data"
            )
        return result

    def _wait_for_next_poll(self, partition_id, last_debug_time):
        if self._sampling and partition_id == "train":
            with self.observer.waiting(self._wait_reason):
                return super()._wait_for_next_poll(partition_id, last_debug_time)
        return super()._wait_for_next_poll(partition_id, last_debug_time)

    def _select_prompt_uids(self, partition_id, sampleable_keys, batch_size):
        result = super()._select_prompt_uids(partition_id, sampleable_keys, batch_size)
        if self._sampling and partition_id == "train":
            with self._floor_lock:
                floors = [self._floors.pop(key, None) for key in result[0]]
            oldest = min(floors) if floors and all(v is not None for v in floors) else None
            self.observer.consumed_groups(len(result[0]), oldest)
        return result

    def _clear_groups(self, partition_id, uids):
        result = super()._clear_groups(partition_id, uids)
        if partition_id == "train":
            with self._floor_lock:
                for uid in uids:
                    self._floors.pop(uid, None)
        return result
