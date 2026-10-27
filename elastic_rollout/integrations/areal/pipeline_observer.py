"""Observe the single AReaL controller that owns all workflow result shards.

The controller dispatcher already aggregates worker/DP results. Observe it
once; querying each FSDP rank would duplicate both counts and timings.
"""

from threading import RLock, get_ident

from elastic_rollout.pipeline_observer import PipelineObserver, exclude_method, wrap_method


class AReaLPipelineObserver:
    def __init__(self, trainer, control):
        self.trainer = trainer
        self.rollout = trainer.rollout
        self.dispatcher = self.rollout.dispatcher
        self.floors, self.lock = {}, RLock()
        self.owner = None
        self.batch_size = trainer.train_dataloader.batch_size
        self.observer = PipelineObserver(control, "workflow_group", self.metadata)
        self._install()

    def metadata(self, status):
        dispatcher = self.dispatcher
        with dispatcher._result_cv:
            # None is a rejected attempt, never usable training inventory.
            keys = [
                key for key, value in dispatcher._pending_results.items() if value.data is not None
            ]
            with self.lock:
                floors = [self.floors.get(key) for key in keys]
        stats = dispatcher.staleness_manager.get_stats()
        oldest = min(floors) if floors and all(v is not None for v in floors) else None
        return dict(
            available_work=bool(stats.running or stats.enqueued),
            dispatch_unconstrained=(stats.running + stats.enqueued > status["ready_replicas"]),
            attribution_known=True,
            eligible_units=len(keys),
            next_batch_units=self.batch_size,
            eligibility_known=True,
            oldest_buffer_policy_version=oldest,
        )

    def _wait_reason(self):
        manager = self.dispatcher.staleness_manager
        with manager.lock:
            stat = manager.rollout_stat
            capacity = (
                (manager.max_staleness + self.rollout.get_version() + 1)
                * max(1, manager.consumer_batch_size)
                - stat.accepted
                - stat.running
            )
            # When tasks are already running they can still supply the batch.
            # No runnable task and no native version budget is staleness wait.
            return "staleness" if capacity <= 0 and stat.running == 0 else "data"

    def _install(self):
        observer, trainer, dispatcher = self.observer, self.trainer, self.dispatcher

        def submit(original, task):
            with self.lock:
                self.floors[task.task_id] = self.rollout.get_version()
            return original(task)

        def wait(original, *args, **kwargs):
            if get_ident() != self.owner:
                return original(*args, **kwargs)
            with observer.waiting(self._wait_reason()):
                return original(*args, **kwargs)

        def results(original, *args, **kwargs):
            values = original(*args, **kwargs)
            if get_ident() == self.owner:
                accepted = [value for value in values if value is not None]
                with dispatcher._result_cv:
                    active = set(dispatcher._active_task_ids)
                with self.lock:
                    floors = [self.floors.get(value.task_id) for value in accepted]
                    self.floors = {
                        key: value for key, value in self.floors.items() if key in active
                    }
                oldest = min(floors) if floors and all(v is not None for v in floors) else None
                observer.consumed_groups(len(accepted), oldest)
            return values

        def prepare(original, *args, **kwargs):
            observer.begin()
            self.owner = get_ident()
            try:
                return original(*args, **kwargs)
            except BaseException:
                observer.finish(False)
                raise
            finally:
                self.owner = None
                observer.acquired()

        def save(original, *args, **kwargs):
            # This native boundary follows actor AND critic computation and
            # precedes save/recovery/update/eval, even when saving is disabled.
            observer.finish()
            with observer.excluded("paused"):
                return original(*args, **kwargs)

        def version(original, value):
            result = original(value)
            with observer.lock:
                observer.version = value
            return result

        def train(original, *args, **kwargs):
            observer.start()
            try:
                return original(*args, **kwargs)
            finally:
                observer.finish(False)
                observer.close()

        wrap_method(dispatcher, "submit_task_input", submit)
        wrap_method(dispatcher._result_cv, "wait", wait)
        # Count only the final accepted batch, after native dynamic_bs has
        # applied its attempt budget (wait_results can drain discarded surplus).
        wrap_method(dispatcher, "active_submit_and_wait", results)
        wrap_method(self.rollout, "prepare_batch", prepare)
        wrap_method(self.rollout, "set_version", version)
        wrap_method(trainer, "_save_hf", save)
        wrap_method(trainer, "train", train)
        exclude_method(trainer, "_save_recover_checkpoint", observer, "paused")
        exclude_method(trainer, "_evaluate", observer, "eval")
        exclude_method(trainer.actor, "update_weights", observer, "sync")
