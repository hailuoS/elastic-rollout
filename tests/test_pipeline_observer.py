from types import SimpleNamespace as NS

import pytest
from helpers_elastic import grow, update

from elastic_rollout.pipeline import snapshot_from_body
from elastic_rollout.pipeline_observer import PipelineObserver
from elastic_rollout.runtime import PolicyConflict


@pytest.mark.parametrize("destroy_fails", [False, True])
@pytest.mark.parametrize("backend,device", [("hccl", "npu"), ("nccl", "cuda")])
def test_optional_sender_failure_isolates_only_after_native_destroy(
    monkeypatch, destroy_fails, backend, device
):
    import threading

    from test_integrations import stub_module

    from elastic_rollout.hccl_sender import HCCLSender

    owner = threading.get_ident()
    calls = []

    def destroy(comm):
        assert threading.get_ident() == owner
        calls.append("destroy")
        if destroy_fails:
            raise RuntimeError("destroy failed")

    stub_module(
        monkeypatch,
        "torch",
        **{
            device: NS(
                current_stream=lambda: NS(synchronize=lambda: None), synchronize=lambda: None
            )
        },
    )
    sender = HCCLSender(1024, True, endpoint="http://unused", deployment_id="test", backend=backend)
    child = HCCLSender(1024, True, endpoint="http://unused", deployment_id="test", backend=backend)
    base = dict(update_id="u", session_id="base", transport_id="t0", target_version=1)
    sender.session = dict(base, transports=[{"transport_key": "base", "session": base}])
    child.session = dict(base, session_id="child", transport_id="t1")
    child.group = NS(
        comm=123, available=True, **{backend: NS(**{f"{backend}CommDestroy": destroy})}
    )
    child._optional, child._transport_key = True, "candidate"
    sender._children = {"candidate": child}
    sender._active_transports = [sender, child]
    sender._send_bucket = lambda *args: calls.append("base")
    child.control.transfer_operation = lambda *args, **kwargs: {"optional_failure": "send error"}

    def control(uid, action, **kwargs):
        calls.append(action)
        return {"receiver_destroyed": "child"}

    sender.control.transfer_operation = control
    child._send_bucket = lambda *args: child._overlap("group", lambda: None)
    if destroy_fails:
        with pytest.raises(RuntimeError, match="destroy failed"):
            sender._fanout_bucket([], {}, 0)
        assert sender._poisoned and "isolate" not in calls and "fail" in calls
    else:
        sender._fanout_bucket([], {}, 0)
        sender._fanout_bucket([], {}, 1)
        assert calls == ["base", "destroy", "destroy", "isolate", "base"]
        assert not sender._poisoned and not sender._children


def test_route_cancellation_survives_concurrent_paused_admission():
    import asyncio
    import threading

    from elastic_rollout.client import ControlClient, RoutingPaused
    from elastic_rollout.generation import GenerationClient

    async def run():
        entered, release = threading.Event(), threading.Event()
        client = ControlClient("http://unused", "test")

        def acquire(key):
            entered.set()
            assert release.wait(2)
            raise RoutingPaused(503, "paused")

        client.acquire = acquire
        task = asyncio.create_task(
            GenerationClient(client, "policy").generate("key", [1], {"max_tokens": 1})
        )
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)

    asyncio.run(run())


def test_areal_native_dynamic_batch_counts_delivered_groups_and_excludes_none(monkeypatch):
    import threading
    import time

    from test_integrations import ROOT, source_methods

    from elastic_rollout.integrations.areal.pipeline_observer import AReaLPipelineObserver

    methods = source_methods(
        ROOT / "AReaL/areal/infra/workflow_executor.py",
        "BatchTaskDispatcher",
        ["active_submit_and_wait", "wait_results"],
        {"time": time, "random": NS(shuffle=lambda values: None)},
    )
    dispatcher = type("NativeDispatcher", (), methods)()
    dispatcher._input_cv = threading.Condition()
    dispatcher._result_cv = threading.Condition()
    dispatcher._pending_inputs = []
    dispatcher._pending_results = {}
    dispatcher._active_task_ids = set()
    dispatcher._check_thread_exception = lambda: None
    dispatcher.enable_tracing = False
    dispatcher.runner = NS(max_queue_size=100, get_input_queue_size=lambda: 0)
    stat = NS(accepted=10000, running=3, enqueued=2)  # recovered cumulative count != inventory
    dispatcher.staleness_manager = NS(
        get_pending_limit=lambda: 100,
        get_stats=lambda: stat,
        lock=threading.Lock(),
        rollout_stat=stat,
        max_staleness=2,
        consumer_batch_size=4,
    )

    def submit(task):
        dispatcher._active_task_ids.add(task.task_id)
        dispatcher._pending_results[task.task_id] = NS(
            task_id=task.task_id,
            create_time=task.task_id,
            data=NS(task_id=task.task_id) if task.task_id % 2 == 0 else None,
        )

    dispatcher.submit_task_input = submit
    rollout = NS(dispatcher=dispatcher, get_version=lambda: 2, set_version=lambda version: None)
    rollout.prepare_batch = lambda: dispatcher.active_submit_and_wait(
        (NS(task_id=i) for i in range(100)),
        batch_size=4,
        dynamic_bs=True,
    )
    trainer = NS(
        rollout=rollout,
        train_dataloader=NS(batch_size=4),
        _save_hf=lambda: None,
        _save_recover_checkpoint=lambda: None,
        _evaluate=lambda: None,
        train=lambda: None,
        actor=NS(update_weights=lambda: None),
    )
    observer = AReaLPipelineObserver(trainer, NS())
    delivered = rollout.prepare_batch()
    trainer._save_hf()
    assert len(delivered) == 2
    assert observer.observer.window.consumed_total == 2
    assert observer.observer.versions[-1] == 2
    metadata = observer.metadata({"ready_replicas": 1})
    assert metadata["eligible_units"] == 0  # never accepted - consumed
    dispatcher._pending_results = {42: NS(data=None), 43: NS(data=object())}
    observer.floors[43] = 1
    metadata = observer.metadata({"ready_replicas": 1})
    assert metadata["eligible_units"] == 1
    assert metadata["oldest_buffer_policy_version"] == 1


@pytest.mark.parametrize("backend", ["elastic_hccl", "elastic_nccl"])
def test_verl_sampler_uses_native_wait_predicate_and_committed_version_floor(monkeypatch, backend):
    from test_integrations import PACKAGE, ROOT, load_adapter, stub_module

    for name in ("verl", "verl.trainer", "verl.trainer.ppo", "verl.trainer.ppo.v1", "verl.utils"):
        stub_module(monkeypatch, name)
    stub_module(monkeypatch, "numpy")
    data = {
        "train": {
            "a": {"is_prompt": True, "status": "finished", "global_steps": 10},
            "b": {"is_prompt": True, "status": "finished", "global_steps": 10},
            "c": {"is_prompt": True, "status": "running", "global_steps": 1},
            "a_0_0": {"status": "finished"},
            "b_0_0": {"status": "finished"},
        }
    }
    stub_module(
        monkeypatch,
        "transfer_queue",
        KVBatchMeta=NS,
        kv_list=lambda: data,
        kv_clear=lambda **kwargs: None,
    )
    stub_module(monkeypatch, "omegaconf", DictConfig=dict)
    stub_module(
        monkeypatch,
        "verl.utils.skip",
        SkipManager=NS(annotate_tq=lambda **kwargs: lambda method: method),
    )
    native = load_adapter(
        monkeypatch,
        "verl.trainer.ppo.v1.replay_buffer",
        ROOT / "verl/verl/trainer/ppo/v1/replay_buffer.py",
    )
    module = load_adapter(
        monkeypatch,
        "elastic_test_pipeline_sampler",
        PACKAGE / "integrations/verl/pipeline_sampler.py",
    )

    class Trainer:
        parameter_sync_step = 2
        global_steps = 10
        config = NS(
            data=NS(train_batch_size=4),
            actor_rollout_ref=NS(
                hybrid_engine=False,
                rollout=NS(
                    checkpoint_engine=NS(
                        backend=backend,
                        engine_kwargs={
                            backend: {"endpoint": "http://unused", "deployment_id": "test"}
                        },
                    )
                ),
            ),
        )

        def _add_prompts_to_generate(self, count):
            return count

        def _resolve_filter_groups_metric(self):
            return None

        _submit_batch_to_rollout = _step_once = on_step_end = fit = _validate = _save_checkpoint = (
            lambda *args, **kwargs: None
        )

    trainer = Trainer()
    sampler = module.ElasticPipelineReplayBuffer(
        trainer_mode="separate_async",
        trainer_config={},
        max_off_policy_threshold=3,
        max_off_policy_strategy="wait",
        sampler_kwargs={},
        refill_fn=trainer._add_prompts_to_generate,
    )
    assert isinstance(sampler, native.ReplayBufferAsync)
    sampler._floors = {"a": 2, "b": 3, "c": 0}
    status = {"ready_replicas": 1}
    assert sampler._metadata(status)["eligible_units"] == 0  # native wait barrier
    sampler.max_off_policy_strategy = sampler._view.max_off_policy_strategy = "drop"
    sampler.max_off_policy_threshold = sampler._view.max_off_policy_threshold = 100
    metadata = sampler._metadata(status)
    assert metadata["eligible_units"] == 2
    assert metadata["next_batch_units"] == 2  # parameter_sync_step, not tensor rows
    assert metadata["oldest_buffer_policy_version"] == 2  # not global_steps=10
    sampler._sync_metadata_from_transfer_queue()
    sampler._sampling = True
    sampler.observer.begin()
    sampler._select_prompt_uids("train", {"a", "b"}, 2)
    assert sampler.observer.consumed == 2
    assert sampler.observer.consumed_version == 2


def observer():
    clock = [0.0]
    status = {"spec": {"elastic_scaling": True}, "scaling": {"revision": 1}}
    reports = []
    metadata = dict(
        available_work=True,
        dispatch_unconstrained=True,
        attribution_known=True,
        eligible_units=10,
        next_batch_units=2,
        eligibility_known=True,
        oldest_buffer_policy_version=0,
    )
    client = NS(status=lambda: status, publish_pipeline=reports.append)
    instance = PipelineObserver(client, "prompt_group", lambda _: metadata, clock=lambda: clock[0])
    return instance, clock, status, reports


def test_disjoint_timings_exclude_sync_and_keep_native_consumption():
    o, clock, _, reports = observer()
    o.begin()
    with o.waiting("data"):
        clock[0] += 2
    with o.waiting("staleness"):
        clock[0] += 3
    o.acquired()
    o.consumed_groups(2, 0)
    clock[0] += 7
    with o.excluded("sync"):
        clock[0] += 100
        assert o.report_once()["phase"] == "sync"
    o.finish()
    body = o.report_once()
    assert (body["wait_data_s"], body["wait_staleness_s"], body["train_path_s"]) == (2, 3, 7)
    assert body["completed_cycles"] == 1 and body["consumed_total"] == 2
    assert body["ongoing_wait_s"] == 0
    snapshot_from_body(body)
    assert reports[-1] == body
    o.close()
    assert reports[-1]["phase"] == "shutdown"
    assert reports[-1]["sequence"] > body["sequence"]


def test_long_wait_reports_before_cycle_finishes_and_accumulates_polls():
    o, clock, _, _ = observer()
    o.begin()
    for _ in range(3):
        with o.waiting("data"):
            clock[0] += 2
    body = o.report_once()
    assert body["ongoing_wait_s"] == body["wait_data_s"] == 6
    assert body["completed_cycles"] == body["consumed_total"] == 0
    snapshot_from_body(body)


def test_revision_discards_old_capacity_evidence_without_resetting_counters():
    o, clock, status, _ = observer()
    o.report_once()
    o.begin()
    o.consumed_groups(2, 0)
    clock[0] += 10
    o.finish()
    assert o.report_once()["train_path_s"] == 10
    status["scaling"]["revision"] += 1
    body = o.report_once()
    assert body["train_path_s"] == 0 and body["consumed_total"] == 2
    assert body["oldest_consumed_policy_version"] is None


def test_revision_mid_cycle_waits_for_complete_new_cycles_without_poisoning_window():
    o, clock, status, _ = observer()
    o.report_once()
    o.begin()
    o.consumed_groups(2, 0)
    clock[0] += 10
    status["scaling"]["revision"] += 1
    body = o.report_once()
    assert body["train_path_s"] == body["wait_unknown_s"] == 0
    clock[0] += 2
    o.finish()
    body = o.report_once()
    assert body["window_cycles"] == 0 and body["completed_cycles"] == 1
    assert body["consumed_total"] == 2
    o.begin()
    o.consumed_groups(2, 0)
    clock[0] += 3
    o.finish()
    body = o.report_once()
    assert body["window_cycles"] == 1 and body["completed_cycles"] == 2
    assert body["train_path_s"] == 3 and body["consumed_total"] == 4
    assert body["wait_unknown_s"] == 0


def test_missing_version_is_not_replaced_by_later_known_group():
    o, clock, _, _ = observer()
    o.begin()
    o.consumed_groups(1, None)
    o.consumed_groups(1, 0)
    clock[0] += 1
    o.finish()
    assert o.report_once()["oldest_consumed_policy_version"] is None


def test_optional_finish_failure_requires_native_destroy_ack_before_commit(elastic):
    c, m, driver, _ = elastic
    update(m, 1)
    (candidate,) = grow(c, m, 3)
    original = driver.weight_transfer_control.finish_weight_update

    def finish(endpoint, **kwargs):
        if endpoint == candidate.endpoint:
            raise RuntimeError("explicit receiver finish error")
        return original(endpoint, **kwargs)

    driver.weight_transfer_control.finish_weight_update = finish

    def before_commit():
        uid = m.policy_update.update_id
        with pytest.raises(PolicyConflict, match="acknowledge isolation"):
            m.commit_policy_update(uid, 2)
        child = m.transfer.children[candidate.replica_id]
        body = dict(
            transport_key=candidate.replica_id,
            session_id=child.session.session_id,
            transport_id=child.transport.transport_id,
            sender_destroyed=True,
        )
        with pytest.raises(PolicyConflict, match="sender and receiver"):
            m.transfer.execute(uid, "isolate", body)
        m.transfer.execute(uid, "destroy", body)
        m.transfer.execute(uid, "isolate", body)

    update(m, 2, before_commit=before_commit)
    assert m.serving_policy_version == 2 and candidate.cancel_requested
    assert candidate.replica_id not in m.transfer.children
    assert not m.transfer.poisoned
