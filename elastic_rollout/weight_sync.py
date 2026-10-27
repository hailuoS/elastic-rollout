"""Shared policy transactions for framework-owned distributed weight producers."""

import json
import logging
import time

from elastic_rollout.client import ControlError

logger = logging.getLogger(__name__)
_LIFECYCLE_RETRY_WAIT_BUDGET_S = 10.0


def _is_retryable_lifecycle_conflict(exc: ControlError) -> bool:
    if exc.status != 409:
        return False
    try:
        payload = json.loads(str(exc))
    except (TypeError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("error") == "deployment lifecycle mutation in progress"
    )


class ElasticWeightSync:
    def _begin_elastic_update(self, version, spec):
        deadline = time.monotonic() + spec["drain_grace_period"] + spec["retirement_timeout_s"] + 10
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("elastic topology did not reach policy safe point")
            self.maintain_transports()
            current = self.control.status()
            previous = current.get("policy_update")
            if previous and previous["state"] != "COMMITTED":
                raise RuntimeError("previous HCCL update active or failed; cannot begin another")
            if "shutdown" in current.get("route_pause_reasons", ()):
                raise RuntimeError("controller is shutting down")
            if current["scaling"]["policy_safe_point"]:
                try:
                    return self.control.begin_external_update(version)
                except ControlError as exc:
                    try:
                        code = json.loads(str(exc)).get("code")
                    except (ValueError, AttributeError):
                        code = None
                    # Only explicit, pre-transaction rejections are retryable.
                    # Never retry transport/start/commit calls or lost responses.
                    if not (
                        _is_retryable_lifecycle_conflict(exc)
                        or (exc.status == 409 and code == "ELASTIC_TOPOLOGY_NOT_READY")
                    ):
                        raise
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    def maintain_transports(self):
        pending = self.control.transport_maintenance()
        if pending.get("poisoned"):
            raise RuntimeError("transport retirement ambiguous; clean restart required")
        if pending["requests"]:
            try:
                results = self._actors("retire_transports", requests=pending["requests"])
            except BaseException as exc:
                try:
                    self.control.transport_maintenance({"error": str(exc)})
                except Exception:
                    pass
                raise
            masters = [r for r in results if r]
            if len(masters) != 1:
                raise ValueError("exactly one retirement acknowledgement producer required")
            self.control.transport_maintenance(masters[0])

    def _actors(self, method, **kwargs):
        raise NotImplementedError("adapter must dispatch producer control to all training ranks")

    def _send_weights(self, version):
        raise NotImplementedError("adapter must await every training rank's weight export")

    def _begin_external_update_with_retry(self, version):
        # Only rejected lifecycle-lock acquisitions are safe to retry. This is
        # cumulative sleep time; individual HTTP calls have their own timeout.
        remaining_wait = _LIFECYCLE_RETRY_WAIT_BUDGET_S
        delay = 0.1
        retries = 0
        while True:
            try:
                update = self.control.begin_external_update(version)
                if retries:
                    logger.info(
                        "external update began after lifecycle contention version=%s retries=%s",
                        version,
                        retries,
                    )
                return update
            except ControlError as exc:
                if not _is_retryable_lifecycle_conflict(exc) or remaining_wait <= 0:
                    raise
                if retries == 0:
                    logger.warning(
                        "deployment lifecycle busy before external update version=%s; retrying",
                        version,
                    )
                sleep_for = min(delay, remaining_wait)
                before_sleep = time.monotonic()
                time.sleep(sleep_for)
                remaining_wait -= max(sleep_for, time.monotonic() - before_sleep)
                delay = min(delay * 2, 0.5)
                retries += 1

    def _update(self, version):
        started = time.monotonic()
        status = self.control.status()
        if version == 0 and status.get("spec", {}).get("elastic_scaling"):
            self.maintain_transports()
            # A prior drain may need time to finish or roll back. Maintenance
            # runs here, before BEGIN freezes topology, and never on NPU workers.
            budget = (
                status["spec"]["drain_grace_period"] + status["spec"]["retirement_timeout_s"] + 10
            )
            deadline = time.monotonic() + budget
            while not self.control.status()["scaling"]["policy_safe_point"]:
                current = self.control.status()
                if current.get("policy_update") and current["policy_update"]["state"] == "FAILED":
                    raise RuntimeError("previous HCCL update failed; clean restart required")
                if time.monotonic() >= deadline:
                    raise TimeoutError("elastic topology did not reach policy safe point")
                self.maintain_transports()
                time.sleep(0.1)
        if version == 0:
            if (
                status["serving_policy_version"] != 0
                or status["bootstrap_snapshot"]
                != {"version": 0, "model_path": self.bootstrap_model_path}
                or status["policy_update"] is not None
                or not status["accepting_routes"]
            ):
                raise ValueError(
                    "initial v0 equivalence not confirmed; use matching fresh bootstrap"
                )
            return {}
        update = (
            self._begin_elastic_update(version, status["spec"])
            if status.get("spec", {}).get("elastic_scaling")
            else self._begin_external_update_with_retry(version)
        )
        uid = update["update_id"]
        metrics = {}

        def timed(name, function, *args, **kwargs):
            before = time.monotonic()
            result = function(*args, **kwargs)
            metrics["elastic_hccl/" + name] = time.monotonic() - before
            return result

        try:
            pause_started = time.monotonic()
            timed("pause_to_inflight_zero_s", self.control.pause_generation)
            metadata = self._actors(
                "prepare",
                **(
                    {"transport_plan": update["transport_plan"]}
                    if update.get("transport_plan")
                    else {}
                ),
            )
            if len([item for item in metadata if item]) != 1:
                raise ValueError("exactly one trainer rank0 must provide rendezvous metadata")
            master = next(item for item in metadata if item)
            session = self.control.transfer_operation(uid, "prepare", backend="hccl", **master)
            if session["needs_initialize"]:
                timed("hccl_init_s", self._actors, "configure_session", session=session)
            else:
                self._actors("configure_session", session=session)
                metrics["elastic_hccl/hccl_init_s"] = 0.0
            sid = session["session_id"]
            started_session = timed(
                "start_weight_update_s",
                self.control.transfer_operation,
                uid,
                "start",
                session_id=sid,
            )
            if update.get("transport_plan"):
                self._actors("settle_optional", session=started_session)
            for result in self._send_weights(version):
                if isinstance(result, dict):
                    metrics.update(result)
            finished_session = timed(
                "finish_verify_s", self.control.transfer_operation, uid, "finish", session_id=sid
            )
            if update.get("transport_plan"):
                self._actors("settle_optional", session=finished_session)
            timed("commit_s", self.control.commit_policy_update, uid, version)
            self.control.resume_generation()
            resumed_at = time.monotonic()
        except BaseException as exc:
            try:
                self.control.transfer_operation(uid, "fail", error=str(exc))
            except Exception:
                pass
            # No resume/rollback, including ambiguous sender/receiver failures.
            raise
        metrics["elastic_hccl/weight_sync_total_s"] = time.monotonic() - started
        metrics["elastic_hccl/logical_generation_pause_s"] = resumed_at - pause_started
        logger.info(
            "HCCL update committed update_id=%s version=%s metrics=%s", uid, version, metrics
        )
        return metrics
