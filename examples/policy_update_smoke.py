"""Shared-FS snapshot prepare/commit smoke. Leaves the deployment at four replicas."""

import argparse
import asyncio
import json
import os
import time

from elastic_rollout.client import ControlClient, ControlError
from elastic_rollout.generation import GenerationClient


def stable_call(call, timeout=900):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return call()
        except ControlError as exc:
            if exc.status != 409 or time.monotonic() >= deadline:
                raise
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--prompt-ids", required=True)
    args = parser.parse_args()
    control = ControlClient(args.endpoint, args.deployment, os.environ.get("ELASTIC_ROLLOUT_TOKEN"))
    status = control.status()
    if args.version <= status["policy"]["version"]:
        raise ValueError("--version must advance the committed policy")
    generator = GenerationClient(control, status["spec"]["replica"]["served_model_name"])
    prompts = json.loads(args.prompt_ids)

    def generate(expected):
        result = asyncio.run(
            generator.generate(
                f"policy-smoke-{expected}", prompts, {"max_tokens": 8, "temperature": 0}
            )
        )
        assert result.token_ids and set(result.versions) == {expected}
        print(f"generation policy={expected}, tokens={len(result.token_ids)}")

    stable_call(lambda: control.set_desired_replicas(2))
    control.wait_ready(2)
    generate(status["policy"]["version"])
    update = stable_call(lambda: control.begin_policy_update(args.snapshot))
    control.wait_policy_update(update["update_id"])
    assert not control.status()["accepting_routes"]
    stable_call(lambda: control.commit_policy_update(update["update_id"], args.version))
    generate(args.version)
    stable_call(lambda: control.set_desired_replicas(4))
    control.wait_ready(4)
    replicas = [r for r in control.replicas() if r["state"] == "READY"]
    assert len(replicas) == 4
    assert all(
        r["loaded_model_path"] == args.snapshot and r["serving_policy_version"] == args.version
        for r in replicas
    )
    generate(args.version)
    print(f"all four replicas serve committed policy {args.version} from {args.snapshot}")


if __name__ == "__main__":
    main()
