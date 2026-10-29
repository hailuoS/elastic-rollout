import argparse
import json
import logging
import os
import signal
import threading
from pathlib import Path

from .api import DeploymentSpec, ResourceSlot
from .client import ControlClient
from .engine.vllm import VLLMServiceDriver
from .resource import StaticResourceProvider
from .runtime import Controller, Reconciler
from .server import make_server


def main() -> None:
    parser = argparse.ArgumentParser(description="ElasticRollout control plane")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--config", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--ray-address", default="auto")
    serve.add_argument("--expected-gcs-address")
    serve.add_argument("--shutdown-timeout", type=float, default=900)
    autoscale = sub.add_parser(
        "autoscale", help="Pipeline pressure controller (advisory by default)"
    )
    autoscale.add_argument("--endpoint", required=True)
    autoscale.add_argument("--deployment", required=True)
    autoscale.add_argument("--config", type=Path)
    autoscale.add_argument("--once", action="store_true")
    autoscale.add_argument("--activate", action="store_true")
    verl = sub.add_parser("verl", help="verl separate_async integration")
    verl_sub = verl.add_subparsers(dest="verl_command", required=True)
    run = verl_sub.add_parser("run")
    run.add_argument("-c", "--config", type=Path, required=True)
    run.add_argument("--report", type=Path, help="Write process and cleanup results as JSON")
    for command in ("status", "scale"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--endpoint", required=True)
        cmd.add_argument("--deployment", required=True)
        if command == "scale":
            cmd.add_argument("count", type=int)
            cmd.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "verl":
        from .integrations.verl.launcher import run_launch

        raise SystemExit(
            run_launch(
                args.config,
                report_path=args.report,
            )
        )
    token = os.environ.get("ELASTIC_ROLLOUT_TOKEN")
    if args.command != "serve":
        client = ControlClient(args.endpoint, args.deployment, token=token)
        if args.command == "autoscale":
            from .autoscaling import AutoscalingConfig, run_autoscaler

            config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            run_autoscaler(
                client, AutoscalingConfig(**config), once=args.once, activate=args.activate
            )
            return
        if args.command == "scale":
            client.set_desired_replicas(args.count)
            if args.wait:
                client.wait_ready(args.count)
        print(json.dumps(client.status(), indent=2))
        return

    from .launcher.ray import RayProcessLauncher

    config = json.loads(args.config.read_text(encoding="utf-8"))
    slots = [ResourceSlot(**slot) for slot in config["slots"]]
    resources = StaticResourceProvider(slots)
    controller = Controller(
        resources,
        VLLMServiceDriver(
            RayProcessLauncher(args.ray_address, expected_gcs=args.expected_gcs_address)
        ),
    )
    # Ray.init replaces the driver's SIGTERM handler. Install ours afterwards.
    stop_requested = threading.Event()
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, lambda _signum, _frame: stop_requested.set())
    server = None
    reconciler = None
    serve_thread = None
    serve_errors = []
    try:
        for deployment in config["deployments"]:
            if stop_requested.is_set():
                break
            controller.ensure_deployment(DeploymentSpec.from_dict(deployment))
        if not stop_requested.is_set():
            server = make_server(controller, args.host, args.port, token)
            reconciler = Reconciler(controller)
            reconciler.start()

            def serve_loop():
                try:
                    server.serve_forever(poll_interval=0.2)
                except BaseException as exc:
                    serve_errors.append(exc)
                finally:
                    stop_requested.set()

            serve_thread = threading.Thread(target=serve_loop, name="control-http")
            serve_thread.start()
            stop_requested.wait()
    finally:
        # Keep the API alive while draining, so clients can release/renew leases.
        try:
            if reconciler is not None:
                reconciler.stop()
            controller.close(timeout=args.shutdown_timeout)
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
            if serve_thread is not None:
                serve_thread.join()
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
    if serve_errors:
        raise serve_errors[0]


if __name__ == "__main__":
    main()
