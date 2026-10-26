import argparse
import json
import logging
import os
from pathlib import Path

from .api import DeploymentSpec, ResourceSlot
from .client import ControlClient
from .engine.vllm import VLLMServiceDriver
from .resource import StaticResourceProvider
from .runtime import Controller, Reconciler
from .server import make_server


def main() -> None:
    parser = argparse.ArgumentParser(description="ElasticRollout inference control plane")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--config", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--ray-address", default="auto")
    serve.add_argument("--shutdown-timeout", type=float, default=900)
    for command in ("status", "scale"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--endpoint", required=True)
        cmd.add_argument("--deployment", required=True)
        if command == "scale":
            cmd.add_argument("count", type=int)
            cmd.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    token = os.environ.get("ELASTIC_ROLLOUT_TOKEN")
    if args.command != "serve":
        client = ControlClient(args.endpoint, args.deployment, token=token)
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
    controller = Controller(resources, VLLMServiceDriver(RayProcessLauncher(args.ray_address)))
    for deployment in config["deployments"]:
        controller.ensure_deployment(DeploymentSpec.from_dict(deployment))
    server = make_server(controller, args.host, args.port, token)
    reconciler = Reconciler(controller)
    reconciler.start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        # Keep the API alive while draining, so clients can release/renew leases.
        import threading

        draining_server = threading.Thread(target=server.serve_forever, daemon=True)
        draining_server.start()
        try:
            reconciler.stop()
            controller.close(timeout=args.shutdown_timeout)
        finally:
            server.shutdown()
            server.server_close()
            draining_server.join()


if __name__ == "__main__":
    main()
