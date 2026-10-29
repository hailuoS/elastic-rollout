"""Control-only HTTP API backed by one authoritative Controller process."""

import hmac
import json
import logging
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from .api import DeploymentSpec
from .runtime import Controller
from .traffic import NoReadyReplica

logger = logging.getLogger(__name__)


def _require_fields(body: dict, *names: str) -> None:
    missing = [name for name in names if name not in body]
    if missing:
        raise ValueError(f"missing required request fields: {', '.join(missing)}")


class ControlHTTPServer(ThreadingHTTPServer):
    """Control server with capacity for concurrent route bursts."""

    request_queue_size = 256


def make_server(
    controller: Controller, host: str = "127.0.0.1", port: int = 8080, token: str | None = None
) -> ControlHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

        def _reply(self, code: int, data: dict) -> None:
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self) -> None:
            try:
                if token and not hmac.compare_digest(
                    self.headers.get("Authorization", ""), f"Bearer {token}"
                ):
                    self._reply(401, {"error": "unauthorized"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 65536:
                    self._reply(413, {"error": "control payload too large"})
                    return
                body = json.loads(self.rfile.read(length)) if length else {}
                if not isinstance(body, dict):
                    raise ValueError("JSON body must be an object")
                path = urlsplit(self.path).path
                if path == "/health" and self.command == "GET":
                    self._reply(200, {"status": "ok"})
                    return
                parts = [unquote(p) for p in path.strip("/").split("/")]
                if len(parts) < 3 or parts[:2] != ["v1", "deployments"]:
                    self._reply(404, {"error": "unknown control endpoint"})
                    return
                deployment_id = parts[2]
                tail = parts[3:]
                if not tail and self.command == "PUT":
                    _require_fields(body, "deployment_id", "replica")
                    spec = DeploymentSpec.from_dict(body)
                    if spec.deployment_id != deployment_id:
                        raise ValueError("deployment ID in URL and body differ")
                    result = controller.ensure_deployment(spec)
                else:
                    manager = controller.deployments[deployment_id]
                    if not tail and self.command == "GET":
                        result = manager.status()
                    elif tail == ["replicas"] and self.command == "GET":
                        with manager.traffic.lock:
                            result = {
                                "replicas": [
                                    asdict(r)
                                    | {"inflight": manager.traffic.inflight.get(r.replica_id, 0)}
                                    for r in manager.traffic.replicas.values()
                                ]
                            }
                    elif tail == ["desired"] and self.command == "PUT":
                        _require_fields(body, "desired_replicas")
                        result = controller.set_desired_replicas(
                            deployment_id, body["desired_replicas"]
                        )
                    elif tail == ["routes"] and self.command == "POST":
                        _require_fields(body, "routing_key")
                        result = asdict(manager.traffic.acquire(body["routing_key"]))
                    elif len(tail) == 2 and tail[0] == "leases" and self.command == "DELETE":
                        result = {"released": manager.traffic.release(tail[1])}
                    elif len(tail) == 2 and tail[0] == "leases" and self.command == "PUT":
                        result = asdict(manager.traffic.renew(tail[1]))
                    else:
                        self._reply(404, {"error": "unknown control endpoint"})
                        return
                self._reply(200, result)
            except NoReadyReplica as exc:
                self._reply(503, {"error": str(exc)})
            except KeyError as exc:
                self._reply(404, {"error": f"unknown deployment or lease: {exc}"})
            except (ValueError, TypeError) as exc:
                self._reply(400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass  # Leases returned to a disconnected client expire normally.
            except Exception:
                logger.exception("control request failed")
                self._reply(500, {"error": "internal control error; consult runtime logs"})

        do_GET = do_POST = do_PUT = do_DELETE = _handle

    server = ControlHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
