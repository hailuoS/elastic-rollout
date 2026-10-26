import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from elastic_rollout.generation import GenerationClient, parse_completion


@pytest.fixture
def inference_server():
    requests = []
    aborted = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            nonlocal aborted
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            # First call exercises full-context resume; second finishes.
            reason = "stop" if aborted else "abort"
            aborted = True
            body = json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": reason,
                            "logprobs": {"tokens": ["token_id:42"], "token_logprobs": [-0.1]},
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_direct_generation_resume_versions_and_release(http_runtime, inference_server):
    client, runtime = http_runtime
    endpoint, requests = inference_server
    manager = runtime[1]
    for replica in manager.traffic.replicas.values():
        replica.endpoint = endpoint
        replica.serving_policy_version = 7
    generator = GenerationClient(client, "policy")
    params = {"max_tokens": 3, "temperature": 0}
    result = asyncio.run(generator.generate("trajectory", [1, 2], params))
    assert result.token_ids == [42, 42]
    assert result.versions == [7, 7]
    assert requests[1]["prompt"] == [1, 2, 42]
    assert requests[1]["max_tokens"] == 2
    assert params["max_tokens"] == 3
    assert manager.status()["inflight"] == 0


@pytest.mark.parametrize(
    "tokens,probs", [(["text"], [-1.0]), (["token:1"], []), (["token:1"], [None])]
)
def test_malformed_token_outputs_rejected(tokens, probs):
    with pytest.raises(ValueError):
        parse_completion(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "logprobs": {"tokens": tokens, "token_logprobs": probs},
                    }
                ]
            }
        )
