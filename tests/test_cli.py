"""CLI JSON results stay separate from diagnostic logs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

CLIENT_STUB = """
import json
import logging
import sys
from unittest.mock import Mock

from elastic_rollout import cli

expected = json.loads(sys.argv.pop(1))

def status():
    logging.getLogger("elastic_rollout.client").info("control request completed")
    return expected

client = Mock()
client.status.side_effect = status
cli.ControlClient = lambda *args, **kwargs: client
cli.main()
"""


@pytest.mark.parametrize("command", [["status"], ["scale", "4"], ["scale", "4", "--wait"]])
def test_cli_outputs_json_on_stdout_and_logs_on_stderr(command):
    expected = {"deployment_id": "test", "desired_replicas": 4, "ready_replicas": 4}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            CLIENT_STUB,
            json.dumps(expected),
            *command,
            "--endpoint",
            "http://unused",
            "--deployment",
            "test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert json.loads(result.stdout) == expected
    assert "control request completed" not in result.stdout
    assert "control request completed" in result.stderr
