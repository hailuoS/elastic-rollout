import asyncio
import json
import os

import pytest


@pytest.mark.cluster
@pytest.mark.parametrize("framework", ["core", "areal", "verl"])
def test_real_cluster_2_4_2_and_cleanup(framework):
    if os.environ.get("ELASTIC_E2E") != "1":
        pytest.skip(
            "requires explicit ELASTIC_E2E=1, Linux Ray/Ascend cluster and dedicated deployment"
        )
    if framework != "core":
        pytest.importorskip(framework)
    from examples.cluster_smoke import run

    result = asyncio.run(
        run(
            framework,
            os.environ["ELASTIC_ROLLOUT_ENDPOINT"],
            os.environ["ELASTIC_ROLLOUT_DEPLOYMENT"],
            json.loads(os.environ["ELASTIC_E2E_PROMPT_IDS"]),
        )
    )
    assert [step["ready"] for step in result] == [2, 4, 2]


@pytest.mark.cluster
@pytest.mark.parametrize("agent", ["math", "multi-turn", "tools"])
def test_real_areal_agent_proxy_2_4_2(agent, tmp_path):
    if os.environ.get("ELASTIC_AGENT_E2E") != "1":
        pytest.skip("requires ELASTIC_AGENT_E2E=1 and an installed AReaL/Ascend test deployment")
    pytest.importorskip("areal")
    from examples.areal_agent import run

    result = run(
        os.environ["ELASTIC_ROLLOUT_ENDPOINT"],
        os.environ["ELASTIC_ROLLOUT_DEPLOYMENT"],
        os.environ["ELASTIC_AGENT_TOKENIZER"],
        tmp_path / agent,
        agent=agent,
    )
    assert [step["replicas"] for step in result] == [2, 4, 2]
    assert all(step["workflow_workers"] == step["proxy_workers"] == 2 for step in result)


@pytest.mark.cluster
def test_real_partial_rollout_across_policy_update():
    if os.environ.get("ELASTIC_PARTIAL_E2E") != "1":
        pytest.skip("requires ELASTIC_PARTIAL_E2E=1, stock vLLM and a shared candidate snapshot")
    from elastic_rollout.client import ControlClient
    from examples.partial_rollout_smoke import run

    client = ControlClient(
        os.environ["ELASTIC_ROLLOUT_ENDPOINT"],
        os.environ["ELASTIC_ROLLOUT_DEPLOYMENT"],
        os.environ.get("ELASTIC_ROLLOUT_TOKEN"),
        timeout=180,
    )
    result = asyncio.run(
        run(
            client,
            json.loads(os.environ["ELASTIC_E2E_PROMPT_IDS"]),
            snapshot=os.environ["ELASTIC_PARTIAL_SNAPSHOT"],
            version=int(os.environ["ELASTIC_PARTIAL_VERSION"]),
        )
    )
    assert len(result["version_token_counts"]) == 2
