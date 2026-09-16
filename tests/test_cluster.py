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
