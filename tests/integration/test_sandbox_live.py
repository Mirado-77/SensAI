from pathlib import Path

import pytest

from sensai.core.config import load_config
from sensai.tools.sandbox.docker import docker_run_args, docker_unavailable_reason, run_container

pytestmark = pytest.mark.feature("T2")


@pytest.mark.integration
async def test_demo_repo_tests_run_in_the_real_sandbox():
    sandbox = load_config().sandbox
    if await docker_unavailable_reason(sandbox.image):
        pytest.skip("Docker ou l'image du bac à sable est indisponible.")
    workspace = Path(__file__).resolve().parents[2] / "demo-repo"
    args = docker_run_args(sandbox, "sensai-it-demo", workspace, "python -m pytest -q")
    result = await run_container(args, "sensai-it-demo", sandbox.timeout_s)
    assert result.exit_code == 1
    assert b"test_completion_rate_empty_list" in result.output
