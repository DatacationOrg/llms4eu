"""Run pre-commit hooks as pytest tests. Integrates nicely with VSCode Test Explorer."""

import subprocess  # nosec: B404
import tomllib
from pathlib import Path

import pytest
import yaml

try:
    with Path("pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
except FileNotFoundError:
    pyproject: dict[str, dict[str, dict[str, str]]] = {}
allowed_fail_hooks = (
    pyproject.get("tool", {}).get("pytest_pre-commit", {}).get("allowed_fail_hooks", [])
)

with Path(".pre-commit-config.yaml").open() as f:
    config: dict[str, list[dict[str, list[dict[str, str]]]]] = yaml.safe_load(f)
HOOK_IDS = [hook["id"] for repo in config["repos"] for hook in repo["hooks"]]


@pytest.mark.parametrize("hook_id", HOOK_IDS)
def test_precommit_hook(hook_id: str) -> None:
    """Runs a specified pre-commit hook as a test using subprocess.

    Args:
        hook_id (str): The identifier of the pre-commit hook to run.

    Behavior:
        - Executes the pre-commit hook for all files.
        - If the hook fails and is in the allowed_fail_hooks list, marks the test as expected to fail.
        - Otherwise, fails the test and logs the hook's stdout and stderr output.
    """
    result = subprocess.run(  # noqa: S603
        ["uvx", "prek", "run", hook_id, "--all-files"],  # noqa: S607
        capture_output=True,
        check=False,
    )  # nosec: B603, B607

    # Use pytest's capsys to capture output, or log output on failure
    if result.returncode != 0:
        if hook_id in allowed_fail_hooks:
            pytest.xfail(reason=f"\t{hook_id} is allowed to fail\n")
        message = f"Pre-commit hook {hook_id} failed or made changes.\n"
        if result.stdout:
            message += f"STDOUT:\n{result.stdout.decode()}"
        if result.stderr:
            message += f"STDERR:\n{result.stderr.decode()}"
        pytest.fail(message, pytrace=False)
