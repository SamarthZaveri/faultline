import tomllib
from pathlib import Path

import faultline

ROOT = Path(__file__).parents[1]


def test_package_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert faultline.__version__ == pyproject["project"]["version"]


def test_mypy_targets_minimum_supported_python() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["requires-python"] == ">=3.11"
    assert pyproject["tool"]["mypy"]["python_version"] == "3.11"


def test_dvp_fixture_excluded_from_lint() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "fixtures/dvp" in pyproject["tool"]["ruff"]["extend-exclude"]


def test_env_example_has_every_compose_variable() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()
    env_keys = {
        line.split("=", 1)[0]
        for line in (ROOT / ".env.example").read_text().splitlines()
        if line and not line.startswith("#")
    }
    for key in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
        assert f"${{{key}" in compose
        assert key in env_keys
