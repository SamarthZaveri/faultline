import json
from pathlib import Path

import pytest

from faultline.cli import main
from faultline.scan import scan


def test_scan_combines_graph_and_findings(tiny_repo: Path) -> None:
    result = scan(tiny_repo)
    assert result.files_scanned == 5
    assert {f.type for f in result.findings} == {"pipeline.dependency.unpinned"}
    node_ids = {n.id for n in result.graph.nodes}
    assert all(f.location.node_id in node_ids for f in result.findings)


def test_cli_scan_writes_json(tiny_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "scan.json"
    assert main(["scan", str(tiny_repo), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert {"graph", "findings", "files_scanned"} <= data.keys()
    assert "pipeline.dependency.unpinned" in capsys.readouterr().out


def test_cli_scan_fail_on(tiny_repo: Path) -> None:
    assert main(["scan", str(tiny_repo), "--fail-on", "low"]) == 1
    assert main(["scan", str(tiny_repo), "--fail-on", "high"]) == 0
