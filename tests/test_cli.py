import json
from pathlib import Path

import pandas as pd
import pytest
from data_helpers import mixed

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


def test_cli_data_compare_reports_drift(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ref, cur = mixed(500, seed=1), mixed(500, seed=2)
    cur["amount"] = cur["amount"] * 1000
    ref.to_csv(tmp_path / "ref.csv", index=False)
    cur.to_csv(tmp_path / "cur.csv", index=False)
    out = tmp_path / "findings.json"
    args = ["data", "compare", str(tmp_path / "ref.csv"), str(tmp_path / "cur.csv"), "--target", "label"]
    assert main([*args, "--dataset", "orders.csv", "--out", str(out)]) == 0
    assert "data.drift.covariate" in capsys.readouterr().out
    findings = json.loads(out.read_text())
    assert any(f["location"]["columns"] == ["amount"] for f in findings)
    assert all(f["location"]["file"] == "orders.csv" for f in findings)


def test_cli_data_overlap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    train = mixed(300, seed=1)
    test = pd.concat([train.iloc[:30], mixed(270, seed=2)], ignore_index=True)  # 10% copied
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    assert main(["data", "overlap", str(tmp_path / "train.csv"), str(tmp_path / "test.csv")]) == 0
    assert "data.leakage.duplicate_across_split" in capsys.readouterr().out


def test_cli_scan_fail_on(tiny_repo: Path) -> None:
    assert main(["scan", str(tiny_repo), "--fail-on", "low"]) == 1
    assert main(["scan", str(tiny_repo), "--fail-on", "high"]) == 0
