from pathlib import Path

import pytest

from faultline.eval.baseline import freeze
from faultline.eval.bench import BenchReport, Injection
from faultline.eval.yang import Counts, YangReport
from faultline.schema import Evidence, Finding, Location


def write_reports(reports: Path) -> None:
    reports.mkdir()
    finding = Finding(
        type="data.leakage.preprocessing_before_split",
        title="t",
        severity="medium",
        confidence=0.8,
        location=Location(file="nb.ipynb", line_start=3),
        evidence=Evidence(rule_id="r", explanation="e", snippet="X = StandardScaler().fit_transform(X)"),
        detector="leakage@0.2",
    )
    for split, stamp in (
        ("train", "20261103-100000"),
        ("train", "20261103-120000"),
        ("dev", "20261103-120001"),
    ):
        report = BenchReport(
            split=split,  # type: ignore[arg-type]
            seed=0,
            repos=["a", "b"],
            files_scanned=10,
            loc_scanned=1000,
            injections=[
                Injection(
                    repo="a",
                    file="x.py",
                    mutator="m",
                    expected_type="pipeline.injection.sql",
                    line_start=1,
                    line_end=2,
                    detected=True,
                    collateral=0,
                )
            ],
            recall_by_type={"pipeline.injection.sql": 1.0 if stamp.endswith("120000") else 0.5},
            recall_overall=1.0,
            clean_findings=[finding],
            clean_findings_by_type={"data.leakage.preprocessing_before_split": 1},
        )
        (reports / f"bench_{split}_{stamp}.json").write_text(report.model_dump_json())
    yang = YangReport(
        split="dev",
        labeled=29,
        ours={
            "pre": Counts(tp=6, fp=1, fn=5),
            "overlap": Counts(tp=1, fn=3),
            "multi": Counts(tp=7, fp=1, fn=9),
        },
        baseline={"pre": Counts(tp=8, fn=3), "overlap": Counts(tp=4), "multi": Counts(tp=10, fn=6)},
        misses={},
    )
    (reports / "yang2022_dev_20261103-120002.json").write_text(yang.model_dump_json())


def test_freeze_copies_latest_reports_and_writes_summary(tmp_path: Path) -> None:
    write_reports(tmp_path / "reports")
    out = freeze("rules-v0.2", tmp_path / "reports", tmp_path / "results", commit="abc1234")
    assert {p.name for p in out.iterdir()} == {
        "SUMMARY.md",
        "bench_train.json",
        "bench_dev.json",
        "yang2022_dev.json",
    }
    train = BenchReport.model_validate_json((out / "bench_train.json").read_text())
    assert train.recall_by_type["pipeline.injection.sql"] == 1.0  # the newest train report, not the older one
    assert all(f.evidence.snippet is None for f in train.clean_findings)  # no third-party code copied in
    summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "rules-v0.2" in summary and "abc1234" in summary
    assert "| pre | 85.7% | 54.5% |" in summary  # faultline precision / recall on Yang dev
    assert "optimistic" in summary  # caveats travel with the numbers


def test_freeze_refuses_to_overwrite(tmp_path: Path) -> None:
    write_reports(tmp_path / "reports")
    freeze("v1", tmp_path / "reports", tmp_path / "results", commit="x")
    with pytest.raises(FileExistsError):
        freeze("v1", tmp_path / "reports", tmp_path / "results", commit="x")


def test_freeze_needs_reports(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir()
    with pytest.raises(FileNotFoundError, match="bench_train"):
        freeze("v1", tmp_path / "reports", tmp_path / "results", commit="x")
