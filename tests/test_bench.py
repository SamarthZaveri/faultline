import shutil
from pathlib import Path

import pytest
import yaml

from faultline.eval.bench import run_bench
from faultline.eval.corpus import Manifest, load_manifest

ROOT = Path(__file__).parents[1]


def write_manifest(tmp_path: Path, tiny_repo: Path) -> Path:
    repos = tmp_path / "repos"
    for name in ("a", "b"):
        shutil.copytree(tiny_repo, repos / name)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "repos": [
                    {"name": "a", "url": "https://x/a.git", "commit": "0" * 40, "split": "train"},
                    {"name": "b", "url": "https://x/b.git", "commit": "1" * 40, "split": "test"},
                ]
            }
        )
    )
    return manifest


def test_committed_manifest_is_valid_and_split_by_repo() -> None:
    m = load_manifest(ROOT / "corpus" / "manifest.yaml")
    splits = {r.split for r in m.repos}
    assert splits == {"train", "dev", "test"}
    assert len({r.name for r in m.repos}) == len(m.repos)


def test_manifest_rejects_duplicate_names() -> None:
    entry = {"name": "a", "url": "https://x/a.git", "commit": "0" * 40, "split": "train"}
    with pytest.raises(ValueError, match="duplicate"):
        Manifest.model_validate({"repos": [entry, entry]})


def test_bench_on_train_split(tmp_path: Path, tiny_repo: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path, tiny_repo))
    report = run_bench(manifest, "train", tmp_path / "repos", per_mutator=2, seed=0)
    assert report.repos == ["a"]  # only the train split
    assert report.injections, "expected applicable mutations on tiny repo"
    assert all(i.repo == "a" for i in report.injections)
    assert set(report.recall_by_type) == {i.expected_type for i in report.injections}
    assert all(0.0 <= r <= 1.0 for r in report.recall_by_type.values())
    assert report.clean_findings_by_type.get("pipeline.dependency.unpinned") == 1


def test_bench_is_deterministic(tmp_path: Path, tiny_repo: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path, tiny_repo))
    a = run_bench(manifest, "train", tmp_path / "repos", per_mutator=2, seed=3)
    b = run_bench(manifest, "train", tmp_path / "repos", per_mutator=2, seed=3)
    assert a.injections == b.injections


def test_test_split_requires_final_flag(tmp_path: Path, tiny_repo: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path, tiny_repo))
    with pytest.raises(PermissionError, match="held out"):
        run_bench(manifest, "test", tmp_path / "repos")
    report = run_bench(manifest, "test", tmp_path / "repos", final=True)
    assert report.repos == ["b"]


def test_missing_repo_is_reported_not_fatal(tmp_path: Path, tiny_repo: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path, tiny_repo))
    shutil.rmtree(tmp_path / "repos" / "a")
    report = run_bench(manifest, "train", tmp_path / "repos")
    assert report.repos == [] and report.missing == ["a"]


def test_triage_labels_give_precision(tmp_path: Path, tiny_repo: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path, tiny_repo))
    first = run_bench(manifest, "train", tmp_path / "repos")
    [fid] = [f.id for f in first.clean_findings]
    triage = tmp_path / "triage"
    triage.mkdir()
    (triage / "a.yaml").write_text(yaml.safe_dump({"labels": {fid: "tp"}}))
    report = run_bench(manifest, "train", tmp_path / "repos", triage_dir=triage)
    assert report.triaged == 1 and report.precision_on_real == 1.0


def test_identical_files_counted_once(tmp_path: Path, tiny_repo: Path) -> None:
    # e.g. translated copies of the same notebook must not inflate counts
    (tiny_repo / "translations" / "fr").mkdir(parents=True)
    shutil.copy(tiny_repo / "requirements.txt", tiny_repo / "translations" / "fr" / "requirements.txt")
    manifest = load_manifest(write_manifest(tmp_path, tiny_repo))
    report = run_bench(manifest, "train", tmp_path / "repos")
    assert report.clean_findings_by_type == {"pipeline.dependency.unpinned": 1}
    assert report.duplicates_skipped == 1


def test_excluded_files_are_skipped(tmp_path: Path, tiny_repo: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path, tiny_repo))
    report = run_bench(manifest, "train", tmp_path / "repos", excluded={"a/requirements.txt"})
    assert report.clean_findings_by_type == {}
    assert report.duplicates_skipped == 1
