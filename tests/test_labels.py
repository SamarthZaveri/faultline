from pathlib import Path

import pytest
from pydantic import ValidationError

from faultline.eval.corpus import load_manifest
from faultline.eval.labels import CompetitionLabel, NotebookLabel, load_labels, load_notebook_labels


def test_every_kaggle_competition_in_the_manifest_has_a_label() -> None:
    labels = load_labels()
    kaggle = {e.competition for e in load_manifest().repos if e.source == "kaggle"}
    assert kaggle, "manifest has no Kaggle entries"
    assert not kaggle - set(labels), f"unlabeled: {sorted(kaggle - set(labels))}"


def test_label_file_has_no_competitions_missing_from_the_list() -> None:
    listed = Path(__file__).resolve().parents[1] / "corpus" / "kaggle_competitions.txt"
    names = {s.strip() for s in listed.read_text(encoding="utf-8").splitlines()}
    names = {s for s in names if s and not s.startswith("#")}
    assert set(load_labels()) <= names


def label(**kw: object) -> CompetitionLabel:
    base: dict[str, object] = {"split_type": "random", "task": "binary", "metric": "auc", "reason": "r"}
    return CompetitionLabel.model_validate(base | kw)


def test_group_split_needs_a_group_key() -> None:
    with pytest.raises(ValidationError):
        label(split_type="group")
    assert label(split_type="group", group_key="customer_ID").group_key == "customer_ID"


def test_temporal_split_needs_a_time_key() -> None:
    with pytest.raises(ValidationError):
        label(split_type="temporal")
    assert label(split_type="temporal", time_key="date").time_key == "date"


def test_random_split_has_no_keys() -> None:
    with pytest.raises(ValidationError):
        label(group_key="id")
    assert label().stratify is False


def test_stratify_only_for_classification() -> None:
    with pytest.raises(ValidationError):
        label(task="regression", metric="rmse", stratify=True)
    assert label(stratify=True).stratify


def test_notebook_label_files_match_the_manifest_and_competition_labels() -> None:
    files = load_notebook_labels()
    assert files, "no notebook labels yet"
    competitions = load_labels()
    kernels = {
        e.competition: {k.dirname for k in e.kernels} for e in load_manifest().repos if e.source == "kaggle"
    }
    for comp, f in files.items():
        assert comp in competitions, comp
        unknown = {path.split("/")[0] for path in f.notebooks} - kernels.get(comp, set())
        assert not unknown, f"{comp}: notebooks not in the manifest: {sorted(unknown)}"


def test_relevant_notebook_label_needs_validation() -> None:
    with pytest.raises(ValidationError):
        NotebookLabel.model_validate({"issues": []})
    assert NotebookLabel.model_validate({"relevant": False}).issues == []


def test_issue_types_must_be_known() -> None:
    issue = {"type": "data.leakage.made_up", "line": 3, "note": "n"}
    with pytest.raises(ValidationError):
        NotebookLabel.model_validate({"validation": "random", "issues": [issue]})


def test_load_labels_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "labels.yaml"
    path.write_text(
        "competitions:\n  toy:\n    split_type: random\n    task: regression\n    metric: rmse\n"
        "    reason: test rows drawn at random\n",
        encoding="utf-8",
    )
    assert load_labels(path)["toy"].metric == "rmse"
