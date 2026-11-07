import pytest
from helpers import py

from faultline.contract import DatasetProfile
from faultline.detectors.fit import DataFitDetector
from faultline.eval.labels import CompetitionLabel, Issue, NotebookLabel
from faultline.eval.notebooks import (
    LabeledNotebook,
    TypeScore,
    add_counts,
    competition_profile,
    match_lines,
    rule_findings,
    run_label_bench,
)

TEMPORAL = "data.leakage.temporal_split"

RANDOM_SPLIT = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv('../input/train.csv')
X, y = df.drop(columns=['sales']), df['sales']
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2)
RandomForestRegressor().fit(X_tr, y_tr).score(X_va, y_va)
"""
TIME_SPLIT = """
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv('../input/train.csv')
X, y = df.drop(columns=['sales']), df['sales']
cross_val_score(RandomForestRegressor(), X, y, cv=TimeSeriesSplit(5))
"""


def comp(**kw: object) -> CompetitionLabel:
    base: dict[str, object] = {"split_type": "temporal", "time_key": "date", "task": "regression",
                               "metric": "rmse", "reason": "r"}  # fmt: skip
    return CompetitionLabel.model_validate(base | kw)


def labeled(code: str, issues: list[Issue], competition: CompetitionLabel | None = None) -> LabeledNotebook:
    label = NotebookLabel(validation="random", issues=issues)
    return LabeledNotebook("comp", "nb/nb.py", py(code), label, competition or comp())


def test_match_lines_pairs_same_type_within_tolerance() -> None:
    tp, fp, fn, by_type = match_lines([(TEMPORAL, 10), ("data.fit.unstratified_split", 10)], [(TEMPORAL, 12)])
    assert (tp, fp, fn) == (1, 1, 0)
    assert by_type[TEMPORAL].tp == 1


def test_match_lines_far_or_other_type_is_not_a_match() -> None:
    tp, fp, fn, _ = match_lines([(TEMPORAL, 40)], [(TEMPORAL, 12)])
    assert (tp, fp, fn) == (0, 1, 1)


def test_add_counts_notebook_level() -> None:
    scores: dict[str, TypeScore] = {}
    add_counts(scores, {TEMPORAL, "x"}, {TEMPORAL, "y"})
    assert (scores[TEMPORAL].tp, scores["x"].fp, scores["y"].fn) == (1, 1, 1)
    assert scores[TEMPORAL].precision == 1.0 and scores["x"].recall is None


def test_competition_context_lets_rules_see_a_temporal_dataset() -> None:
    nb = labeled(RANDOM_SPLIT, [Issue(type=TEMPORAL, line=7, note="random split of daily sales")])
    assert TEMPORAL in {f.type for f in rule_findings(nb, context=True)}
    assert TEMPORAL not in {f.type for f in rule_findings(nb, context=False)}


def test_competition_context_accepts_a_time_split() -> None:
    nb = labeled(TIME_SPLIT, [])
    assert TEMPORAL not in {f.type for f in rule_findings(nb, context=True)}


def test_random_competition_gives_no_temporal_finding() -> None:
    nb = labeled(RANDOM_SPLIT, [], comp(split_type="random", time_key=None))
    assert TEMPORAL not in {f.type for f in rule_findings(nb, context=True)}


def test_profile_without_columns_does_not_flag_every_column() -> None:
    profile = competition_profile("comp", comp())
    findings = DataFitDetector([profile]).detect(py(RANDOM_SPLIT))
    assert not [f for f in findings if f.type == "data.fit.schema_mismatch"]


def test_profile_with_columns_still_checks_the_schema() -> None:
    profile = DatasetProfile.model_validate(
        {
            "name": "train.csv",
            "rows": 5000,
            "columns": [{"name": "store", "kind": "numeric", "missing_rate": 0, "distinct": 10}],
        }
    )
    findings = DataFitDetector([profile]).detect(py(RANDOM_SPLIT))
    assert [f for f in findings if f.type == "data.fit.schema_mismatch"]


def test_run_label_bench_scores_notebook_and_line_level() -> None:
    nb = labeled(RANDOM_SPLIT, [Issue(type=TEMPORAL, line=7, note="random split")])
    report = run_label_bench([nb], "train")
    assert report.by_type[TEMPORAL].tp == 1
    assert report.by_type_lines[TEMPORAL].tp == 1


def test_model_trains_and_reports_per_type() -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("sklearn")
    from faultline.eval.notebook_model import MODEL_TYPES, train_and_evaluate

    def many(code: str, issues: list[Issue], prefix: str) -> list[LabeledNotebook]:
        return [
            LabeledNotebook(
                f"{prefix}{i % 5}",
                f"nb{i}/nb.py",
                py(code),
                NotebookLabel(validation="random", issues=issues),
                comp(),
            )
            for i in range(10)
        ]

    bad = [Issue(type=TEMPORAL, line=7, note="random split")]
    train = many(RANDOM_SPLIT, bad, "a") + many(TIME_SPLIT, [], "b")
    report = train_and_evaluate(train, many(RANDOM_SPLIT, bad, "c"), "dev")
    assert set(report.eval_model) == set(MODEL_TYPES)
    assert report.eval_model[TEMPORAL].tp + report.eval_model[TEMPORAL].fn == 10
