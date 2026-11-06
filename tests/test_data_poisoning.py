import numpy as np
import pandas as pd
from data_helpers import cancer, halves, mixed

from faultline.data.poisoning import PoisoningDetector, confident_label_issues, find_label_issues
from faultline.inject.data import DuplicateFlood, InjectedRows, LabelFlip
from faultline.schema import Finding


def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def of(findings: list[Finding], rule_id: str) -> list[Finding]:
    return [f for f in findings if f.evidence.rule_id == rule_id]


def test_clean_real_data_has_no_poisoning_findings() -> None:
    ref, cur = halves(cancer())
    assert PoisoningDetector().check(cur, dataset="cancer.csv", target="target", reference=ref) == []


def test_clean_real_data_without_reference_has_no_findings() -> None:
    assert PoisoningDetector().check(cancer(), dataset="cancer.csv", target="target") == []


def test_confident_learning_core() -> None:
    # thresholds: class 0 = mean(0.9, 0.8) = 0.85; class 1 = mean(0.9, 0.12, 0.8) ~ 0.61
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.88, 0.12], [0.2, 0.8]])
    y = np.array([0, 0, 1, 1, 1])  # row 3 is labeled 1 but clears class 0's threshold
    rows, believed, self_conf = confident_label_issues(proba, y)
    assert rows.tolist() == [3]
    assert believed.tolist() == [0]
    assert self_conf.tolist() == [0.12]


def test_find_label_issues_recovers_flips_in_separable_data() -> None:
    r = np.random.default_rng(0)
    x = np.vstack([r.normal(-3, 1, (200, 2)), r.normal(3, 1, (200, 2))])
    labels = np.array(["a"] * 200 + ["b"] * 200, dtype=object)
    flipped = r.choice(400, 20, replace=False)
    labels[flipped] = np.where(labels[flipped] == "a", "b", "a")
    found = set(find_label_issues(x, labels).rows.tolist())
    assert len(found & set(flipped.tolist())) >= 15
    assert len(found - set(flipped.tolist())) <= 5


def test_flipped_labels_raise_issue_rate_over_reference() -> None:
    ref, cur = halves(cancer())
    cur, fault = LabelFlip(rate=0.15).apply(cur, "target", rng())
    findings = PoisoningDetector().check(cur, dataset="cancer.csv", target="target", reference=ref)
    [hit] = of(findings, "poisoning.confident_learning")
    assert hit.type == "data.poisoning.label_flip"
    assert hit.location.columns == ["target"]
    assert int(hit.evidence.metrics["issues"]) >= len(fault.rows) // 2


def test_without_reference_only_a_high_issue_rate_is_reported_and_as_low() -> None:
    noisy, _ = LabelFlip(rate=0.25).apply(cancer(), "target", rng())
    [hit] = of(
        PoisoningDetector().check(noisy, dataset="cancer.csv", target="target"),
        "poisoning.confident_learning",
    )
    assert hit.severity == "low"


def test_injected_outlier_rows_are_found_against_reference() -> None:
    ref, cur = halves(cancer())
    cur, fault = InjectedRows(rate=0.1).apply(cur, "target", rng())
    [hit] = of(
        PoisoningDetector().check(cur, dataset="cancer.csv", target="target", reference=ref),
        "poisoning.outlier_rows",
    )
    assert hit.type == "data.poisoning.injected_rows"
    assert set(hit.location.columns) & set(fault.columns)
    assert int(hit.evidence.metrics["flagged"]) >= len(fault.rows) // 2


def test_outlier_check_needs_a_reference() -> None:
    cur, _ = InjectedRows(rate=0.1).apply(cancer(), "target", rng())
    assert not of(
        PoisoningDetector().check(cur, dataset="cancer.csv", target="target"), "poisoning.outlier_rows"
    )


def test_rows_breaking_a_correlation_are_flagged_inside_the_marginal_ranges() -> None:
    # each value is ordinary on its own; only the pair is impossible (what the distance score adds)
    r = np.random.default_rng(0)
    a = r.normal(size=1200)
    df = pd.DataFrame({"a": a, "b": a + r.normal(scale=0.1, size=1200), "c": r.normal(size=1200)})
    ref, cur = df.iloc[:600].reset_index(drop=True), df.iloc[600:].reset_index(drop=True)
    cur.loc[:29, "b"] = -cur.loc[:29, "a"]  # within b's range, opposite to a
    [hit] = of(PoisoningDetector().check(cur, dataset="pairs.csv", reference=ref), "poisoning.outlier_rows")
    assert int(hit.evidence.metrics["flagged"]) >= 15


def test_heavy_tailed_clean_data_has_no_outlier_finding() -> None:
    r = np.random.default_rng(1)
    size = r.lognormal(0, 1, 1200)
    df = pd.DataFrame({"size": size, "area": size**2 * r.lognormal(0, 0.1, 1200), "x": r.normal(size=1200)})
    ref, cur = df.iloc[:600].reset_index(drop=True), df.iloc[600:].reset_index(drop=True)
    assert not of(
        PoisoningDetector().check(cur, dataset="tails.csv", reference=ref), "poisoning.outlier_rows"
    )


def test_duplicate_flood_is_found() -> None:
    ref, cur = halves(cancer())
    cur, fault = DuplicateFlood().apply(cur, "target", rng())
    [hit] = of(
        PoisoningDetector().check(cur, dataset="cancer.csv", target="target", reference=ref),
        "poisoning.duplicate_flood",
    )
    assert hit.type == "data.poisoning.injected_rows"
    assert hit.evidence.metrics["flagged"] == len(fault.rows)


def test_repeats_the_reference_also_has_are_not_a_flood() -> None:
    # 8 copies of one row in both datasets: natural for this source, so not reported
    ref, cur = mixed(seed=1), mixed(seed=2)
    row = ref.iloc[[0]]
    ref = pd.concat([ref, *[row] * 8], ignore_index=True)
    cur = pd.concat([cur, *[row] * 8], ignore_index=True)
    assert of(PoisoningDetector().check(cur, dataset="orders.csv"), "poisoning.duplicate_flood")
    assert not of(
        PoisoningDetector().check(cur, dataset="orders.csv", reference=ref), "poisoning.duplicate_flood"
    )
