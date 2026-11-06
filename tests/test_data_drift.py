import numpy as np
from data_helpers import cancer, halves, mixed

from faultline.data.drift import DriftDetector, scale_ratio
from faultline.inject.data import (
    ConceptShift,
    CovariateShift,
    LabelPrior,
    MissingSpike,
    NewCategory,
    SchemaDrop,
    UnitChange,
)
from faultline.schema import Finding


def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def of(findings: list[Finding], type_: str, rule_id: str | None = None) -> list[Finding]:
    return [f for f in findings if f.type == type_ and (rule_id is None or f.evidence.rule_id == rule_id)]


def test_random_halves_of_real_data_do_not_drift() -> None:
    ref, cur = halves(cancer())
    assert DriftDetector().compare(ref, cur, dataset="cancer.csv", target="target") == []


def test_two_samples_of_one_distribution_do_not_drift() -> None:
    findings = DriftDetector().compare(mixed(seed=1), mixed(seed=2), dataset="orders.csv", target="label")
    assert findings == []


def test_mean_shift_is_covariate_drift_on_that_column_only() -> None:
    ref, cur = halves(cancer())
    cur, fault = CovariateShift().apply(cur, "target", rng())
    findings = DriftDetector().compare(ref, cur, dataset="cancer.csv", target="target")
    hits = of(findings, "data.drift.covariate", "drift.psi_ks")
    assert [f.location.columns for f in hits] == [list(fault.columns)]
    assert hits[0].location.node_id == "data:cancer.csv"


def test_unit_change_is_named_as_such() -> None:
    ref, cur = halves(cancer())
    cur, fault = UnitChange().apply(cur, "target", rng())
    [hit] = of(
        DriftDetector().compare(ref, cur, dataset="cancer.csv"), "data.drift.covariate", "drift.psi_ks"
    )
    assert hit.location.columns == list(fault.columns)
    assert hit.evidence.metrics["scale_ratio"] == 1000.0
    assert "unit change" in hit.evidence.explanation


def test_scale_ratio_only_for_powers_of_ten() -> None:
    base = np.random.default_rng(0).lognormal(2, 0.5, 500)
    assert scale_ratio(base, base * 1000) == 1000.0
    assert scale_ratio(base, base / 100) == 0.01
    assert scale_ratio(base, base * 1.3) is None


def test_missing_spike() -> None:
    cur, fault = MissingSpike().apply(mixed(seed=2), "label", rng())
    findings = DriftDetector().compare(mixed(seed=1), cur, dataset="orders.csv", target="label")
    [hit] = of(findings, "data.drift.missing_spike")
    assert hit.location.columns == list(fault.columns)
    assert hit.severity == "high"  # 30% of the column blanked


def test_new_category() -> None:
    cur, fault = NewCategory().apply(mixed(seed=2), "label", rng())
    findings = DriftDetector().compare(mixed(seed=1), cur, dataset="orders.csv", target="label")
    [hit] = of(findings, "data.drift.new_category")
    assert hit.location.columns == list(fault.columns) == ["color"]
    assert "unseen_" not in hit.title + hit.evidence.explanation  # raw values never copied into findings


def test_dropped_column_is_schema_drift() -> None:
    cur, fault = SchemaDrop().apply(mixed(seed=2), "label", rng())
    findings = DriftDetector().compare(mixed(seed=1), cur, dataset="orders.csv", target="label")
    [hit] = of(findings, "data.drift.schema")
    assert hit.location.columns == list(fault.columns)
    assert hit.evidence.rule_id == "drift.schema.missing_column"


def test_numbers_read_as_text_is_schema_drift() -> None:
    cur = mixed(seed=2)
    cur["x1"] = cur["x1"].map(lambda v: f"{v:.3f}")
    [hit] = of(DriftDetector().compare(mixed(seed=1), cur, dataset="orders.csv"), "data.drift.schema")
    assert hit.location.columns == ["x1"]
    assert hit.evidence.rule_id == "drift.schema.type_change"


def test_class_balance_change_is_label_prior_drift() -> None:
    cur, _ = LabelPrior().apply(mixed(seed=2), "label", rng())
    findings = DriftDetector().compare(mixed(seed=1), cur, dataset="orders.csv", target="label")
    assert [f.location.columns for f in of(findings, "data.drift.label_prior")] == [["label"]]


def test_changed_relationship_is_concept_drift() -> None:
    cur, _ = ConceptShift().apply(mixed(seed=2), "label", rng())
    findings = DriftDetector().compare(mixed(seed=1), cur, dataset="orders.csv", target="label")
    assert of(findings, "data.drift.concept")


def test_joint_shift_invisible_per_column_is_found_by_classifier_test() -> None:
    # x2 follows -x1 instead of x1: both marginals are unchanged, the pair is not.
    cur = mixed(seed=2)
    cur["x2"] = -cur["x1"] + np.random.default_rng(3).normal(scale=0.3, size=len(cur))
    findings = DriftDetector().compare(mixed(seed=1), cur, dataset="orders.csv", target="label")
    assert not of(findings, "data.drift.covariate", "drift.psi_ks")
    [hit] = of(findings, "data.drift.covariate", "drift.c2st")
    assert {"x1", "x2"} & set(hit.location.columns)


def test_id_columns_are_not_tested_for_drift() -> None:
    ref, cur = mixed(seed=1), mixed(seed=2)  # order ids differ completely between the two
    findings = DriftDetector().compare(ref, cur, dataset="orders.csv", target="label")
    assert all("order_id" not in f.location.columns for f in findings)


def test_too_few_rows_is_not_judged() -> None:
    cur, _ = CovariateShift().apply(mixed(30, seed=2), "label", rng())
    assert DriftDetector().compare(mixed(30, seed=1), cur, dataset="orders.csv") == []
