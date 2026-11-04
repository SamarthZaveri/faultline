import numpy as np
from data_helpers import cancer, halves

from faultline.data.overlap import SplitOverlapDetector
from faultline.inject.data import leak_duplicates


def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_clean_split_has_no_overlap() -> None:
    train, test = halves(cancer())
    assert SplitOverlapDetector().check(train, test, dataset="cancer.csv", target="target") == []


def test_training_rows_copied_into_test_are_found() -> None:
    train, test = halves(cancer())
    test, fault = leak_duplicates(train, test, rate=0.1, rng=rng())
    [hit] = SplitOverlapDetector().check(train, test, dataset="cancer.csv", target="target")
    assert hit.type == "data.leakage.duplicate_across_split"
    assert hit.evidence.metrics["shared_rows"] == len(fault.rows)
    assert hit.severity == "high"


def test_id_columns_do_not_hide_copies() -> None:
    train, test = halves(cancer())
    train["row_id"] = [f"tr{i}" for i in range(len(train))]
    test["row_id"] = [f"te{i}" for i in range(len(test))]
    test, fault = leak_duplicates(train, test, rate=0.1, rng=rng())
    test["row_id"] = [f"new{i}" for i in range(len(test))]  # copies got fresh ids
    [hit] = SplitOverlapDetector().check(train, test, dataset="cancer.csv", target="target")
    assert hit.evidence.metrics["shared_rows"] == len(fault.rows)


def test_one_shared_row_is_below_the_rate_threshold() -> None:
    train, test = halves(cancer())
    test, fault = leak_duplicates(train, test, rate=0.001, rng=rng())
    assert len(fault.rows) == 1
    assert SplitOverlapDetector().check(train, test, dataset="cancer.csv", target="target") == []


def test_int_and_float_copies_hash_the_same() -> None:
    train, test = halves(cancer())
    test, fault = leak_duplicates(train, test, rate=0.1, rng=rng())
    test["target"] = test["target"].astype(float)  # e.g. re-read with a missing value elsewhere
    [hit] = SplitOverlapDetector().check(train, test, dataset="cancer.csv")  # target compared too
    assert hit.evidence.metrics["shared_rows"] == len(fault.rows)
