import numpy as np
import pandas as pd
import pytest
from data_helpers import mixed

from faultline.inject.data import (
    DRIFT_INJECTORS,
    POISONING_INJECTORS,
    DataInjector,
    InjectedRows,
    LabelFlip,
    leak_duplicates,
)
from faultline.schema import FINDING_TYPES


def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.mark.parametrize("injector", [*DRIFT_INJECTORS, *POISONING_INJECTORS], ids=lambda i: i.name)
def test_injector_changes_a_copy_and_reports_a_known_type(injector: DataInjector) -> None:
    data = mixed(300)
    before = data.copy()
    assert injector.applicable(data, "label")
    out, fault = injector.apply(data, "label", rng())
    pd.testing.assert_frame_equal(data, before)
    assert fault.expected_type in FINDING_TYPES
    assert all(0 <= r < len(out) for r in fault.rows)
    assert not out.equals(data)


def test_label_flip_changes_exactly_the_reported_rows() -> None:
    data = mixed(300)
    out, fault = LabelFlip(rate=0.1).apply(data, "label", rng())
    changed = np.flatnonzero((out["label"] != data["label"]).to_numpy())
    assert changed.tolist() == list(fault.rows)
    assert len(fault.rows) == 30


def test_injected_rows_are_appended_far_out_on_the_reported_columns() -> None:
    data = mixed(300)
    out, fault = InjectedRows().apply(data, "label", rng())
    assert len(out) == len(data) + len(fault.rows)
    assert len(fault.columns) == 3 and "label" not in fault.columns
    for c in fault.columns:
        assert (out.loc[list(fault.rows), c] - data[c].mean()).min() > 2 * data[c].std()


def test_leak_duplicates_copies_training_rows_into_test() -> None:
    train, test = mixed(200, seed=1), mixed(200, seed=2)
    out, fault = leak_duplicates(train, test, rate=0.1, rng=rng())
    assert len(out) == len(test)
    train_rows = set(map(tuple, train.astype(str).to_numpy()))
    copied = out.iloc[list(fault.rows)].astype(str).to_numpy()
    assert len(copied) == 20 and all(tuple(r) in train_rows for r in copied)


def test_injectors_that_need_a_label_say_so() -> None:
    data = mixed(300).drop(columns=["label"])
    assert not LabelFlip().applicable(data, None)
    with pytest.raises(ValueError, match="target"):
        LabelFlip().apply(data, None, rng())
