from pathlib import Path

import numpy as np
import pandas as pd
from data_helpers import cancer

from faultline.contract import DatasetProfile, load_profiles, save_profile
from faultline.data.profile import profile_frame


def orders(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(n)],
            "customer_id": rng.integers(0, n // 8, n),  # ~8 orders per customer
            "date": pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M"),
            "amount": rng.lognormal(3, 1, n),
            "price_text": [f"{v:.2f}" for v in rng.uniform(1, 100, n)],
            "color": rng.choice(["red", "green", "blue"], n),
            "label": (rng.uniform(size=n) < 0.05).astype(int),
        }
    )


def test_time_order_entity_keys_and_ids() -> None:
    p = profile_frame(orders(), "orders.csv", target="label")
    assert p.time_columns == ["date"] and p.time_ordered
    assert p.group_keys == ["customer_id"]
    cols = {c.name: c for c in p.columns}
    assert cols["order_id"].identifier and not cols["customer_id"].identifier
    assert cols["date"].time_like and cols["date"].monotonic


def test_shuffled_rows_are_not_time_ordered() -> None:
    df = orders().sample(frac=1.0, random_state=0)
    p = profile_frame(df, "orders.csv", target="label")
    assert p.time_columns == ["date"] and not p.time_ordered


def test_numbers_stored_as_text_are_not_ids_or_dates() -> None:
    cols = {c.name: c for c in profile_frame(orders(), "orders.csv").columns}
    assert cols["price_text"].numeric_text
    assert not cols["price_text"].identifier and not cols["price_text"].time_like
    assert not cols["color"].time_like and not cols["color"].numeric_text


def test_class_balance_and_task() -> None:
    p = profile_frame(orders(), "orders.csv", target="label")
    assert p.task == "classification" and p.n_classes == 2
    assert p.minority_share is not None and p.minority_share < 0.1 and p.imbalanced
    real = profile_frame(cancer(), "cancer.csv", target="target")
    assert real.task == "classification" and not real.imbalanced  # ~37% malignant
    assert real.time_columns == [] and real.group_keys == [] and real.duplicate_rate == 0.0


def test_duplicate_rate_ignores_ids() -> None:
    df = orders()
    copies = df.iloc[:40].copy()
    copies["order_id"] = [f"copy{i}" for i in range(40)]  # same order, new id
    p = profile_frame(pd.concat([df, copies], ignore_index=True), "orders.csv")
    assert p.duplicate_rate == round(40 / 440, 4)


def test_profile_holds_no_cell_values_and_round_trips(tmp_path: Path) -> None:
    p = profile_frame(orders(), "orders.csv", target="label")
    save_profile(p, tmp_path / "orders.yaml")
    text = (tmp_path / "orders.yaml").read_text(encoding="utf-8")
    assert "o17" not in text and "2024-01" not in text and "green" not in text
    assert load_profiles(tmp_path) == [p]


def test_matches_by_file_name() -> None:
    p = DatasetProfile(name="data/raw/train.csv", rows=10)
    assert p.matches("/kaggle/input/titanic/train.csv")
    assert p.matches(r"C:\data\train.csv")
    assert not p.matches("test.csv")
