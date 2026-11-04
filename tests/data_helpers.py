import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def cancer() -> pd.DataFrame:
    """Real data bundled with scikit-learn (no download): 569 tumors, 30 measurements, `target`."""
    frame: pd.DataFrame = load_breast_cancer(as_frame=True).frame
    return frame


def halves(df: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = np.random.default_rng(seed).permutation(len(df))
    mid = len(df) // 2
    return df.iloc[order[:mid]].reset_index(drop=True), df.iloc[order[mid:]].reset_index(drop=True)


def mixed(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Numbers, a category, an id column and a label that depends on x1."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    return pd.DataFrame(
        {
            "amount": rng.lognormal(3, 1, n),
            "x1": x1,
            "x2": x1 + rng.normal(scale=0.3, size=n),
            "color": rng.choice(["red", "green", "blue"], n, p=[0.5, 0.3, 0.2]),
            "order_id": [f"o{seed}-{i}" for i in range(n)],
            "label": (x1 + rng.normal(scale=0.5, size=n) > 0).astype(int),
        }
    )
