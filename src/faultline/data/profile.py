"""Profile a dataset (pandas) into a DatasetProfile: column kinds, ids, entity keys, time order,
class balance and duplicates. Aggregates only, never cell values."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from faultline.contract import ColumnProfile, DatasetProfile, looks_like_id_name
from faultline.data.frame import as_str_columns, column_kind, is_identifier, row_hashes

MIN_TIME_ROWS = 20
ORDERED_SHARE = 0.99  # consecutive rows in time order
PARSE_SHARE = 0.95  # values that must parse as numbers / dates


def _numeric_text(s: pd.Series) -> bool:
    if column_kind(s) != "categorical" or pd.api.types.is_bool_dtype(s):
        return False
    sample = s.dropna().astype("string").str.strip().head(1000)
    return bool(len(sample)) and float(pd.to_numeric(sample, errors="coerce").notna().mean()) >= PARSE_SHARE


def _as_dates(s: pd.Series) -> pd.Series | None:
    """The column as datetimes if it is one (or is text that parses as dates), else None."""
    kind = column_kind(s)
    if kind == "datetime":
        return s
    if kind != "categorical" or pd.api.types.is_bool_dtype(s) or _numeric_text(s):
        return None  # "12" parses as a day of the month; numbers are not dates
    values = s.dropna().astype("string")
    if values.empty:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        head = pd.to_datetime(values.head(500), errors="coerce", format="mixed")
        if float(head.notna().mean()) < PARSE_SHARE:
            return None
        return pd.to_datetime(s.astype("string"), errors="coerce", format="mixed")


def _in_order(dates: pd.Series) -> bool:
    values = dates.dropna()
    if len(values) < MIN_TIME_ROWS or values.nunique() < 10:
        return False
    seconds = (values - values.min()).dt.total_seconds().to_numpy(dtype=float)
    return float((np.diff(seconds) >= 0).mean()) >= ORDERED_SHARE


def profile_frame(df: pd.DataFrame, name: str, target: str | None = None) -> DatasetProfile:
    df = as_str_columns(df)
    n = len(df)
    columns: list[ColumnProfile] = []
    time_columns: list[str] = []
    ordered = False
    group_keys: list[str] = []
    for c in df.columns:
        s = df[c]
        kind = column_kind(s)
        numeric_text = _numeric_text(s)
        distinct = int(s.nunique(dropna=True))
        present = int(s.notna().sum())
        identifier = is_identifier(s) and (not numeric_text or looks_like_id_name(c))
        dates = _as_dates(s) if c != target else None
        monotonic = dates is not None and _in_order(dates)
        if dates is not None and distinct >= 10:
            time_columns.append(c)
            ordered = ordered or monotonic
        if (
            c != target
            and looks_like_id_name(c)
            and kind in ("numeric", "categorical")
            and 2 <= distinct <= 0.8 * present
            and present / max(distinct, 1) >= 1.5
        ):
            group_keys.append(c)
        columns.append(
            ColumnProfile(
                name=c,
                kind=kind,
                missing_rate=round(1 - present / n, 4) if n else 0.0,
                distinct=distinct,
                identifier=identifier,
                numeric_text=numeric_text,
                time_like=dates is not None,
                monotonic=monotonic,
            )
        )
    profile = DatasetProfile(
        name=name,
        rows=n,
        columns=columns,
        target=target if target in df.columns else None,
        group_keys=group_keys,
        time_columns=time_columns,
        time_ordered=ordered,
    )
    content = [c.name for c in columns if not c.identifier]
    if n and content:
        profile.duplicate_rate = round(float(pd.Series(row_hashes(df, content)).duplicated().mean()), 4)
    if profile.target is not None:
        y = df[profile.target].dropna()
        if column_kind(df[profile.target]) == "numeric" and y.nunique() > 20:
            profile.task = "regression"
        elif y.nunique() >= 2:
            profile.task = "classification"
            profile.n_classes = int(y.nunique())
            profile.minority_share = round(float(y.value_counts(normalize=True).min()), 4)
    return profile
