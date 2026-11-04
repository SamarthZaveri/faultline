"""Shared helpers for detectors that look at data (DataFrames) rather than code.

A data finding points at a dataset (`Location.file` is its path or name, `node_id` its graph node
as in discovery/graph.py) and the columns involved. It never carries raw cell values, only row
positions and aggregate statistics, so a finding can't leak PII (rule 2).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from faultline.schema import Evidence, Finding, Location, Severity, Tier

ColumnKind = Literal["numeric", "categorical", "datetime", "other"]
MAX_ROWS_SHOWN = 20
ID_NAME = re.compile(r"(^|_)(id|uuid|guid|index)$|^unnamed: ?0$", re.IGNORECASE)
CAMEL_ID = re.compile(r"[a-z]I[dD]$")  # PassengerId, userID (case-sensitive: not "paid", "valid")


def as_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df if all(isinstance(c, str) for c in df.columns) else df.rename(columns=str)


def column_kind(s: pd.Series) -> ColumnKind:
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if (
        isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
    ):
        return "categorical"
    return "other"


def looks_like_id_name(name: str) -> bool:
    return bool(ID_NAME.search(name) or CAMEL_ID.search(name))


def is_identifier(s: pd.Series) -> bool:
    """Row ids, hashes, free text: (nearly) every value distinct, so distributions mean nothing."""
    n = int(s.notna().sum())
    if n < 20:
        return False
    distinct = int(s.nunique(dropna=True))
    kind = column_kind(s)
    if kind == "categorical":
        return distinct > 0.9 * n
    return (
        kind == "numeric"
        and pd.api.types.is_integer_dtype(s)
        and distinct == n
        and looks_like_id_name(str(s.name))
    )


def feature_columns(frames: Sequence[pd.DataFrame], exclude: Sequence[str] = ()) -> list[str]:
    """Columns in every frame, of the same kind everywhere, that a model can use: numbers and
    categories, not ids or datetimes."""
    first, rest = frames[0], frames[1:]
    out = []
    for c in first.columns:
        if c in exclude or any(c not in f.columns for f in rest):
            continue
        kind = column_kind(first[c])
        if (
            kind in ("numeric", "categorical")
            and all(column_kind(f[c]) == kind for f in rest)
            and not is_identifier(first[c])
        ):
            out.append(c)
    return out


def encode(frames: Sequence[pd.DataFrame], columns: Sequence[str]) -> list[np.ndarray]:
    """Float matrices with one shared encoding: numbers as they are, categories as codes learned
    from all frames together (so the same value gets the same code everywhere). Missing stays NaN."""
    blocks: list[list[np.ndarray]] = [[] for _ in frames]
    for c in columns:
        if column_kind(frames[0][c]) == "numeric":
            for block, f in zip(blocks, frames, strict=True):
                block.append(pd.to_numeric(f[c], errors="coerce").to_numpy(dtype=float, na_value=np.nan))
            continue
        values = [f[c].astype("string") for f in frames]
        vocab = sorted({value for v in values for value in v.dropna()})
        codes = {v: float(i) for i, v in enumerate(vocab)}
        for block, v in zip(blocks, values, strict=True):
            block.append(v.map(codes).to_numpy(dtype=float, na_value=np.nan))
    return [
        np.column_stack(block) if block else np.empty((len(f), 0))
        for block, f in zip(blocks, frames, strict=True)
    ]


def row_hashes(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """One content hash per row. Numbers are hashed as float64 and everything else as text, so
    the same row hashes the same whether a column was read as int, float or object."""
    norm = {
        c: (
            pd.to_numeric(df[c], errors="coerce").astype("float64")
            if column_kind(df[c]) == "numeric"
            else df[c].astype("string")
        )
        for c in columns
    }
    return np.asarray(pd.util.hash_pandas_object(pd.DataFrame(norm), index=False).to_numpy())


def rows_preview(rows: np.ndarray | Sequence[int]) -> str:
    shown = ",".join(str(int(r)) for r in list(rows)[:MAX_ROWS_SHOWN])
    return shown + (",..." if len(rows) > MAX_ROWS_SHOWN else "")


def data_finding(
    *,
    detector: str,
    type: str,
    title: str,
    severity: Severity,
    confidence: float,
    dataset: str,
    rule_id: str,
    explanation: str,
    columns: Sequence[str] = (),
    metrics: dict[str, float | int | str] | None = None,
    fix: str | None = None,
    tier: Tier = Tier.unknown,
) -> Finding:
    return Finding(
        type=type,
        title=title,
        severity=severity,
        confidence=round(confidence, 3),
        location=Location(file=dataset, columns=list(columns), node_id=f"data:{dataset}"),
        evidence=Evidence(rule_id=rule_id, explanation=explanation, metrics=metrics or {}),
        detector=detector,
        suggested_fix=fix,
        tier=tier,
    )
