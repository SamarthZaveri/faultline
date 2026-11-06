"""Inject known faults into real datasets to get labeled ground truth for the data detectors.

Each injector changes a copy of a DataFrame and records which rows and columns it touched, so
recall can be measured per fault type. Drift injectors are applied to the "current" side of a
reference/current pair, poisoning injectors to data a model is about to be trained on, and
`leak_duplicates` to a train/test pair. Faults are injected after sampling (docs/PHASES.md,
Phase 3), so the ground truth stays exact. Row numbers are 0-based positions in the returned frame.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from faultline.data.frame import column_kind, is_identifier


@dataclass(frozen=True)
class InjectedDataFault:
    injector: str
    expected_type: str
    columns: tuple[str, ...] = ()
    rows: tuple[int, ...] = ()


class DataInjector(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def expected_type(self) -> str: ...

    def applicable(self, df: pd.DataFrame, target: str | None) -> bool: ...

    def apply(
        self, df: pd.DataFrame, target: str | None, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, InjectedDataFault]: ...


def _numeric(df: pd.DataFrame, target: str | None) -> list[str]:
    """Continuous-ish feature columns: worth shifting."""
    return [
        c
        for c in df.columns
        if c != target
        and column_kind(df[c]) == "numeric"
        and not is_identifier(df[c])
        and df[c].nunique() > 10
    ]


def _categorical(df: pd.DataFrame, target: str | None) -> list[str]:
    return [
        c
        for c in df.columns
        if c != target
        and column_kind(df[c]) == "categorical"
        and not is_identifier(df[c])
        and 2 <= df[c].nunique() <= 50
    ]


def _classes(df: pd.DataFrame, target: str | None) -> list[object]:
    if target is None or target not in df.columns:
        return []
    values = list(df[target].dropna().unique())
    return values if 2 <= len(values) <= 20 else []


def _pick(columns: list[str], rng: np.random.Generator) -> str:
    return columns[int(rng.integers(len(columns)))]


def _target(target: str | None) -> str:
    if target is None:
        raise ValueError("this injector needs a target column")
    return target


def _std(s: pd.Series) -> float:
    std = float(pd.to_numeric(s, errors="coerce").std())
    return std if np.isfinite(std) and std > 0 else 1.0


def _fault(inj: DataInjector, columns: tuple[str, ...] = (), rows: Iterable[Any] = ()) -> InjectedDataFault:
    return InjectedDataFault(inj.name, inj.expected_type, columns=columns, rows=tuple(int(r) for r in rows))


# --- poisoning -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelFlip:
    """Flip a share of labels to another class at random."""

    rate: float = 0.1
    name: str = "label_flip"
    expected_type: str = "data.poisoning.label_flip"

    def applicable(self, df: pd.DataFrame, target: str | None) -> bool:
        return bool(_classes(df, target))

    def apply(
        self, df: pd.DataFrame, target: str | None, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, InjectedDataFault]:
        target = _target(target)
        classes = _classes(df, target)
        labeled = np.flatnonzero(df[target].notna().to_numpy())
        rows = np.sort(rng.choice(labeled, size=max(1, int(self.rate * len(labeled))), replace=False))
        out = df.copy()
        col = out.columns.get_loc(target)
        for r in rows:
            others = [c for c in classes if c != out.iat[r, col]]
            out.iat[r, col] = others[int(rng.integers(len(others)))]
        return out, _fault(self, (target,), rows)


@dataclass(frozen=True)
class InjectedRows:
    """Append copies of real rows with a few features pushed far out, labeled as one class."""

    rate: float = 0.05
    sigma: float = 6.0
    n_columns: int = 3
    name: str = "injected_rows"
    expected_type: str = "data.poisoning.injected_rows"

    def applicable(self, df: pd.DataFrame, target: str | None) -> bool:
        return bool(_numeric(df, target))

    def apply(
        self, df: pd.DataFrame, target: str | None, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, InjectedDataFault]:
        numeric = _numeric(df, target)
        columns = [str(c) for c in rng.choice(numeric, size=min(self.n_columns, len(numeric)), replace=False)]
        m = max(1, int(self.rate * len(df)))
        fake = df.iloc[rng.choice(len(df), size=m, replace=True)].copy()
        for c in columns:
            fake[c] = fake[c].astype(float) + self.sigma * _std(df[c])
        if classes := _classes(df, target):
            fake[_target(target)] = classes[int(rng.integers(len(classes)))]
        out = pd.concat([df, fake], ignore_index=True)
        return out, _fault(self, tuple(columns), range(len(df), len(out)))


@dataclass(frozen=True)
class DuplicateFlood:
    """Append many identical copies of one row, with its label flipped when there is a target."""

    copies: int = 0  # 0 = max(10, 2% of rows)
    name: str = "duplicate_flood"
    expected_type: str = "data.poisoning.injected_rows"

    def applicable(self, df: pd.DataFrame, target: str | None) -> bool:
        return len(df) > 0

    def apply(
        self, df: pd.DataFrame, target: str | None, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, InjectedDataFault]:
        m = self.copies or max(10, int(0.02 * len(df)))
        src = int(rng.integers(len(df)))
        fake = df.iloc[[src] * m].copy()
        classes = _classes(df, target)
        if classes:
            target = _target(target)
            others = [c for c in classes if c != df[target].iloc[src]]
            fake[target] = others[int(rng.integers(len(others)))]
        out = pd.concat([df, fake], ignore_index=True)
        return out, _fault(self, (), range(len(df), len(out)))


# --- drift (apply to the current side) -----------------------------------------------------------


@dataclass(frozen=True)
class CovariateShift:
    """Shift one numeric column by `shift` standard deviations."""

    shift: float = 1.5
    name: str = "covariate_shift"
    expected_type: str = "data.drift.covariate"

    def applicable(self, df: pd.DataFrame, target: str | None) -> bool:
        return bool(_numeric(df, target))

    def apply(
        self, df: pd.DataFrame, target: str | None, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, InjectedDataFault]:
        c = _pick(_numeric(df, target), rng)
        out = df.copy()
        out[c] = out[c].astype(float) + self.shift * _std(df[c])
        return out, _fault(self, (c,))


@dataclass(frozen=True)
class UnitChange:
    """Multiply one positive numeric column by a unit factor (kg -> g)."""

    factor: float = 1000.0
    name: str = "unit_change"
    expected_type: str = "data.drift.covariate"

    def _candidates(self, df: pd.DataFrame, target: str | None) -> list[str]:
        return [c for c in _numeric(df, target) if float(df[c].median()) > 0]

    def applicable(self, df: pd.DataFrame, target: str | None) -> bool:
        return bool(self._candidates(df, target))

    def apply(
        self, df: pd.DataFrame, target: str | None, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, InjectedDataFault]:
        c = _pick(self._candidates(df, target), rng)
        out = df.copy()
        out[c] = out[c].astype(float) * self.factor
        return out, _fault(self, (c,))


@dataclass(frozen=True)
class MissingSpike:
    """Blank out a share of one column's values."""

    rate: float = 0.3
    name: str = "missing_spike"
    expected_type: str = "data.drift.missing_spike"

    def _candidates(self, df: pd.DataFrame, target: str | None) -> list[str]:
        return [
            c for c in df.columns if c != target and not is_identifier(df[c]) and df[c].isna().mean() < 0.5
        ]

    def applicable(self, df: pd.DataFrame, target: str | None) -> bool:
        return bool(self._candidates(df, target))

    def apply(
        self, df: pd.DataFrame, target: str | None, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, InjectedDataFault]:
        c = _pick(self._candidates(df, target), rng)
        present = np.flatnonzero(df[c].notna().to_numpy())
        rows = np.sort(rng.choice(present, size=max(1, int(self.rate * len(df))), replace=False))
        mask = np.zeros(len(df), dtype=bool)
        mask[rows] = True
        out = df.copy()
        out[c] = out[c].mask(mask)
        return out, _fault(self, (c,), rows)


@dataclass(frozen=True)
class NewCategory:
    """Replace a share of one categorical column with a value never seen before."""

    share: float = 0.1
    name: str = "new_category"
    expected_type: str = "data.drift.new_category"

    def applicable(self, df: pd.DataFrame, target: str | None) -> bool:
        return bool(_categorical(df, target))

    def apply(
        self, df: pd.DataFrame, target: str | None, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, InjectedDataFault]:
        c = _pick(_categorical(df, target), rng)
        rows = np.sort(rng.choice(len(df), size=max(1, int(self.share * len(df))), replace=False))
        mask = np.zeros(len(df), dtype=bool)
        mask[rows] = True
        out = df.copy()
        out[c] = out[c].astype(object).mask(mask, f"unseen_{int(rng.integers(1_000_000)):06d}")
        return out, _fault(self, (c,), rows)


@dataclass(frozen=True)
class SchemaDrop:
    """Drop one feature column."""

    name: str = "schema_drop"
    expected_type: str = "data.drift.schema"

    def _candidates(self, df: pd.DataFrame, target: str | None) -> list[str]:
        return [c for c in df.columns if c != target and not is_identifier(df[c])]

    def applicable(self, df: pd.DataFrame, target: str | None) -> bool:
        return bool(self._candidates(df, target))

    def apply(
        self, df: pd.DataFrame, target: str | None, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, InjectedDataFault]:
        c = _pick(self._candidates(df, target), rng)
        return df.drop(columns=[c]), _fault(self, (c,))


@dataclass(frozen=True)
class LabelPrior:
    """Keep only a share of the majority class's rows, so the class balance moves."""

    keep: float = 0.25
    name: str = "label_prior"
    expected_type: str = "data.drift.label_prior"

    def applicable(self, df: pd.DataFrame, target: str | None) -> bool:
        return bool(_classes(df, target))

    def apply(
        self, df: pd.DataFrame, target: str | None, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, InjectedDataFault]:
        target = _target(target)
        majority = df[target].value_counts().idxmax()
        rows = np.flatnonzero((df[target] == majority).to_numpy())
        drop = rng.choice(rows, size=int((1 - self.keep) * len(rows)), replace=False)
        out = df.drop(df.index[drop]).reset_index(drop=True)
        return out, _fault(self, (target,))


@dataclass(frozen=True)
class ConceptShift:
    """Move the label to the next class wherever one feature is above its median: the inputs stay
    the same, the relationship between inputs and label changes."""

    name: str = "concept_shift"
    expected_type: str = "data.drift.concept"

    def applicable(self, df: pd.DataFrame, target: str | None) -> bool:
        return bool(_classes(df, target)) and bool(_numeric(df, target))

    def apply(
        self, df: pd.DataFrame, target: str | None, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, InjectedDataFault]:
        target = _target(target)
        classes = _classes(df, target)
        following = {cls: classes[(i + 1) % len(classes)] for i, cls in enumerate(classes)}
        c = _pick(_numeric(df, target), rng)
        values = pd.to_numeric(df[c], errors="coerce")
        rows = np.flatnonzero(((values > values.median()) & df[target].notna()).to_numpy())
        labels = df[target].to_numpy(copy=True)
        for r in rows:
            labels[r] = following[labels[r]]
        out = df.copy()
        out[target] = labels
        return out, _fault(self, (target, c), rows)


def leak_duplicates(
    train: pd.DataFrame, test: pd.DataFrame, *, rate: float = 0.05, rng: np.random.Generator
) -> tuple[pd.DataFrame, InjectedDataFault]:
    """Replace a share of test rows with copies of training rows (data.leakage.duplicate_across_split)."""
    if any(c not in train.columns for c in test.columns):
        raise ValueError("train must have every test column")
    m = max(1, int(rate * len(test)))
    replaced = rng.choice(len(test), size=m, replace=False)
    copies = train.iloc[rng.choice(len(train), size=m, replace=m > len(train))][list(test.columns)]
    kept = test.drop(test.index[replaced])
    out = pd.concat([kept, copies], ignore_index=True)
    fault = InjectedDataFault(
        "leak_duplicates", "data.leakage.duplicate_across_split", rows=tuple(range(len(kept), len(out)))
    )
    return out, fault


POISONING_INJECTORS: tuple[DataInjector, ...] = (LabelFlip(), InjectedRows(), DuplicateFlood())
DRIFT_INJECTORS: tuple[DataInjector, ...] = (
    CovariateShift(),
    UnitChange(),
    MissingSpike(),
    NewCategory(),
    SchemaDrop(),
    LabelPrior(),
    ConceptShift(),
)
