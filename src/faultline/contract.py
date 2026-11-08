"""Dataset profiles: what a dataset looks like, so detectors can check that a pipeline fits it
(Phase 3b). Plain pydantic and no pandas, so code detectors can import this.

A profile is made once per dataset by `faultline data profile` (faultline.data.profile) and stored
as YAML. It holds column names and aggregates only, never cell values. The full data contract
(required split, metrics, drift setup) will grow from this.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ID_NAME = re.compile(r"(^|_)(id|uuid|guid|index)$|^unnamed: ?0$", re.IGNORECASE)
CAMEL_ID = re.compile(r"[a-z]I[dD]$")  # PassengerId, userID (case-sensitive: not "paid", "valid")
IMBALANCED_BELOW = 0.2  # minority class share


def looks_like_id_name(name: str) -> bool:
    return bool(ID_NAME.search(name) or CAMEL_ID.search(name))


class ColumnProfile(BaseModel):
    name: str
    kind: Literal["numeric", "categorical", "datetime", "other"]
    missing_rate: float = Field(ge=0.0, le=1.0)
    distinct: int = Field(ge=0)
    identifier: bool = False  # (nearly) one value per row: an id, hash or free text
    numeric_text: bool = False  # stored as text, but the values are numbers
    time_like: bool = False  # datetime, or text that parses as dates
    monotonic: bool = False  # rows are already in order of this (time) column


class DatasetProfile(BaseModel):
    name: str  # path or file name, matched against the paths the code reads
    rows: int = Field(ge=0)
    columns: list[ColumnProfile] = Field(default_factory=list)
    target: str | None = None
    task: Literal["classification", "regression", "unknown"] = "unknown"
    n_classes: int | None = None
    minority_share: float | None = None
    duplicate_rate: float = 0.0  # rows that repeat an earlier row (ids ignored)
    group_keys: list[str] = Field(default_factory=list)  # entity columns repeated across rows
    time_columns: list[str] = Field(default_factory=list)
    time_ordered: bool = False
    metric: str | None = None  # the metric results are judged by (e.g. auc, logloss), when known

    def column_names(self) -> set[str]:
        return {c.name for c in self.columns}

    @property
    def imbalanced(self) -> bool:
        return self.minority_share is not None and self.minority_share < IMBALANCED_BELOW

    def matches(self, path: str) -> bool:
        """A path in code refers to this dataset: same path, or the same file name."""
        own = PurePosixPath(self.name.replace("\\", "/"))
        other = PurePosixPath(path.replace("\\", "/"))
        return path == self.name or own.name == other.name


def save_profile(profile: DatasetProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_profiles(directory: Path) -> list[DatasetProfile]:
    return [
        DatasetProfile.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
        for p in sorted(directory.glob("*.yaml"))
    ]
