"""Hand labels for Kaggle competitions: what each dataset requires of a pipeline.

The first training labels for the FaultLine model (Phase 4). Each label comes from how the
competition's own test set was built (its data page and rules), never from the notebooks, so a
notebook that splits differently from its label is a candidate fit mismatch. Labels are written
for every split; consumers must use test-split labels only in `--final` runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator

from faultline.eval.corpus import DEFAULT_MANIFEST
from faultline.schema import FINDING_TYPES

DEFAULT_LABELS = DEFAULT_MANIFEST.parent / "labels" / "competitions.yaml"

SplitType = Literal["random", "group", "temporal"]
Task = Literal["binary", "multiclass", "multilabel", "ordinal", "regression"]


class CompetitionLabel(BaseModel):
    split_type: SplitType  # how a validation split must be made to mirror the test set
    group_key: str | None = None  # entity whose rows must stay on one side (group split)
    time_key: str | None = None  # column the test set lies after (temporal split)
    task: Task
    metric: str  # the official metric
    stratify: bool = False  # classification with a rare class, or few rows
    small: bool = False  # few training rows: one holdout is too noisy, use cross-validation
    confidence: Literal["high", "medium"] = "high"  # medium: the test setup isn't fully documented
    reason: str = Field(min_length=1)
    notes: str | None = None  # known leaks and pitfalls, for the human reviewer

    @model_validator(mode="after")
    def _keys_match_split(self) -> Self:
        if self.split_type == "group" and not self.group_key:
            raise ValueError("a group split needs group_key")
        if self.split_type == "temporal" and not self.time_key:
            raise ValueError("a temporal split needs time_key")
        if self.split_type == "random" and (self.group_key or self.time_key):
            raise ValueError("a random split has no group_key or time_key")
        if self.stratify and self.task == "regression":
            raise ValueError("stratify applies to classification only")
        return self


class LabelFile(BaseModel):
    competitions: dict[str, CompetitionLabel]


def load_labels(path: Path = DEFAULT_LABELS) -> dict[str, CompetitionLabel]:
    return LabelFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))).competitions


# ---- per-notebook labels (corpus/labels/notebooks/<competition>.yaml) -------------------------
DEFAULT_NOTEBOOK_LABELS = DEFAULT_LABELS.parent / "notebooks"

Validation = Literal[
    "none",  # fits and predicts the competition test set with no validation of its own
    "random",  # holdout or (K)Fold without stratification or groups
    "stratified",  # stratified holdout or StratifiedKFold
    "group",  # GroupKFold / grouped holdout
    "temporal",  # holdout after a cutoff, TimeSeriesSplit, walk-forward
    "slice",  # unshuffled positional slicing (df[:n]) on data with no meaningful order
]


class Issue(BaseModel):
    type: str  # a FINDING_TYPES entry
    line: int = Field(ge=1)  # in the notebook's concatenated code, as detectors number it
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _known_type(self) -> Self:
        if self.type not in FINDING_TYPES:
            raise ValueError(f"unknown finding type {self.type!r}")
        return self


class NotebookLabel(BaseModel):
    relevant: bool = True  # false: no model is trained (EDA only, unrelated tutorial)
    validation: Validation | None = None  # how the notebook estimates its own score
    validation_line: int | None = None
    issues: list[Issue] = Field(default_factory=list)  # [] = checked and clean
    unsure: bool = False
    note: str | None = None

    @model_validator(mode="after")
    def _relevant_has_validation(self) -> Self:
        if self.relevant and self.validation is None:
            raise ValueError("a relevant notebook needs `validation`")
        return self


class NotebookLabelFile(BaseModel):
    competition: str
    labeler: str  # who labeled; audits add their own file
    notebooks: dict[str, NotebookLabel]  # path relative to the competition's checkout


def load_notebook_labels(directory: Path = DEFAULT_NOTEBOOK_LABELS) -> dict[str, NotebookLabelFile]:
    files = (yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.yaml")))
    return {f.competition: f for f in (NotebookLabelFile.model_validate(d) for d in files)}
