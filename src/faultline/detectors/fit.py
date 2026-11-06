"""Data–pipeline fit (Phase 3b): does the pipeline split, select features and score its data the
way the data requires?

Evidence comes from the code itself (lag features or per-entity aggregates written into the data,
class weights or resampling) and, when given, from a profile of the dataset the code reads
(`faultline data profile`). Code-only rules need strong evidence; with a profile the rules know
the data and can say more. Like leakage.py, facts are collected per file, flow-insensitively.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from faultline.contract import DatasetProfile, looks_like_id_name
from faultline.detectors import _ast as A
from faultline.detectors.base import make_finding
from faultline.detectors.leakage import CV_FUNCS, CV_SEARCH, RESAMPLERS
from faultline.discovery.sources import SourceFile
from faultline.schema import Finding, Severity, Tier

SplitKind = Literal["random", "ordered", "group", "time"]
RANDOM_SPLITTERS = frozenset(
    {
        "KFold",
        "StratifiedKFold",
        "ShuffleSplit",
        "StratifiedShuffleSplit",
        "RepeatedKFold",
        "RepeatedStratifiedKFold",
    }
)
GROUP_SPLITTERS = frozenset(
    {"GroupKFold", "GroupShuffleSplit", "LeaveOneGroupOut", "LeavePGroupsOut", "StratifiedGroupKFold"}
)
TIME_SPLITTERS = frozenset({"TimeSeriesSplit"})
LAG_METHODS = frozenset({"shift", "rolling", "diff", "expanding", "ewm", "pct_change"})
MODULE_ROOTS = frozenset({"numpy", "math", "scipy", "statistics", "torch", "tensorflow", "jax"})
CLASSIFIER = re.compile(r"(Classifier|LogisticRegression(CV)?|SVC|NB)$")
IMBALANCE_KWARGS = ("class_weight", "scale_pos_weight", "is_unbalance", "auto_class_weights")
ROBUST_METRICS = frozenset(
    {
        "roc_auc_score", "f1_score", "fbeta_score", "precision_score", "recall_score",
        "average_precision_score", "balanced_accuracy_score", "classification_report", "confusion_matrix",
        "precision_recall_curve", "matthews_corrcoef", "cohen_kappa_score", "log_loss", "roc_curve",
    }
)  # fmt: skip
ROBUST_SCORING = re.compile(r"roc_auc|f1|precision|recall|balanced|average_precision|log_loss|matthews")
READERS = frozenset({"read_csv", "read_parquet", "read_excel", "read_json", "read_table", "read_feather"})
# Reassigning a loaded frame with these keeps its columns (or only removes/renames known ones).
FRAME_KEEPING = frozenset(
    {
        "dropna", "fillna", "copy", "sort_values", "sort_index", "reset_index", "drop", "query", "rename",
        "drop_duplicates", "astype", "replace", "head", "tail", "sample", "interpolate", "ffill", "bfill",
        "clip", "round", "infer_objects", "convert_dtypes",
    }
)  # fmt: skip
TARGET_NAME = re.compile(r"y|y_\w+|target|labels?")
SMALL_DATA_ROWS = 1000


@dataclass(frozen=True)
class _Split:
    line: int
    call: str
    kind: SplitKind
    stratified: bool


@dataclass
class _Facts:
    splits: list[_Split] = field(default_factory=list)
    lag_lines: list[int] = field(default_factory=list)  # lag/rolling features written into the data
    entity_aggs: list[tuple[int, str]] = field(default_factory=list)  # per-entity aggregates written in
    grouped: set[str] = field(default_factory=set)  # id-like groupby keys
    merges: list[tuple[int, str]] = field(default_factory=list)
    imbalance_line: int | None = None  # class weights or resampling: the author knows it's imbalanced
    classifier: bool = False
    accuracy_lines: list[int] = field(default_factory=list)
    score_lines: list[int] = field(default_factory=list)  # estimator.score(X, y): accuracy for classifiers
    robust_metric: bool = False
    reads: list[tuple[int, str]] = field(default_factory=list)
    frames: dict[str, str] = field(default_factory=dict)  # variable -> dataset path it was read from
    unstable: set[str] = field(default_factory=set)  # frames later reshaped, merged or replaced
    loads: list[tuple[int, str, str]] = field(default_factory=list)  # (line, frame, column) reads
    made: set[str] = field(default_factory=set)  # columns the code creates or renames to
    dropped: set[str] = field(default_factory=set)
    drop_selects: list[tuple[int, set[str]]] = field(default_factory=list)  # X = df.drop(cols): by exclusion
    dedupe: bool = False
    cv: bool = False
    target: str | None = None


def _strs(node: ast.expr | None) -> list[str]:
    if node is None:
        return []
    if (s := A.const_str(node)) is not None:
        return [s]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [s for e in node.elts if (s := A.const_str(e)) is not None]
    return []


def _given(node: ast.expr | None) -> bool:
    return node is not None and not (isinstance(node, ast.Constant) and node.value is None)


def _is_lag(call: ast.Call, aliases: dict[str, str]) -> bool:
    if A.last_segment(call) not in LAG_METHODS or not isinstance(call.func, ast.Attribute):
        return False
    name = A.call_name(call, aliases)  # None for df["x"].shift(): a computed receiver
    return name is None or name.split(".", 1)[0] not in MODULE_ROOTS


def _collect(src: SourceFile) -> _Facts:
    facts = _Facts()
    aliases = A.import_aliases(src.nodes)
    name_assigns: list[tuple[str, ast.expr]] = []
    for node in src.nodes:
        if isinstance(node, ast.Assign):
            _assign(node, facts, aliases, name_assigns)
        elif isinstance(node, ast.Subscript):
            cols = _strs(node.slice)
            if isinstance(node.ctx, ast.Store):
                facts.made.update(cols)
            elif isinstance(node.value, ast.Name):
                facts.loads += [(node.lineno, node.value.id, c) for c in cols]
        elif isinstance(node, ast.Call):
            _call(node, facts, aliases)
    facts.entity_aggs += [(line, c) for line, c in facts.merges if c in facts.grouped]
    for name, value in name_assigns:
        if name not in facts.frames:
            continue
        seg = A.last_segment(value) if isinstance(value, ast.Call) else None
        if seg not in FRAME_KEEPING and seg not in READERS:
            facts.unstable.add(name)
    return facts


def _assign(
    node: ast.Assign, facts: _Facts, aliases: dict[str, str], names: list[tuple[str, ast.expr]]
) -> None:
    value = node.value
    writes_column = any(isinstance(t, ast.Subscript) for t in node.targets) or (
        isinstance(value, ast.Call) and A.last_segment(value) == "assign"
    )
    if writes_column:
        for call in (n for n in ast.walk(value) if isinstance(n, ast.Call)):
            if _is_lag(call, aliases):
                facts.lag_lines.append(node.lineno)
            if A.last_segment(call) == "groupby":
                keys = _strs(A.arg(call, 0, "by"))
                facts.entity_aggs += [(node.lineno, k) for k in keys if looks_like_id_name(k)]
    for t in node.targets:
        if isinstance(t, ast.Attribute) and t.attr == "columns" and isinstance(t.value, ast.Name):
            facts.unstable.add(t.value.id)  # df.columns = [...] renames everything
    if len(node.targets) != 1:
        return
    target = node.targets[0]
    pairs: list[tuple[str, ast.expr]] = []
    if isinstance(target, ast.Name):
        pairs = [(target.id, value)]
    elif (
        isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple) and len(target.elts) == len(value.elts)
    ):
        pairs = [(t.id, v) for t, v in zip(target.elts, value.elts, strict=True) if isinstance(t, ast.Name)]
    for name, v in pairs:
        names.append((name, v))
        if (
            isinstance(v, ast.Subscript)
            and isinstance(v.value, ast.Name)
            and len(cols := _strs(v.slice)) == 1
            and TARGET_NAME.fullmatch(name)
        ):
            facts.target = facts.target or cols[0]
        if not isinstance(v, ast.Call):
            continue
        seg = A.last_segment(v)
        if seg in READERS and (path := A.const_str(A.arg(v, 0, "filepath_or_buffer", "path", "io"))):
            facts.frames[name] = path
        elif seg == "drop":
            dropped = set(_strs(A.arg(v, 0, "labels")) + _strs(A.kwarg(v, "columns")))
            facts.drop_selects.append((node.lineno, dropped))


def _call(call: ast.Call, facts: _Facts, aliases: dict[str, str]) -> None:
    seg = A.last_segment(call) or ""
    line = call.lineno
    if seg == "train_test_split":
        kind: SplitKind = "ordered" if A.is_false(A.kwarg(call, "shuffle")) else "random"
        facts.splits.append(_Split(line, seg, kind, _given(A.kwarg(call, "stratify"))))
    elif seg in RANDOM_SPLITTERS:
        facts.splits.append(_Split(line, seg, "random", "Stratified" in seg))
        facts.cv = True
    elif seg in GROUP_SPLITTERS or seg in TIME_SPLITTERS:
        facts.splits.append(_Split(line, seg, "group" if seg in GROUP_SPLITTERS else "time", False))
        facts.cv = True
    elif seg in CV_FUNCS or seg in CV_SEARCH:
        facts.cv = True
        cv = A.kwarg(call, "cv")
        if _given(A.kwarg(call, "groups")):
            facts.splits.append(_Split(line, seg, "group", False))
        elif cv is None or (isinstance(cv, ast.Constant) and isinstance(cv.value, int)):
            facts.splits.append(_Split(line, seg, "random", True))  # sklearn stratifies classifiers
    if seg in RESAMPLERS or any(_given(A.kwarg(call, k)) for k in IMBALANCE_KWARGS):
        facts.imbalance_line = facts.imbalance_line or line
    if seg[:1].isupper() and CLASSIFIER.search(seg):
        facts.classifier = True
    if seg == "accuracy_score":
        facts.accuracy_lines.append(line)
    elif seg in ROBUST_METRICS:
        facts.robust_metric = True
    elif seg == "score" and len(call.args) == 2:
        facts.score_lines.append(line)
    scoring = A.const_str(A.kwarg(call, "scoring"))
    if scoring == "accuracy":
        facts.accuracy_lines.append(line)
    elif scoring and ROBUST_SCORING.search(scoring):
        facts.robust_metric = True
    if seg in READERS:
        if path := A.const_str(A.arg(call, 0, "filepath_or_buffer", "path", "io")):
            facts.reads.append((line, path))
        facts.dropped.update(_strs(A.kwarg(call, "index_col")))
    elif seg in ("drop_duplicates", "duplicated"):
        facts.dedupe = True
    elif seg == "drop":
        facts.dropped.update(_strs(A.arg(call, 0, "labels")) + _strs(A.kwarg(call, "columns")))
    elif seg == "set_index":
        facts.dropped.update(_strs(A.arg(call, 0, "keys")))
    elif seg == "assign":
        facts.made.update(a for kw in call.keywords if (a := kw.arg))
    elif seg == "rename":
        mapping = A.kwarg(call, "columns") or A.arg(call, 0, "mapper")
        if isinstance(mapping, ast.Dict):
            facts.made.update(s for v in mapping.values if (s := A.const_str(v)) is not None)
    elif seg == "groupby":
        facts.grouped.update(k for k in _strs(A.arg(call, 0, "by")) if looks_like_id_name(k))
    elif seg == "merge":
        facts.merges += [(line, c) for c in _strs(A.kwarg(call, "on")) if looks_like_id_name(c)]


class DataFitDetector:
    name = "data_fit"
    version = "0.1"

    def __init__(self, profiles: Sequence[DatasetProfile] = ()) -> None:
        self.profiles = list(profiles)

    def detect(self, src: SourceFile) -> list[Finding]:
        if src.tree is None:
            return []
        if not self.profiles and not any(
            k in src.code for k in ("split", "Split", "KFold", "cross_val", "CV(")
        ):
            return []
        facts = _collect(src)
        profile = next((p for _, path in facts.reads for p in self.profiles if p.matches(path)), None)
        out = [
            *self._temporal(src, facts, profile),
            *self._group(src, facts, profile),
            *self._stratify(src, facts, profile),
            *self._metric(src, facts, profile),
        ]
        if profile is not None:
            out += [
                *self._schema(src, facts, profile),
                *self._id_feature(src, facts, profile),
                *self._duplicates(src, facts, profile),
                *self._small(src, facts, profile),
            ]
        seen: set[tuple[int | None, str]] = set()
        unique = []
        for f in out:
            if (key := (f.location.line_start, f.type)) not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    @staticmethod
    def _random(facts: _Facts) -> list[_Split]:
        return [s for s in facts.splits if s.kind == "random"]

    def _temporal(self, src: SourceFile, facts: _Facts, profile: DatasetProfile | None) -> list[Finding]:
        if any(s.kind == "time" for s in facts.splits):
            return []
        if facts.lag_lines:
            evidence = f"lag/rolling features are written into the data (line {min(facts.lag_lines)})"
            severity, confidence = Severity.high, 0.8
        elif profile is not None and profile.time_ordered and profile.time_columns:
            evidence = f"{profile.name} is ordered by the time column {profile.time_columns[0]!r}"
            severity, confidence = Severity.medium, 0.6
        else:
            return []
        return [
            make_finding(
                self,
                src,
                type="data.leakage.temporal_split",
                title=f"Random {s.call} on time-ordered data",
                severity=severity,
                confidence=confidence,
                line=s.line,
                rule_id="fit.temporal.random_split",
                explanation=f"This split shuffles rows, but {evidence}. Training rows then come from after "
                "the test rows, so the model is scored on predicting the past from the future and the "
                "score is better than it will be in production.",
                fix="Sort by time and split at a cutoff (train_test_split(..., shuffle=False) on sorted "
                "data), or use TimeSeriesSplit for cross-validation.",
                tier=Tier.tier1,
            )
            for s in self._random(facts)
        ]

    def _group(self, src: SourceFile, facts: _Facts, profile: DatasetProfile | None) -> list[Finding]:
        if any(s.kind == "group" for s in facts.splits):
            return []
        if facts.entity_aggs:
            line, key = min(facts.entity_aggs)
            evidence = f"features are aggregated per {key!r} (line {line})"
            severity, confidence = Severity.high, 0.75
        elif profile is not None and profile.group_keys:
            key = profile.group_keys[0]
            evidence = f"{key!r} repeats across rows of {profile.name}"
            severity, confidence = Severity.medium, 0.55
        else:
            return []
        return [
            make_finding(
                self,
                src,
                type="data.leakage.group_split",
                title=f"Random {s.call} splits rows of the same {key}",
                severity=severity,
                confidence=confidence,
                line=s.line,
                rule_id="fit.group.random_split",
                explanation=f"This split assigns rows at random, but {evidence}. Rows of one {key} land on "
                "both sides, so the model is scored on entities it has already seen. If it will be used "
                "on new ones, the score is too high.",
                fix=f"Split by the key: GroupShuffleSplit / GroupKFold with groups=df[{key!r}].",
                tier=Tier.tier1,
            )
            for s in self._random(facts)
        ]

    @staticmethod
    def _imbalance(facts: _Facts, profile: DatasetProfile | None) -> str | None:
        """Why we think the classes are imbalanced, or None."""
        if profile is not None and profile.imbalanced and profile.minority_share is not None:
            return f"the smallest class is {profile.minority_share:.1%} of {profile.name}"
        if facts.imbalance_line is not None:
            return f"class imbalance is handled at line {facts.imbalance_line}"
        return None

    @staticmethod
    def _classification(facts: _Facts, profile: DatasetProfile | None) -> bool:
        return facts.classifier or (profile is not None and profile.task == "classification")

    def _stratify(self, src: SourceFile, facts: _Facts, profile: DatasetProfile | None) -> list[Finding]:
        why = self._imbalance(facts, profile)
        if why is None or not self._classification(facts, profile):
            return []
        return [
            make_finding(
                self,
                src,
                type="data.fit.unstratified_split",
                title="Imbalanced classes split without stratify",
                severity=Severity.low,
                confidence=0.7,
                line=s.line,
                rule_id="fit.stratify.missing",
                explanation=f"The split doesn't stratify, and {why}. The test set can end up with few or "
                "no minority rows, so its score is unstable and can differ a lot between random seeds.",
                fix="Pass stratify=y to train_test_split.",
                tier=Tier.tier1,
            )
            for s in facts.splits
            if s.call == "train_test_split" and s.kind == "random" and not s.stratified
        ]

    def _metric(self, src: SourceFile, facts: _Facts, profile: DatasetProfile | None) -> list[Finding]:
        why = self._imbalance(facts, profile)
        if why is None or not self._classification(facts, profile) or facts.robust_metric:
            return []
        lines = facts.accuracy_lines + (facts.score_lines if facts.classifier else [])
        if not lines:
            return []
        baseline = ""
        if profile is not None and profile.minority_share is not None and profile.n_classes == 2:
            baseline = (
                f" Always predicting the majority class already scores {1 - profile.minority_share:.0%}."
            )
        return [
            make_finding(
                self,
                src,
                type="data.fit.metric_mismatch",
                title="Accuracy is the only metric on imbalanced classes",
                severity=Severity.medium,
                confidence=0.7,
                line=min(lines),
                rule_id="fit.metric.accuracy_only",
                explanation=f"The pipeline reports only accuracy, but {why}.{baseline} Accuracy hides a "
                "model that misses the minority class.",
                fix="Report average_precision_score / balanced_accuracy_score, or a classification_report, "
                "next to accuracy.",
                tier=Tier.tier1,
            )
        ]

    def _schema(self, src: SourceFile, facts: _Facts, profile: DatasetProfile) -> list[Finding]:
        known = profile.column_names()
        frames = {v for v, path in facts.frames.items() if profile.matches(path)} - facts.unstable
        out: list[Finding] = []
        seen: set[str] = set()
        for line, frame, col in sorted(facts.loads):
            if frame not in frames or col in known or col in facts.made or col in seen:
                continue
            seen.add(col)
            out.append(
                make_finding(
                    self,
                    src,
                    type="data.fit.schema_mismatch",
                    title=f"Column {col!r} is not in {profile.name}",
                    severity=Severity.high,
                    confidence=0.8,
                    line=line,
                    rule_id="fit.schema.missing_column",
                    explanation=f"The code reads {col!r} from a frame loaded from {profile.name}, but the "
                    f"profiled data has no such column ({len(known)} columns). The pipeline fails with a "
                    "KeyError, or it was written for another version of the data.",
                    fix="Check whether the column was renamed or dropped upstream; update the code or the "
                    "dataset profile.",
                    tier=Tier.tier2,
                )
            )
        return out

    def _id_feature(self, src: SourceFile, facts: _Facts, profile: DatasetProfile) -> list[Finding]:
        target = profile.target or facts.target
        # "features = everything but the label": a drop that removes the target
        selects = [line for line, cols in facts.drop_selects if target is not None and target in cols]
        if not selects:
            return []
        ids = [
            c.name
            for c in profile.columns
            if c.identifier
            and looks_like_id_name(c.name)
            and c.name != target
            and c.name not in facts.dropped
        ]
        if not ids:
            return []
        return [
            make_finding(
                self,
                src,
                type="data.fit.id_as_feature",
                title=f"Id column {ids[0]!r} is left in the features",
                severity=Severity.medium,
                confidence=0.65,
                line=min(selects),
                rule_id="fit.features.id_kept",
                explanation=f"Features are 'everything except the dropped columns', which keeps {ids} in. "
                "An id carries no information about new rows, but a model can memorize it, or pick up "
                "the order in which ids were assigned.",
                fix=f"Drop {ids} as well, or select the feature columns explicitly.",
                tier=Tier.tier1,
            )
        ]

    def _duplicates(self, src: SourceFile, facts: _Facts, profile: DatasetProfile) -> list[Finding]:
        if profile.duplicate_rate < 0.01 or facts.dedupe:
            return []
        return [
            make_finding(
                self,
                src,
                type="data.leakage.duplicate_across_split",
                title=f"{profile.duplicate_rate:.1%} duplicate rows are split without dedupe",
                severity=Severity.medium,
                confidence=0.6,
                line=s.line,
                rule_id="fit.duplicates.before_split",
                explanation=f"{profile.duplicate_rate:.1%} of rows in {profile.name} repeat another row, and "
                "nothing removes them before this random split, so copies land in both train and test "
                "and the test score partly measures memorization.",
                fix="drop_duplicates() on the feature columns before splitting.",
                metrics={"duplicate_rate": profile.duplicate_rate},
                tier=Tier.tier1,
            )
            for s in self._random(facts)
        ]

    def _small(self, src: SourceFile, facts: _Facts, profile: DatasetProfile) -> list[Finding]:
        if profile.rows >= SMALL_DATA_ROWS or facts.cv:
            return []
        splits = [s for s in facts.splits if s.call == "train_test_split"]
        if not splits:
            return []
        return [
            make_finding(
                self,
                src,
                type="data.fit.single_split_small",
                title=f"One train/test split of {profile.rows} rows",
                severity=Severity.low,
                confidence=0.6,
                line=splits[0].line,
                rule_id="fit.small.single_split",
                explanation=f"{profile.name} has {profile.rows} rows, so one split leaves a small test set "
                "and its score moves by several points with the random seed.",
                fix="Use cross_val_score (or repeated CV) and report the mean and spread.",
                tier=Tier.tier2,
            )
        ]
