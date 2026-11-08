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
from pathlib import PurePosixPath
from typing import Literal

from faultline.contract import DatasetProfile, looks_like_id_name
from faultline.detectors import _ast as A
from faultline.detectors.base import make_finding
from faultline.detectors.leakage import CV_FUNCS, CV_SEARCH, RESAMPLERS, random_holdouts
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
TARGET_NAME = re.compile(r"y|y_\w+|target|labels?|y\w*", re.IGNORECASE)
AUC_CALLS = frozenset({"roc_auc_score", "roc_curve", "auc", "AUC", "average_precision_score"})
AUC_STR = re.compile(r"^(roc_)?auc$|^AUC$")
LOGLOSS_CALLS = frozenset({"log_loss"})
LOGLOSS_STR = re.compile(r"log_?loss|crossentropy|mlogloss|^neg_log_loss$", re.IGNORECASE)
ACCURACY_STR = re.compile(r"^(accuracy|acc|categorical_accuracy|binary_accuracy)$")
OTHER_CLS_METRICS = frozenset(
    {"precision_score", "recall_score", "f1_score", "classification_report", "confusion_matrix",
     "mean_squared_error", "mean_absolute_error", "balanced_accuracy_score", "matthews_corrcoef"}
)  # fmt: skip
TARGET_STATS = frozenset({"mean", "median", "sum", "std", "agg", "transform", "count", "max", "min"})
FEATURE_LIST = re.compile(r"feat|col|predictor|use|var", re.IGNORECASE)
EXCLUSION_LIST = re.compile(r"drop|exclu|remov|not_|ignore|skip|original|cat_|num_", re.IGNORECASE)
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
    holdouts: list[int] = field(default_factory=list)  # single held-out splits (incl. validation_split=)
    cv_eval: bool = False  # scores come from cross-validation (not only a search inside a holdout)
    auc: bool = False
    logloss: bool = False
    other_metric_lines: list[int] = field(default_factory=list)  # precision/recall/f1/mse/...
    indicator_filter: bool = False  # rows filtered to one per entity (df.query('is_head == 1'))
    target_cols: set[str] = field(default_factory=set)  # the label column and columns derived from it
    target_encodings: list[tuple[int, str, str]] = field(default_factory=list)  # (line, column, written)
    fold_loops: list[tuple[int, int]] = field(default_factory=list)  # line ranges of loops over CV folds
    feature_lists: list[tuple[int, str, set[str]]] = field(default_factory=list)  # name = ['a', 'id']
    unused_splitters: set[int] = field(default_factory=set)  # id() of splitter calls never used
    one_row_per_group: set[str] = field(default_factory=set)  # keys the data is aggregated to
    lists: dict[str, set[str]] = field(default_factory=dict)  # name = ['a', 'b'] (any list of strings)
    drop_names: set[str] = field(default_factory=set)  # names passed to drop(...): resolved via lists


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


def _metric_facts(src: SourceFile, facts: _Facts) -> None:
    """Which metric families the code computes (calls and metric strings; imports don't count)."""
    for n in src.nodes:
        if isinstance(n, ast.Call):
            seg = A.last_segment(n) or ""
            if seg in AUC_CALLS:
                facts.auc = True
            elif seg in LOGLOSS_CALLS:
                facts.logloss = True
            elif seg in OTHER_CLS_METRICS:
                facts.other_metric_lines.append(n.lineno)
            for kw in n.keywords:
                if kw.arg not in ("metrics", "metric", "eval_metric", "scoring", "loss", "objective"):
                    continue
                for c in ast.walk(kw.value):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str):
                        if AUC_STR.search(c.value):
                            facts.auc = True
                        elif LOGLOSS_STR.search(c.value):
                            facts.logloss = True
                        elif ACCURACY_STR.search(c.value):
                            facts.accuracy_lines.append(n.lineno)
        elif isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values, strict=True):
                if A.const_str(k) in ("metric", "eval_metric", "objective") and (val := A.const_str(v)):
                    facts.auc = facts.auc or bool(AUC_STR.search(val))
                    facts.logloss = facts.logloss or bool(LOGLOSS_STR.search(val)) or "multi" in val
    for n in src.nodes:
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "query"
            and re.search(r"==\s*1\b", A.const_str(A.arg(n, 0, "expr")) or "")
        ):
            facts.indicator_filter = True  # df.query('is_head == 1'): one row per entity


def _unused_splitters(nodes: list[ast.AST]) -> set[int]:
    """`kf = KFold(5)` that nothing reads before kf is assigned again (or ever)."""
    loads: dict[str, list[int]] = {}
    stores: dict[str, list[int]] = {}
    for n in nodes:
        if isinstance(n, ast.Name):
            (loads if isinstance(n.ctx, ast.Load) else stores).setdefault(n.id, []).append(n.lineno)
    out: set[int] = set()
    for n in nodes:
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Call)
            and (A.last_segment(n.value) or "") in RANDOM_SPLITTERS | GROUP_SPLITTERS | TIME_SPLITTERS
        ):
            name, line = n.targets[0].id, n.lineno
            nxt = min((ln for ln in stores.get(name, []) if ln > line), default=10**9)
            if not any(line < ln < nxt for ln in loads.get(name, [])):
                out.add(id(n.value))
    return out


def _set_difference(v: ast.expr) -> set[str]:
    """set(df.columns) - set(['y']) / df.columns.difference(['y']): every column but these."""
    for n in ast.walk(v):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub) and "columns" in ast.unparse(n.left):
            return {c for sub in ast.walk(n.right) if isinstance(sub, ast.expr) for c in _strs(sub)}
        if isinstance(n, ast.Call) and A.last_segment(n) == "difference" and "columns" in ast.unparse(n.func):
            return set(_strs(A.arg(n, 0, "other")))
    return set()


def _label_column(v: ast.expr) -> str | None:
    """y = df['col'] (also wrapped: np.log1p(df['col']).values)."""
    for n in ast.walk(v):
        if isinstance(n, ast.Subscript) and len(cols := _strs(n.slice)) == 1:
            return cols[0]
        if isinstance(n, ast.Call) and A.last_segment(n) == "groupby":
            return None  # y = df.groupby(key)[col].mean(): a per-entity label, handled elsewhere
    return None


def _target_facts(src: SourceFile, facts: _Facts) -> None:
    """Columns derived from the label, and group statistics of them written into features."""
    if not facts.target_cols:
        return
    for _ in range(2):  # a derived column can be derived from another derived column
        for node in src.nodes:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) for t in node.targets):
                text = ast.unparse(node)
                if any(f"'{c}'" in text or f'"{c}"' in text for c in facts.target_cols):
                    for t in node.targets:
                        if isinstance(t, ast.Subscript):
                            sl = t.slice.elts[-1] if isinstance(t.slice, ast.Tuple) else t.slice
                            facts.target_cols.update(_strs(sl))
    for node in src.nodes:
        if not isinstance(node, ast.Assign):
            continue
        written = [t for t in node.targets if isinstance(t, (ast.Name, ast.Subscript))]
        if not written or any(isinstance(t, ast.Name) and TARGET_NAME.fullmatch(t.id) for t in written):
            continue
        for c in ast.walk(node.value):
            if not (isinstance(c, ast.Call) and A.last_segment(c) in TARGET_STATS):
                continue
            recv = c.func.value if isinstance(c.func, ast.Attribute) else None
            if recv is None or not any(
                isinstance(g, ast.Call) and A.last_segment(g) == "groupby" for g in ast.walk(recv)
            ):
                continue
            cols = {s_ for sub in ast.walk(recv) if isinstance(sub, ast.Subscript) for s_ in _strs(sub.slice)}
            cols |= {a.attr for a in ast.walk(recv) if isinstance(a, ast.Attribute)}
            hit = cols & facts.target_cols
            if not hit:
                continue
            name = next((t.id for t in written if isinstance(t, ast.Name)), None)
            used = any(isinstance(t, ast.Subscript) for t in written) or (
                name is not None and re.search(rf"\.map\(\s*{name}\b|merge\([^)]*\b{name}\b", src.code)
            )
            if used:
                facts.target_encodings.append((node.lineno, min(hit), name or ""))
            break


def _collect(src: SourceFile) -> _Facts:
    facts = _Facts()
    aliases = A.import_aliases(src.nodes)
    name_assigns: list[tuple[str, ast.expr]] = []
    facts.unused_splitters = _unused_splitters(src.nodes)
    _metric_facts(src, facts)
    for line in random_holdouts(src):  # df.sample(...) + df.drop(val.index), data.iloc[idx_tr] / [idx_va]
        facts.splits.append(_Split(line, "holdout", "random", False))
        facts.holdouts.append(line)
    for node in src.nodes:
        if isinstance(node, (ast.For, ast.AsyncFor)) and ".split(" in ast.unparse(node.iter):
            facts.fold_loops.append((node.lineno, node.end_lineno or node.lineno))
        elif isinstance(node, ast.Delete):
            for t in node.targets:
                if isinstance(t, ast.Subscript):
                    facts.dropped.update(_strs(t.slice))
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
    for name in facts.drop_names:
        facts.dropped.update(facts.lists.get(name, set()))
    _target_facts(src, facts)
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
        if TARGET_NAME.fullmatch(name) and (col := _label_column(v)) is not None:
            facts.target = facts.target or col
            facts.target_cols.add(col)
        if isinstance(v, (ast.List, ast.Tuple)) and (strs_ := set(_strs(v))):
            facts.lists[name] = strs_
        removed_ = _set_difference(v)
        if removed_:
            facts.drop_selects.append((node.lineno, removed_))
        if isinstance(v, (ast.List, ast.Tuple)) and FEATURE_LIST.search(name):
            cols_ = set(_strs(v))
            if cols_:
                facts.feature_lists.append((node.lineno, name, cols_))
        if isinstance(v, ast.ListComp) and len(v.generators) == 1:
            gen = v.generators[0]
            if isinstance(gen.iter, ast.Attribute) and gen.iter.attr == "columns":
                excluded = {c for cond in gen.ifs for cmp in ast.walk(cond) if isinstance(cmp, ast.Compare)
                            for comp in cmp.comparators for c in _strs(comp)}  # fmt: skip
                if excluded:
                    facts.drop_selects.append((node.lineno, excluded))
        if not isinstance(v, ast.Call):
            continue
        seg = A.last_segment(v)
        if seg in READERS and (path := A.const_str(A.arg(v, 0, "filepath_or_buffer", "path", "io"))):
            facts.frames[name] = path
        elif seg == "drop":
            dropped = set(_strs(A.arg(v, 0, "labels")) + _strs(A.kwarg(v, "columns")))
            facts.drop_selects.append((node.lineno, dropped))
        elif seg in ("mean", "sum", "agg", "first", "max", "min", "median", "size", "count", "last"):
            # X = df.groupby(key).agg(...): one row per key (not transform, which keeps every row)
            for c in ast.walk(v):
                if isinstance(c, ast.Call) and A.last_segment(c) == "groupby":
                    facts.one_row_per_group.update(_strs(A.arg(c, 0, "by")))


def _call(call: ast.Call, facts: _Facts, aliases: dict[str, str]) -> None:
    seg = A.last_segment(call) or ""
    line = call.lineno
    if id(call) in facts.unused_splitters:
        return
    if _given(A.kwarg(call, "validation_split")):
        facts.holdouts.append(line)
    vs = A.kwarg(call, "validation_set")  # tflearn model.fit(..., validation_set=0.2): a random holdout
    if isinstance(vs, ast.Constant) and isinstance(vs.value, float):
        facts.splits.append(_Split(line, "validation_set", "random", False))
        facts.holdouts.append(line)
    if seg == "train_test_split":
        kind: SplitKind = "ordered" if A.is_false(A.kwarg(call, "shuffle")) else "random"
        facts.splits.append(_Split(line, seg, kind, _given(A.kwarg(call, "stratify"))))
        facts.holdouts.append(line)
    elif seg in RANDOM_SPLITTERS:
        facts.splits.append(_Split(line, seg, "random", "Stratified" in seg))
        facts.cv = True
        n_splits = A.arg(call, 0, "n_splits")
        if "ShuffleSplit" in seg and isinstance(n_splits, ast.Constant) and n_splits.value == 1:
            facts.holdouts.append(line)
        else:
            facts.cv_eval = True
    elif seg in GROUP_SPLITTERS or seg in TIME_SPLITTERS:
        facts.splits.append(_Split(line, seg, "group" if seg in GROUP_SPLITTERS else "time", False))
        facts.cv = True
        facts.cv_eval = True
    elif seg in CV_FUNCS or seg in CV_SEARCH:
        facts.cv = True
        facts.cv_eval = facts.cv_eval or seg in CV_FUNCS
        cv = A.kwarg(call, "cv")
        if _given(A.kwarg(call, "groups")):
            facts.splits.append(_Split(line, seg, "group", False))
        elif cv is None or (isinstance(cv, ast.Constant) and isinstance(cv.value, int)):
            facts.splits.append(_Split(line, seg, "random", True))  # sklearn stratifies classifiers
    if seg == "cv" and isinstance(call.func, ast.Attribute) and A.dotted(call.func.value) in ("xgb", "lgb"):
        facts.cv = facts.cv_eval = True
        strat = A.kwarg(call, "stratified")
        stratified = A.is_true(strat) if strat is not None else A.dotted(call.func.value) == "lgb"
        facts.splits.append(_Split(line, f"{A.dotted(call.func.value)}.cv", "random", stratified))
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
        facts.one_row_per_group.update(_strs(A.kwarg(call, "subset")))
    elif seg == "pop":
        facts.dropped.update(_strs(A.arg(call, 0, "item")))
    elif seg == "drop":
        facts.dropped.update(_strs(A.arg(call, 0, "labels")) + _strs(A.kwarg(call, "columns")))
        for given in (A.arg(call, 0, "labels"), A.kwarg(call, "columns")):
            if isinstance(given, ast.Name):
                facts.drop_names.add(given.id)
    elif seg in ("append", "extend") and isinstance(call.func, ast.Attribute):
        owner = call.func.value
        if isinstance(owner, ast.Name) and FEATURE_LIST.search(owner.id) and call.args:
            added = set(_strs(call.args[0]))
            if added:
                facts.feature_lists.append((line, owner.id, added))
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
        if profile is None:
            profile = next((p for p in self.profiles if p.matches("")), None)  # describes any data read
        out = [
            *self._temporal(src, facts, profile),
            *self._group(src, facts, profile),
            *self._stratify(src, facts, profile),
            *self._metric(src, facts, profile),
            *self._target_encoding(src, facts),
            *self._id_by_code(src, facts),
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
        if profile is not None and not profile.time_ordered:
            return []  # the profile says the data has no time order to respect
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
        if profile is not None and (
            not profile.group_keys
            or set(profile.group_keys) & facts.one_row_per_group
            or facts.indicator_filter
        ):
            return []  # no entity repeats across rows, or the data is aggregated to one row per entity
        if profile is None and facts.entity_aggs:
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
            if s.kind == "random"
            and not s.stratified
            and (
                s.call in ("train_test_split", "holdout", "validation_set")
                or s.call in RANDOM_SPLITTERS
                or s.call.endswith(".cv")
            )
        ]

    def _metric(self, src: SourceFile, facts: _Facts, profile: DatasetProfile | None) -> list[Finding]:
        if profile is not None and profile.metric:
            return self._official_metric(src, facts, profile.metric)
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

    def _official_metric(self, src: SourceFile, facts: _Facts, metric: str) -> list[Finding]:
        """The notebook scores itself, but never with the metric results are judged by."""
        m = metric.lower()
        family = "auc" if "auc" in m else "logloss" if "logloss" in m or "log_loss" in m else None
        if family is None:
            return []  # accuracy, regression and ranking metrics: nothing reliable to compare
        lines = facts.accuracy_lines + facts.score_lines + facts.other_metric_lines
        if not facts.splits or not lines or (facts.auc if family == "auc" else facts.logloss):
            return []
        return [
            make_finding(
                self,
                src,
                type="data.fit.metric_mismatch",
                title=f"Scored with accuracy, but results are judged by {metric}",
                severity=Severity.medium,
                confidence=0.7,
                line=min(lines),
                rule_id="fit.metric.not_official",
                explanation=f"The notebook only reports accuracy-style scores, but the data's metric is {metric}"
                ", which depends on predicted probabilities and ranking. Model choices made on accuracy can "
                "go the wrong way for it.",
                fix=f"Score validation with the official metric ({metric}) on predicted probabilities.",
                tier=Tier.tier1,
            )
        ]

    def _target_encoding(self, src: SourceFile, facts: _Facts) -> list[Finding]:
        if not facts.target_encodings or not facts.splits:
            return []
        out = []
        for line, col, _ in facts.target_encodings:
            if any(a <= line <= b for a, b in facts.fold_loops):
                continue  # computed inside a CV loop: out of fold
            if not any(s.line > line for s in facts.splits):
                continue
            out.append(
                make_finding(
                    self,
                    src,
                    type="data.leakage.target_derived",
                    title=f"Group statistics of the label {col!r} used as a feature",
                    severity=Severity.high,
                    confidence=0.75,
                    line=line,
                    rule_id="fit.target.encoding_before_split",
                    explanation=f"A per-group statistic of {col!r} (the label or a column made from it) is "
                    "computed on all training rows and written into the features before they are split or "
                    "cross-validated, so each validation row's own label is inside its feature.",
                    fix="Compute target encodings out of fold (inside each CV training fold).",
                    tier=Tier.tier1,
                )
            )
        return out[:1]

    def _id_by_code(self, src: SourceFile, facts: _Facts) -> list[Finding]:
        """An identifier column kept among the model's features."""
        if not facts.splits:
            return []
        for line, name, cols in facts.feature_lists:
            if EXCLUSION_LIST.search(name) or not re.search(rf"\[\s*{name}\s*\]", src.code):
                continue
            ids = sorted(c for c in cols if looks_like_id_name(c) and c not in facts.dropped)
            if ids:
                return [self._id_finding(src, line, ids[0], f"it is listed in {name}")]
        # the row id the notebook reads from the test file (to write the submission)
        row_ids = {
            c
            for _, frame, c in facts.loads
            if looks_like_id_name(c) and "test" in PurePosixPath(facts.frames.get(frame, "")).name.lower()
        }
        target = facts.target
        for line, removed in facts.drop_selects:
            if target is None or target not in removed:
                continue
            ids = sorted(c for c in row_ids if c not in removed and c not in facts.dropped)
            if ids:
                return [self._id_finding(src, line, ids[0], "the features are every column except the label")]
        return []

    def _id_finding(self, src: SourceFile, line: int, col: str, why: str) -> Finding:
        return make_finding(
            self,
            src,
            type="data.fit.id_as_feature",
            title=f"Identifier {col!r} is a model feature",
            severity=Severity.medium,
            confidence=0.6,
            line=line,
            rule_id="fit.features.id_in_code",
            explanation=f"{col!r} looks like a row identifier and {why}. An id carries no meaning; when it "
            "tracks time or order it leaks, otherwise it is noise the model can overfit.",
            fix=f"Drop {col!r} from the features (or set it as the index).",
            tier=Tier.tier1,
        )

    def _schema(self, src: SourceFile, facts: _Facts, profile: DatasetProfile) -> list[Finding]:
        known = profile.column_names()
        if not known:
            return []  # a profile without its column list can't check the schema
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
        if profile.rows >= SMALL_DATA_ROWS or facts.cv_eval or not facts.holdouts:
            return []
        return [
            make_finding(
                self,
                src,
                type="data.fit.single_split_small",
                title=f"One train/test split of {profile.rows} rows",
                severity=Severity.low,
                confidence=0.6,
                line=min(facts.holdouts),
                rule_id="fit.small.single_split",
                explanation=f"{profile.name} has {profile.rows} rows, so one split leaves a small test set "
                "and its score moves by several points with the random seed.",
                fix="Use cross_val_score (or repeated CV) and report the mean and spread.",
                tier=Tier.tier2,
            )
        ]
