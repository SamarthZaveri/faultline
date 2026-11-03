"""Leakage in ML code: test data reaching the model or its preprocessing during training.

Static, per scope (module/notebook or function), ordered by line. Tracks which names flow
into which (flow-insensitive, transitive) and where data is split (train_test_split, CV, or
complementary slices like data[:n] / data[n:]), then flags:
  - preprocessing fit on data that is split later        -> preprocessing_before_split
  - a statistic (mean, std, ...) of data that is split later
    written back into that data                           -> preprocessing_before_split
  - feature selection fit on data that is split later    -> feature_selection_full
  - resampling (SMOTE, ...) of data that is split later  -> resampling_before_split
  - anything fit on the test side of a split, or a model
    fit on all data and then evaluated on the test side  -> test_in_training
  - the test side used for early stopping / validation,
    or to compare several models / hyperparameters        -> tuning_on_test
Modeled on the leakage categories of Yang et al., ASE'22 ("Data Leakage in Notebooks").
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from faultline.detectors import _ast as A
from faultline.detectors.base import make_finding
from faultline.discovery.sources import SourceFile
from faultline.schema import Finding, Severity

PREPROCESSORS = frozenset(
    {
        "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "Normalizer",
        "QuantileTransformer", "PowerTransformer", "PCA", "TruncatedSVD", "KernelPCA",
        "SimpleImputer", "KNNImputer", "IterativeImputer", "OneHotEncoder", "OrdinalEncoder",
        "TargetEncoder", "PolynomialFeatures", "TfidfVectorizer", "CountVectorizer", "KBinsDiscretizer",
    }
)  # fmt: skip
SELECTORS = frozenset(
    {
        "SelectKBest", "SelectPercentile", "SelectFpr", "SelectFdr", "SelectFwe",
        "GenericUnivariateSelect", "RFE", "RFECV", "SelectFromModel", "SequentialFeatureSelector",
    }
)  # fmt: skip
RESAMPLERS = frozenset(
    {
        "SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE", "SMOTENC", "RandomOverSampler",
        "RandomUnderSampler", "NearMiss", "TomekLinks", "SMOTEENN", "SMOTETomek", "ClusterCentroids",
    }
)  # fmt: skip
CV_SEARCH = frozenset({"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV"})
CV_SPLITTERS = frozenset(
    {
        "KFold", "StratifiedKFold", "GroupKFold", "ShuffleSplit", "StratifiedShuffleSplit",
        "RepeatedKFold", "RepeatedStratifiedKFold", "TimeSeriesSplit",
    }
)  # fmt: skip
CV_FUNCS = frozenset({"cross_val_score", "cross_validate", "cross_val_predict", "learning_curve"})
# Composites: fitting one fits every step inside it (and a final estimator, if any).
COMPOSITES = frozenset(
    {"Pipeline", "make_pipeline", "ColumnTransformer", "make_column_transformer", "FeatureUnion"}
)
FIT_BEFORE_SPLIT = PREPROCESSORS | SELECTORS | COMPOSITES
NOT_MODELS = FIT_BEFORE_SPLIT | RESAMPLERS | CV_SEARCH | CV_SPLITTERS
FIT_METHODS = frozenset({"fit", "fit_transform", "fit_resample"})
STAT_METHODS = frozenset({"mean", "median", "std", "var", "min", "max", "quantile", "mode"})
EVAL_METHODS = frozenset(
    {"predict", "predict_proba", "predict_log_proba", "decision_function", "score", "evaluate"}
)
SELECTION_KWARGS = ("eval_set", "validation_data", "valid_sets", "eval_X")
MODULES = frozenset({"np", "numpy", "pd", "pandas", "math", "statistics", "torch", "tf", "scipy", "stats"})
# Names of a held-out split meant for validation (a legitimate early-stopping / selection set).
VALIDATION_NAME = re.compile(r"val|valid|dev", re.IGNORECASE)
TRAIN_NAME = re.compile(r"train", re.IGNORECASE)

FitKind = Literal["fit", "stat", "resample"]
Ancestors = Callable[[set[str]], set[str]]


@dataclass
class _Fit:
    line: int
    scope: int
    cls: str | None
    data: set[str]  # names in X (and y, for models) arguments
    kind: FitKind = "fit"
    receiver: str | None = None  # object name for `name.fit(...)`
    writes: set[str] = field(default_factory=set)  # stat: names the statistic is written into
    selection: set[str] = field(default_factory=set)  # names passed as eval_set / validation_data
    loop: int | None = None


@dataclass
class _Eval:
    line: int
    scope: int
    receiver: str
    data: set[str]
    loop: int | None


@dataclass
class _Split:
    line: int
    scope: int
    inputs: set[str]
    train_out: set[str] = field(default_factory=set)
    test_out: set[str] = field(default_factory=set)


class _Collector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scopes = [0]
        self.n_scopes = 0
        self.loops: list[int] = []
        self.n_loops = 0
        self.deps: dict[str, set[str]] = {}
        self.objects: dict[tuple[int, str], str] = {}  # (scope, name) -> estimator class
        self.targets: dict[int, ast.expr] = {}  # id(call) -> assignment target
        self.slices: dict[tuple[int, str, str, str], tuple[str, int]] = {}
        self.fits: list[_Fit] = []
        self.evals: list[_Eval] = []
        self.splits: list[_Split] = []

    @property
    def scope(self) -> int:
        return self.scopes[-1]

    @property
    def loop(self) -> int | None:
        return self.loops[-1] if self.loops else None

    def _enter(self, node: ast.AST) -> None:
        self.n_scopes += 1
        self.scopes.append(self.n_scopes)
        self.generic_visit(node)
        self.scopes.pop()

    visit_FunctionDef = _enter
    visit_AsyncFunctionDef = _enter

    def _loop(self, node: ast.AST) -> None:
        self.n_loops += 1
        self.loops.append(self.n_loops)
        self.generic_visit(node)
        self.loops.pop()

    visit_For = _loop
    visit_AsyncFor = _loop
    visit_While = _loop

    # -- assignments: dataflow, estimator objects, slicing splits, statistics
    def _bind(self, targets: list[ast.expr], value: ast.expr, line: int) -> None:
        written: set[str] = set()
        for t in targets:
            # a, b = f(x), g(y): a depends on x only, b on y only (pairwise, not all-to-all)
            if isinstance(t, ast.Tuple) and isinstance(value, ast.Tuple) and len(t.elts) == len(value.elts):
                pairs = list(zip(t.elts, value.elts, strict=True))
            else:
                pairs = [(t, value)]
            for tt, vv in pairs:
                sources = A.names_in(vv)
                for name in A.target_names(tt):
                    self.deps.setdefault(name, set()).update(sources)
                    written.add(name)
        if isinstance(value, ast.Call):
            self.targets[id(value)] = targets[0]
            cls = _ctor(value)
            if cls and len(targets) == 1 and isinstance(targets[0], ast.Name):
                self.objects[(self.scope, targets[0].id)] = cls
        for t in targets:
            if isinstance(t, ast.Tuple) and isinstance(value, ast.Tuple) and len(t.elts) == len(value.elts):
                for tt, vv in zip(t.elts, value.elts, strict=True):
                    self._slice(tt, vv, line)
            else:
                self._slice(t, value, line)
        self._stats(value, written, line)

    def _slice(self, target: ast.expr, value: ast.expr, line: int) -> None:
        if not (isinstance(target, ast.Name) and isinstance(value, ast.Subscript)):
            return
        sl = value.slice.elts[0] if isinstance(value.slice, ast.Tuple) and value.slice.elts else value.slice
        if not isinstance(sl, ast.Slice) or sl.step is not None:
            return
        if sl.lower is None and sl.upper is not None:
            side, bound, other = "head", sl.upper, "tail"
        elif sl.lower is not None and sl.upper is None:
            side, bound, other = "tail", sl.lower, "head"
        else:
            return
        key = (self.scope, ast.dump(value.value), ast.dump(bound))
        match = self.slices.get((*key, other))
        if match is None:
            self.slices[(*key, side)] = (target.id, line)
            return
        head, tail = (target.id, match[0]) if side == "head" else (match[0], target.id)
        if TRAIN_NAME.search(tail) and not TRAIN_NAME.search(head):
            head, tail = tail, head  # e.g. y_valid = y[:n]; y_train = y[n:]
        self.splits.append(_Split(line, self.scope, A.names_in(value.value), {head}, {tail}))

    def _stats(self, value: ast.expr, written: set[str], line: int) -> None:
        if not written:
            return
        for n in ast.walk(value):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in STAT_METHODS:
                data = (A.names_in(n.func.value) | {a for x in n.args for a in A.names_in(x)}) - MODULES
                if data:
                    stat = _Fit(line, self.scope, f"{n.func.attr}()", data, kind="stat", writes=written)
                    self.fits.append(stat)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._bind(node.targets, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._bind([node.target], node.value, node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._bind([node.target], node.value, node.lineno)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        # df.fillna(x, inplace=True) writes into df like an assignment would.
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and A.is_true(A.kwarg(call, "inplace"))
            and (base := A.target_names(call.func.value))
        ):
            self._bind([ast.Name(id=n) for n in sorted(base)], call, node.lineno)
        self.generic_visit(node)

    # -- calls: splits, fits, evaluations
    def _cls_of(self, receiver: ast.expr) -> str | None:
        if isinstance(receiver, ast.Call):
            return _ctor(receiver)
        if isinstance(receiver, ast.Name):
            return self.objects.get((self.scope, receiver.id)) or self.objects.get((0, receiver.id))
        return None

    def visit_Call(self, node: ast.Call) -> None:
        seg = A.last_segment(node)
        func = node.func
        if seg == "train_test_split":
            self._split(node)
        elif seg in CV_FUNCS and len(node.args) > 1:
            self.splits.append(_Split(node.lineno, self.scope, A.names_in(node.args[1])))
        elif isinstance(func, ast.Attribute) and seg in FIT_METHODS | {"split"}:
            cls = self._cls_of(func.value)
            x_only = A.names_in(node.args[0]) if node.args else set()
            data = x_only | (A.names_in(node.args[1]) if len(node.args) > 1 else set())
            if (seg == "split" and cls in CV_SPLITTERS) or (seg in FIT_METHODS and cls in CV_SEARCH):
                self.splits.append(_Split(node.lineno, self.scope, data))
            elif seg in FIT_METHODS:
                kind: FitKind = "resample" if seg == "fit_resample" else "fit"
                narrow = kind == "resample" or cls in FIT_BEFORE_SPLIT
                selection = {n for k in SELECTION_KWARGS for n in A.names_in(A.kwarg(node, k))}
                receiver = func.value.id if isinstance(func.value, ast.Name) else None
                fit = _Fit(node.lineno, self.scope, cls, x_only if narrow else data, kind=kind)
                fit.receiver, fit.selection, fit.loop = receiver, selection, self.loop
                self.fits.append(fit)
        elif isinstance(func, ast.Attribute) and seg in EVAL_METHODS and node.args:
            receiver = func.value.id if isinstance(func.value, ast.Name) else f"<inline:{node.lineno}>"
            if self._cls_of(func.value) not in NOT_MODELS:
                data = {n for a in node.args[:2] for n in A.names_in(a)}
                self.evals.append(_Eval(node.lineno, self.scope, receiver, data, self.loop))
        self.generic_visit(node)

    def _split(self, node: ast.Call) -> None:
        arrays = [a for a in node.args if not isinstance(a, ast.Starred)]
        split = _Split(
            node.lineno, self.scope, set().union(*(A.names_in(a) for a in arrays)) if arrays else set()
        )
        target = self.targets.get(id(node))
        if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2 * len(arrays):
            for i, elt in enumerate(target.elts):
                names = A.target_names(elt)
                (split.train_out if i % 2 == 0 else split.test_out).update(names)
                for n in names:
                    self.deps.setdefault(n, set()).update(split.inputs)
        self.splits.append(split)


def _ctor(call: ast.Call) -> str | None:
    seg = A.last_segment(call)
    return seg if seg in NOT_MODELS else None


def _test_names(names: set[str]) -> set[str]:
    return {n for n in names if not VALIDATION_NAME.search(n) and not TRAIN_NAME.search(n)}


class LeakageDetector:
    name = "leakage"
    version = "0.2"

    def detect(self, src: SourceFile) -> list[Finding]:
        tree = src.tree
        if tree is None:
            return []
        c = _Collector()
        c.visit(tree)
        if not c.splits:
            return []
        memo: dict[str, set[str]] = {}

        def ancestors(names: set[str]) -> set[str]:
            out: set[str] = set()
            for n in names:
                if n not in memo:
                    seen, stack = {n}, [n]
                    while stack:
                        for d in c.deps.get(stack.pop(), ()):
                            if d not in seen:
                                seen.add(d)
                                stack.append(d)
                    memo[n] = seen
                out |= memo[n]
            return out

        def reaches(names: set[str], split: _Split) -> bool:
            return bool(names & (split.inputs | ancestors(split.inputs)))

        found: dict[tuple[int, str], Finding] = {}

        def add(f: Finding) -> None:
            found.setdefault((f.location.line_start or 0, f.type), f)

        for fit in sorted(c.fits, key=lambda f: f.line):
            if not fit.data:
                continue
            fit_anc = ancestors(fit.data)
            earlier = [s for s in c.splits if s.scope == fit.scope and s.line < fit.line]
            later = [s for s in c.splits if s.scope == fit.scope and s.line > fit.line]
            on_train = any(fit_anc & s.train_out for s in earlier)

            if fit.kind != "stat":
                on_test = [s for s in earlier if s.test_out and fit_anc & s.test_out]
                if on_test and not (fit_anc & on_test[-1].train_out and fit.data & on_test[-1].train_out):
                    add(self._test_in_training(src, fit, on_test[-1], fit.data & on_test[-1].test_out))
                    continue
            for s in earlier if fit.selection else []:
                used = _test_names(ancestors(fit.selection) & s.test_out)
                if used:
                    add(self._tuning(src, fit.line, s, "early stopping / validation", used))
            if on_train:
                continue  # fit on a training subset from an earlier split: correct
            leaked = next((s for s in later if reaches(fit.data, s)), None)
            if leaked is None:
                continue
            if fit.kind == "stat":
                if reaches(fit.writes, leaked):
                    add(self._before_split(src, fit, leaked))
            elif fit.kind == "resample" or fit.cls in FIT_BEFORE_SPLIT:
                add(self._before_split(src, fit, leaked))
            elif fit.receiver and _evaluated_on_test(c, fit, leaked, ancestors):
                add(self._model_on_all_data(src, fit, leaked))

        for f in self._multi_test(src, c, ancestors):
            add(f)
        return sorted(found.values(), key=lambda f: f.location.line_start or 0)

    def _multi_test(self, src: SourceFile, c: _Collector, ancestors: Ancestors) -> list[Finding]:
        out = []
        for s in c.splits:
            if not _test_names(s.test_out):
                continue
            on_test = [
                e
                for e in sorted(c.evals, key=lambda e: e.line)
                if e.scope == s.scope and e.line > s.line and _test_names(ancestors(e.data) & s.test_out)
            ]
            seen: set[str] = set()
            for e in on_test:
                loop_fit = e.loop is not None and any(f.loop == e.loop and f.kind == "fit" for f in c.fits)
                seen.add(e.receiver)
                if len(seen) >= 2 or loop_fit:
                    why = "inside a loop that refits the model" if loop_fit else "for a second model"
                    out.append(self._tuning(src, e.line, s, f"evaluation {why}", _test_names(s.test_out)))
                    break
        return out

    # -- findings
    def _before_split(self, src: SourceFile, fit: _Fit, split: _Split) -> Finding:
        if fit.kind == "resample":
            ftype, rule, sev = "data.leakage.resampling_before_split", "resampling", Severity.high
        elif fit.cls in SELECTORS:
            ftype, rule, sev = "data.leakage.feature_selection_full", "selection", Severity.high
        else:
            ftype, sev = "data.leakage.preprocessing_before_split", Severity.medium
            rule = "statistic" if fit.kind == "stat" else "preprocessing"
        noun = fit.cls or "resampler"
        if fit.kind == "resample":
            fix = "Split first, then resample only the training split (or use imblearn.pipeline.Pipeline)."
        else:
            fix = (
                f"Split first, compute {noun} on the training split only and apply it to the test split "
                "(or use an sklearn Pipeline/ColumnTransformer so cross-validation refits it per fold)."
            )
        return make_finding(
            self,
            src,
            type=ftype,
            title=f"{noun} computed on data before the train/test split",
            severity=sev,
            confidence=0.65 if fit.kind == "stat" else 0.8,
            line=fit.line,
            rule_id=f"leakage.{rule}_before_split",
            explanation=f"{noun} at line {fit.line} uses data that is split at line {split.line}, so it "
            "includes the test rows and evaluation is optimistic.",
            metrics={"split_line": split.line},
            fix=fix,
        )

    def _test_in_training(self, src: SourceFile, fit: _Fit, split: _Split, names: set[str]) -> Finding:
        what = fit.cls or "an estimator"
        return make_finding(
            self,
            src,
            type="data.leakage.test_in_training",
            title=f"{what} fit on test data",
            severity=Severity.high,
            confidence=0.8 if names else 0.6,
            line=fit.line,
            rule_id="leakage.fit_on_test",
            explanation=f"{what} is fit at line {fit.line} on data derived from the test side of the "
            f"split at line {split.line}; the model or its preprocessing has seen the test set.",
            metrics={"split_line": split.line},
            fix="Fit only on the training split; on the test split call transform/predict, never fit.",
        )

    def _model_on_all_data(self, src: SourceFile, fit: _Fit, split: _Split) -> Finding:
        return make_finding(
            self,
            src,
            type="data.leakage.test_in_training",
            title=f"{fit.receiver} trained on all data, then evaluated on the test split",
            severity=Severity.high,
            confidence=0.7,
            line=fit.line,
            rule_id="leakage.fit_all_then_eval_test",
            explanation=f"{fit.receiver} is fit at line {fit.line} on data that is split at line {split.line}, "
            "and later evaluated on the test side without being refit, so it has seen every test row.",
            metrics={"split_line": split.line},
            fix="Split first and fit only on the training split before evaluating on the test split.",
        )

    def _tuning(self, src: SourceFile, line: int, split: _Split, how: str, names: set[str]) -> Finding:
        return make_finding(
            self,
            src,
            type="data.leakage.tuning_on_test",
            title="Test split used for model selection",
            severity=Severity.medium,
            confidence=0.75,
            line=line,
            rule_id="leakage.test_for_selection",
            explanation=f"The test split from line {split.line} ({', '.join(sorted(names))}) is used for {how}; "
            "choices made on it make the final test score optimistic.",
            metrics={"split_line": split.line},
            fix="Split a validation set off the training data for early stopping and model selection; "
            "evaluate on the test split once, at the end.",
        )


def _evaluated_on_test(c: _Collector, fit: _Fit, split: _Split, ancestors: Ancestors) -> bool:
    """Is this fitted model later evaluated on the test side without being refit first?"""
    refits = [
        f for f in c.fits if f.receiver == fit.receiver and f.scope == fit.scope and f.line > split.line
    ]
    for e in c.evals:
        if e.receiver != fit.receiver or e.scope != fit.scope or e.line <= split.line:
            continue
        if ancestors(e.data) & split.test_out:
            return not any(r.line < e.line for r in refits)
    return False
