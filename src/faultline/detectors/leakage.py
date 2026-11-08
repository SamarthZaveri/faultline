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
from collections.abc import Callable, Iterable
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
        "SimpleImputer", "KNNImputer", "IterativeImputer", "TargetEncoder", "PolynomialFeatures",
        "TfidfVectorizer", "KBinsDiscretizer", "KMeans", "MiniBatchKMeans", "GaussianMixture", "FastICA",
        "FactorAnalysis", "NMF", "LatentDirichletAllocation", "Binarizer",
    }
)  # fmt: skip
# Encoders that only learn a vocabulary (category -> code): fitting them on all rows leaks nothing about
# feature values or the target, so they are not preprocessing leakage.
VOCAB_ONLY = frozenset(
    {
        "OneHotEncoder", "OrdinalEncoder", "CountVectorizer", "HashingVectorizer", "LabelEncoder",
        "LabelBinarizer", "MultiLabelBinarizer", "DictVectorizer",
    }
)  # fmt: skip
# Fit on held-out data by design: they measure a fitted model, they don't train one.
NOT_TRAINING = frozenset({"PermutationImportance", "permutation_importance", "TSNE"})
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
CV_SEARCH = frozenset(
    {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV", "BayesSearchCV"}
)
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
NOT_MODELS = FIT_BEFORE_SPLIT | VOCAB_ONLY | NOT_TRAINING | RESAMPLERS | CV_SEARCH | CV_SPLITTERS
FIT_METHODS = frozenset({"fit", "fit_transform", "fit_resample", "fit_sample"})
STAT_METHODS = frozenset({"mean", "median", "std", "var", "min", "max", "quantile", "mode", "value_counts"})
# groupby(...)[col].transform("mean"): a statistic over rows of other entities
TRANSFORM_STATS = frozenset({"mean", "median", "std", "var", "min", "max", "count", "size", "nunique", "sum"})
MISSINGNESS = frozenset({"isna", "isnull", "notna", "notnull"})  # missing shares choose columns, not values
# counts over groups: groupby(...).size() / .nunique() / .cumcount() (only after a groupby)
GROUP_STATS = frozenset({"size", "nunique", "cumcount", "count"})
# Early stopping / best-checkpoint selection is on when one of these is passed to the fit
EARLY_STOP_KWARGS = ("early_stopping_rounds", "callbacks", "use_best_model", "early_stopping")
BOOSTER_MODULES = frozenset({"xgb", "xgboost", "lgb", "lightgbm"})
METRIC_NAME = re.compile(r"(score|error|loss|rmse|rmsle|rmspe|mae|wmae|auc|accuracy)$", re.IGNORECASE)
BEST_NAME = re.compile(r"best|lowest|highest|min_|max_|top_", re.IGNORECASE)
TOY_DATA = re.compile(
    r"^make_(classification|regression|blobs|moons|circles|friedman\d|s_curve|swiss_roll|gaussian_quantiles"
    r"|hastie_10_2|multilabel_classification|low_rank_matrix|sparse_\w+)$"
    r"|^load_(iris|digits|wine|breast_cancer|diabetes|boston|linnerud)$"
)
EVAL_METHODS = frozenset(
    {"predict", "predict_proba", "predict_log_proba", "decision_function", "score", "evaluate"}
)
SELECTION_KWARGS = ("eval_set", "validation_data", "valid_sets", "eval_X")
MODULES = frozenset({"np", "numpy", "pd", "pandas", "math", "statistics", "torch", "tf", "scipy", "stats"})
# Names of a held-out split meant for validation (a legitimate early-stopping / selection set).
VALIDATION_NAME = re.compile(r"val|valid|dev", re.IGNORECASE)
DATA_WRAPPERS = frozenset({"DMatrix", "Pool", "Dataset", "DataFrame"})  # same rows, another container
TRAIN_NAME = re.compile(r"train", re.IGNORECASE)
HELD_OUT_NAME = re.compile(r"test|val|valid|dev|hold", re.IGNORECASE)

FitKind = Literal["fit", "stat", "resample"]
Ancestors = Callable[..., set[str]]  # (names, at_line=end) -> names they were computed from


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
    early_stop: bool = False  # early stopping / best checkpoint on the selection data
    quiet: bool = False  # verbose=False / verbose_eval=0: the eval scores aren't printed
    outputs: set[str] = field(default_factory=set)  # names the transformed data is written into
    transformer: bool = False  # a preprocessing step (incl. user classes with fit_transform)


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
    via_call: bool = False  # a split inside a function, placed at the line that calls it
    how: str = ""  # how a holdout was cut without a splitter: sample | index | mask | pos | cmp
    seed: str | None = None  # train_test_split(random_state=0, test_size=.3): same seed, same partition
    subset_of: str | None = None  # train = train[n:m]: one side overwrites the frame it was cut from


class _Collector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scopes = [0]
        self.n_scopes = 0
        self.loops: list[int] = []
        self.n_loops = 0
        self.deps: dict[str, list[tuple[int, set[str]]]] = {}  # name -> (line, sources) per assignment
        self.objects: dict[tuple[int, str], str] = {}  # (scope, name) -> estimator class
        self.targets: dict[int, ast.expr] = {}  # id(call) -> assignment target
        self.slices: dict[tuple[int, str, str, str], list[str]] = {}  # one side of a slice -> its names
        self.slice_splits: dict[tuple[int, str, str], tuple[_Split, str]] = {}  # key -> (split, train side)
        self.param_deps: dict[str, list[tuple[int, set[str]]]] = {}  # parameter -> (call line, arguments)
        self.loads: dict[str, list[int]] = {}  # name -> lines where it is read
        self.index_names: set[str] = set()  # idx_tr / idx_va used to cut a holdout
        self.saved: dict[str, set[str]] = {}  # file basename -> names written to it (df.to_csv('x.csv'))
        self.sweep_loops: set[int] = set()  # for d in depths: Model(max_depth=d): a hyperparameter sweep
        self.one_split: set[str] = set()  # sss = StratifiedShuffleSplit(n_splits=1): a single holdout
        self.holdout_idx: list[tuple[int, dict[str, str]]] = []  # loop line, index name -> head/tail
        self.sampled: dict[str, tuple[set[str], float | None]] = {}  # val = df.sample(...): frame, frac
        self.fits: list[_Fit] = []
        self.evals: list[_Eval] = []
        self.splits: list[_Split] = []
        self.group_keys: frozenset[str] = frozenset()
        self.aliases: dict[str, set[str]] = {}  # loop variable -> the frames it stands for
        self.lists: dict[str, set[str]] = {}  # name = [a, b]
        self.call_targets: dict[int, set[str]] = {}  # id(call) -> names its assignment writes
        self.transforms: list[tuple[int, str, int, set[str]]] = []  # (scope, receiver, line, outputs)
        self.func_scopes: dict[int, tuple[str, list[str]]] = {}  # scope -> (function name, params)
        self.func_calls: dict[str, list[tuple[int, int]]] = {}  # function name -> (line, scope) of calls
        self.assigns: list[tuple[str, int, set[str]]] = []  # (name, line, sources), in order
        self.split_lines: set[int] = set()  # lines whose assignment makes (one side of) a split
        self.loop_metrics: list[tuple[int, int, set[str]]] = []  # (line, loop, data) metric calls in loops
        self.best_loops: set[int] = set()  # loops that keep a best score
        self.loop_appends: dict[int, set[str]] = {}  # loop -> lists it appends to
        self.picked: set[str] = set()  # names passed to argmin/argmax/min/max/idxmin
        self.importances: list[tuple[int, str, set[str]]] = []  # (line, model, names written)
        self.toy: set[str] = set()  # names holding toy datasets (make_classification, load_iris, ...)
        self.classes: dict[tuple[int, str], str] = {}  # (scope, name) -> constructor of any object
        self.chained: dict[int, set[str]] = {}  # id(fit call) -> outputs of .fit(...).transform(...)
        self.raw_calls: list[tuple[str, int, int, ast.Call]] = []  # calls to plain names, for user functions
        self.model_pipelines: set[str] = set()  # names bound to make_pipeline(..., <model>)
        self.groupers: dict[str, set[str]] = {}  # g = df.groupby('key'): name -> keys
        self.returns: dict[int, set[str]] = {}  # function scope -> names it returns

    def dep(self, name: str, sources: set[str], line: int) -> None:
        self.deps.setdefault(name, []).append((line, set(sources)))

    @property
    def scope(self) -> int:
        return self.scopes[-1]

    @property
    def loop(self) -> int | None:
        return self.loops[-1] if self.loops else None

    def _enter(self, node: ast.AST) -> None:
        self.n_scopes += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
            self.func_scopes[self.n_scopes] = (node.name, params)
            defaults = node.args.defaults
            for a, d in zip(node.args.args[len(node.args.args) - len(defaults) :], defaults, strict=True):
                self.dep(a.arg, A.names_in(d), 0)
        self.scopes.append(self.n_scopes)
        self.generic_visit(node)
        self.scopes.pop()

    visit_FunctionDef = _enter
    visit_AsyncFunctionDef = _enter

    def _loop(self, node: ast.AST) -> None:
        self.n_loops += 1
        self.loops.append(self.n_loops)
        alias: tuple[str, set[str]] | None = None
        if isinstance(node, (ast.For, ast.AsyncFor)):
            for name in A.target_names(node.target):
                self.dep(name, A.names_in(node.iter), node.lineno)
        if isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name):
            it = node.iter
            frames = (
                {n for e in it.elts for n in A.target_names(e)}
                if isinstance(it, (ast.List, ast.Tuple))
                else self.lists.get(it.id, set())
                if isinstance(it, ast.Name)
                else set()
            )
            if frames:
                alias = (node.target.id, frames)
                self.aliases[alias[0]] = frames
        holdout = isinstance(node, (ast.For, ast.AsyncFor)) and self._one_split_loop(node)
        if isinstance(node, (ast.For, ast.AsyncFor)) and _sweeps(node):
            self.sweep_loops.add(self.n_loops)
        if holdout:
            self.holdout_idx.append(holdout)
        for n in ast.walk(node):
            if isinstance(n, (ast.Assign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(BEST_NAME.search(t) for tt in targets for t in A.target_names(tt)):
                    self.best_loops.add(self.n_loops)
            elif isinstance(n, ast.Call) and A.last_segment(n) in ("argmin", "argmax", "idxmin", "idxmax"):
                self.best_loops.add(self.n_loops)
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "append"
                and isinstance(n.func.value, ast.Name)
            ):
                self.loop_appends.setdefault(self.n_loops, set()).add(n.func.value.id)
        self.generic_visit(node)
        if holdout:
            self.holdout_idx.pop()
        if alias is not None:
            self.aliases.pop(alias[0], None)
        self.loops.pop()

    def _one_split_loop(self, node: ast.For | ast.AsyncFor) -> tuple[int, dict[str, str]] | None:
        """for tr, te in StratifiedShuffleSplit(n_splits=1).split(X, y): runs once, a single holdout."""
        it = node.iter
        if not (isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2):
            return None
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr == "split"):
            return None
        splitter = it.func.value
        if isinstance(splitter, ast.Name):
            single = splitter.id in self.one_split
        else:
            single = isinstance(splitter, ast.Call) and _one_split(splitter)
        tr, te = node.target.elts
        if not (single and isinstance(tr, ast.Name) and isinstance(te, ast.Name)):
            return None
        return node.lineno, {tr.id: "head", te.id: "tail"}

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
                sources = A.names_in(vv) | self._reloaded(vv)
                for name in A.target_names(tt):
                    partial = isinstance(
                        tt, (ast.Subscript, ast.Attribute)
                    )  # df['c'] = ...: df keeps its rows
                    self.dep(name, sources | {name} if partial else sources, line)
                    written.add(name)
                    self.assigns.append((name, line, sources))
                    for frame in self.aliases.get(name, ()):  # `for df in [train, test]: df[...] = ...`
                        self.dep(frame, sources, line)
                        written.add(frame)
                    if isinstance(vv, (ast.List, ast.Tuple)) and all(
                        isinstance(e, ast.Name) for e in vv.elts
                    ):
                        self.lists[name] = {e.id for e in vv.elts if isinstance(e, ast.Name)}
                    if isinstance(vv, ast.Call) and TOY_DATA.match(A.last_segment(vv) or ""):
                        self.toy.add(name)
        for n in ast.walk(value):
            if isinstance(n, ast.Call):
                self.call_targets[id(n)] = written
        if isinstance(value, ast.Call):
            self.targets[id(value)] = targets[0]
            cls = _ctor(value)
            if cls and len(targets) == 1 and isinstance(targets[0], ast.Name):
                self.objects[(self.scope, targets[0].id)] = cls
                if _one_split(value):
                    self.one_split.add(targets[0].id)
            if len(targets) == 1 and isinstance(targets[0], ast.Name) and (seg := A.last_segment(value)):
                self.classes[(self.scope, targets[0].id)] = seg
                if seg in ("make_pipeline", "Pipeline") and _ends_in_model(value):
                    self.model_pipelines.add(targets[0].id)
                if seg == "groupby":
                    self.groupers[targets[0].id] = _group_keys(value)
        self._importances(value, line, written)
        self._pandas_resampling(value, line)
        for t in targets:
            if isinstance(t, ast.Tuple) and isinstance(value, ast.Tuple) and len(t.elts) == len(value.elts):
                for tt, vv in zip(t.elts, value.elts, strict=True):
                    self._slice(tt, vv, line)
            else:
                self._slice(t, value, line)
                self._sample_split(t, value, line)
        self._stats(value, written, line)

    def _slice(self, target: ast.expr, value: ast.expr, line: int) -> None:
        """Complementary pieces of one frame: data[:n] / data[n:], df[mask] / df[~mask],
        df[df.t < c] / df[df.t >= c]. The first side seen waits for its complement."""
        if not isinstance(target, ast.Name):
            return
        value = _unwrap(value)
        if isinstance(value, ast.Call) and A.last_segment(value) in DATA_WRAPPERS and value.args:
            value = _unwrap(value.args[0])  # xgb.DMatrix(train[n:], ...): the same rows
        if not isinstance(value, ast.Subscript):
            return
        base = value.value
        if isinstance(base, ast.Attribute) and base.attr in ("loc", "iloc"):
            base = base.value
        sl = value.slice.elts[0] if isinstance(value.slice, ast.Tuple) and value.slice.elts else value.slice
        if self.holdout_idx and isinstance(sl, ast.Name) and sl.id in self.holdout_idx[-1][1]:
            loop_line, roles = self.holdout_idx[-1]
            split = next((s for s in reversed(self.splits) if s.line == loop_line), None)
            if split is not None:
                (split.train_out if roles[sl.id] == "head" else split.test_out).add(target.id)
                self.split_lines.add(line)
            return
        side: str | None = None
        kind, key_node = "", sl
        if isinstance(sl, ast.Slice) and sl.step is None:
            if sl.lower is None and sl.upper is not None:
                side, kind, key_node = "head", "pos", sl.upper
            elif sl.lower is not None:  # data[n:] or data[n:m] (a validation window after the head)
                side, kind, key_node = "tail", "pos", sl.lower
        elif isinstance(sl, ast.Name) and self.loop is None and (role := _index_role(sl.id)):
            side, kind, key_node = role, "idx", ast.Constant(None)  # data.iloc[idx_tr] / data.iloc[idx_va]
            self.index_names.add(sl.id)
        elif isinstance(sl, ast.Name):
            side, kind, key_node = "head", "mask", sl
        elif (
            isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.Invert) and isinstance(sl.operand, ast.Name)
        ):
            side, kind, key_node = "tail", "mask", sl.operand
        elif isinstance(sl, ast.Compare) and len(sl.ops) == 1:
            op = sl.ops[0]
            if isinstance(op, (ast.Lt, ast.LtE)):
                side = "head"
            elif isinstance(op, (ast.Gt, ast.GtE)):
                side = "tail"
            kind, key_node = "cmp", ast.Tuple(elts=[sl.left, sl.comparators[0]], ctx=ast.Load())
        if side is None or (kind == "pos" and _frame_length(key_node)):
            return  # df[:len(train)] / df[ntrain:]: train and test put back apart, not a holdout
        # a mask picks the same rows from every frame it indexes: X[mask], y.values[mask]
        key = (self.scope, "" if kind == "mask" else ast.dump(base), kind + ast.dump(key_node))
        self.split_lines.add(line)
        if key in self.slice_splits:
            split, train_side = self.slice_splits[key]
            (split.train_out if side == train_side else split.test_out).add(target.id)
            return
        other = "tail" if side == "head" else "head"
        match = self.slices.get((*key, other))
        if match is None:
            self.slices.setdefault((*key, side), []).append(target.id)
            return
        mine, theirs = [target.id, *self.slices.get((*key, side), [])], match
        heads, tails = (mine, theirs) if side == "head" else (theirs, mine)
        train_side = "head"
        trainish_tails = any(TRAIN_NAME.search(t) for t in tails) and not any(
            TRAIN_NAME.search(h) for h in heads
        )
        held_heads = any(HELD_OUT_NAME.search(h) for h in heads) and not any(
            HELD_OUT_NAME.search(t) for t in tails
        )
        if trainish_tails or held_heads:
            heads, tails, train_side = tails, heads, "tail"  # e.g. y_valid = y[:n]; y_train = y[n:]
        split = _Split(line, self.scope, A.names_in(base), set(heads), set(tails), how=kind or "slice")
        overwritten = set(heads + tails) & A.names_in(base)
        split.subset_of = next(iter(overwritten)) if len(overwritten) == 1 else None
        self.splits.append(split)
        self.slice_splits[key] = (split, train_side)

    def _importances(self, value: ast.expr, line: int, written: set[str]) -> None:
        """Columns chosen from a fitted model's importances (feature_importances_, get_fscore, SHAP)."""
        models: set[str] = set()
        for n in ast.walk(value):
            if isinstance(n, ast.Attribute) and n.attr in ("feature_importances_", "coef_"):
                models |= A.names_in(n.value)
            elif (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("get_fscore", "get_score", "feature_importance", "shap_values")
            ):
                models |= A.names_in(n.func.value)
        if models:
            self.importances.append((line, ",".join(sorted(models)), written))

    def _sample_split(self, target: ast.expr, value: ast.expr, line: int) -> None:
        """val = df.sample(n=...); train = df.drop(val.index): a random holdout."""
        if not (isinstance(target, ast.Name) and isinstance(value, ast.Call)):
            return
        func = value.func
        if not isinstance(func, ast.Attribute):
            return
        if func.attr == "sample":
            frac = A.kwarg(value, "frac")
            share = frac.value if isinstance(frac, ast.Constant) and isinstance(frac.value, float) else None
            self.sampled[target.id] = (A.names_in(func.value), share)
            return
        if func.attr != "drop" or not value.args:
            return
        idx = value.args[0]
        if not (isinstance(idx, ast.Attribute) and idx.attr == "index" and isinstance(idx.value, ast.Name)):
            return
        picked = idx.value.id
        if picked not in self.sampled or not (self.sampled[picked][0] & A.names_in(func.value)):
            return
        share = self.sampled[picked][1]
        picked_is_train = TRAIN_NAME.search(picked) is not None or (
            not HELD_OUT_NAME.search(picked) and share is not None and share >= 0.5
        )
        train, test = (picked, target.id) if picked_is_train else (target.id, picked)
        self.splits.append(_Split(line, self.scope, self.sampled[picked][0], {train}, {test}, how="sample"))
        self.split_lines.add(line)

    def _pandas_resampling(self, value: ast.expr, line: int) -> None:
        """df.drop(df[...].sample(...).index) / concat([minority, majority.sample(n)]) / concat([x, x])."""
        for n in ast.walk(value):
            if not isinstance(n, ast.Call):
                continue
            seg = A.last_segment(n)
            if seg == "concat" and n.args and isinstance(n.args[0], (ast.List, ast.Tuple)):
                parts = n.args[0].elts
                names = [e.id for e in parts if isinstance(e, ast.Name)]
                sampled = any(
                    isinstance(c, ast.Call) and A.last_segment(c) in ("sample", "resample")
                    for e in parts
                    for c in ast.walk(e)
                )
                if sampled or len(names) != len(set(names)):
                    data = A.names_in(n) - MODULES
                    self.fits.append(_Fit(line, self.scope, "resampling", data, kind="resample"))
            elif seg == "drop" and any(
                isinstance(c, ast.Call) and A.last_segment(c) == "sample" for a in n.args for c in ast.walk(a)
            ):
                data = A.names_in(n) - MODULES
                self.fits.append(_Fit(line, self.scope, "undersampling", data, kind="resample"))

    def _stats(self, value: ast.expr, written: set[str], line: int) -> None:
        if not written:
            return
        for n in ast.walk(value):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
                continue
            attr = n.func.attr
            is_stat = (
                attr in STAT_METHODS
                or (attr == "transform" and A.const_str(A.arg(n, 0, "func")) in TRANSFORM_STATS)
                or (attr in GROUP_STATS and _chain_has(n.func.value, frozenset({"groupby"})))
            )
            if _row_wise(n):
                continue  # df[cols].mean(axis=1): a per-row feature, not a statistic over rows
            if attr in ("min", "max") and _is_datetime(n.func.value):
                continue  # days since df.date.min(): a constant shift, nothing learned from the rows
            if not is_stat or self._per_entity(n.func.value) or _chain_has(n.func.value, MISSINGNESS):
                continue
            data = (A.names_in(n.func.value) | {a for x in n.args for a in A.names_in(x)}) - MODULES
            if data:
                stat = _Fit(line, self.scope, f"{attr}()", data, kind="stat", writes=written)
                self.fits.append(stat)

    def _per_entity(self, receiver: ast.expr) -> bool:
        """groupby(<group key>)...: aggregates over one entity's own rows."""
        if not self.group_keys:
            return False
        for n in ast.walk(receiver):
            keys: set[str] = set()
            if isinstance(n, ast.Call) and A.last_segment(n) == "groupby":
                keys = _group_keys(n)
            elif isinstance(n, ast.Name) and n.id in self.groupers:
                keys = self.groupers[n.id]
            if keys and keys <= self.group_keys:
                return True
        return False

    def _reloaded(self, value: ast.expr) -> set[str]:
        """pd.read_csv('../input/x/feats.csv') after feats.to_csv('feats.csv'): the frame that was saved."""
        out: set[str] = set()
        for n in ast.walk(value):
            if isinstance(n, ast.Call) and (A.last_segment(n) or "").startswith("read_"):
                out |= self.saved.get(
                    _basename(A.const_str(A.arg(n, 0, "filepath_or_buffer", "path"))), set()
                )
        return out

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loads.setdefault(node.id, []).append(node.lineno)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and self.scope in self.func_scopes:
            self.returns.setdefault(self.scope, set()).update(A.names_in(node.value))
        self.generic_visit(node)

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
        saves = seg in ("to_csv", "to_pickle", "to_parquet", "to_feather") and isinstance(func, ast.Attribute)
        if (
            saves
            and isinstance(func, ast.Attribute)
            and (name := _basename(A.const_str(A.arg(node, 0, "path_or_buf", "path"))))
        ):
            self.saved.setdefault(name, set()).update(A.names_in(func.value))
        if isinstance(func, ast.Name):
            self.raw_calls.append((func.id, node.lineno, self.scope, node))
        if seg == "train_test_split":
            self._split(node)
        elif seg in CV_FUNCS and len(node.args) > 1:
            self.splits.append(_Split(node.lineno, self.scope, A.names_in(node.args[1])))
        elif (
            seg == "train"
            and isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in BOOSTER_MODULES
        ):
            self._booster(node)
        if isinstance(func, ast.Attribute) and seg in FIT_METHODS | EVAL_METHODS and node.args:
            arg = node.args[0]  # net.fit(X[:n], y[:n]) ... net.predict(X[n:]): a holdout cut inline
            if isinstance(arg, ast.Subscript):
                self._slice(ast.Name(id=f"<{seg}:{node.lineno}>", ctx=ast.Store()), arg, node.lineno)
        if isinstance(func, ast.Attribute) and seg in FIT_METHODS | {"split", "fit_generator"}:
            cls = self._cls_of(func.value)
            x_arg, y_arg = A.arg(node, 0, "X", "x"), A.arg(node, 1, "y")
            x_only = A.names_in(x_arg) if x_arg is not None else set()
            data = x_only | (A.names_in(y_arg) if y_arg is not None else set())
            if (seg == "split" and cls in CV_SPLITTERS) or (seg in FIT_METHODS and cls in CV_SEARCH):
                self.splits.append(_Split(node.lineno, self.scope, data))
            elif seg != "split" and cls not in VOCAB_ONLY | NOT_TRAINING and not _encoder_slot(func.value):
                kind: FitKind = "resample" if seg in ("fit_resample", "fit_sample") else "fit"
                in_model_pipe = isinstance(func.value, ast.Name) and func.value.id in self.model_pipelines
                transformer = not in_model_pipe and (
                    cls in FIT_BEFORE_SPLIT or (seg == "fit_transform" and cls not in RESAMPLERS)
                )
                narrow = kind == "resample" or transformer
                selection = {n for k in SELECTION_KWARGS for n in A.names_in(A.kwarg(node, k))}
                receiver = func.value.id if isinstance(func.value, ast.Name) else None
                target = self.targets.get(id(node))
                if receiver is None and isinstance(target, ast.Name):
                    receiver = target.id  # m = RandomForestClassifier().fit(X, y)
                fit = _Fit(node.lineno, self.scope, cls, x_only if narrow else data, kind=kind)
                fit.receiver, fit.selection, fit.loop = receiver, selection, self.loop
                fit.transformer = transformer
                fit.early_stop = self._early_stop(node, func.value)
                fit.quiet = _quiet(node)
                if seg == "fit_transform":
                    fit.outputs = set(self.call_targets.get(id(node), set()))
                elif id(node) in self.chained:
                    fit.outputs = set(self.chained[id(node)])
                self.fits.append(fit)
        elif isinstance(func, ast.Attribute) and seg == "transform":
            outputs = set(self.call_targets.get(id(node), set()))
            if isinstance(func.value, ast.Name):
                self.transforms.append((self.scope, func.value.id, node.lineno, outputs))
            elif isinstance(func.value, ast.Call) and A.last_segment(func.value) == "fit":
                self.chained[id(func.value)] = outputs
        elif isinstance(func, ast.Attribute) and seg in EVAL_METHODS and node.args:
            receiver = func.value.id if isinstance(func.value, ast.Name) else f"<inline:{node.lineno}>"
            in_model_pipe = isinstance(func.value, ast.Name) and func.value.id in self.model_pipelines
            if in_model_pipe or self._cls_of(func.value) not in NOT_MODELS:
                data = {n for a in node.args[:2] for n in A.names_in(a)}
                self.evals.append(_Eval(node.lineno, self.scope, receiver, data, self.loop))
        if seg in ("argmin", "argmax", "idxmin", "idxmax", "min", "max", "nsmallest", "nlargest"):
            self.picked |= {n for a in node.args for n in A.names_in(a)} | (
                A.names_in(func.value) if isinstance(func, ast.Attribute) else set()
            )
        if (
            self.loop is not None
            and seg
            and METRIC_NAME.search(seg)
            and seg not in EVAL_METHODS
            and node.args
        ):
            data = {n for a in node.args[:2] for n in A.names_in(a)}
            self.loop_metrics.append((node.lineno, self.loop, data))
        self.generic_visit(node)

    def _early_stop(self, call: ast.Call, receiver: ast.expr) -> bool:
        if any(_set(A.kwarg(call, k)) for k in EARLY_STOP_KWARGS):
            return True
        cls = self.classes.get((self.scope, receiver.id)) if isinstance(receiver, ast.Name) else None
        if isinstance(receiver, ast.Name) and cls is None:
            cls = self.classes.get((0, receiver.id))
        return bool(cls and cls.startswith("CatBoost"))  # CatBoost keeps the best iteration on eval_set

    def _booster(self, node: ast.Call) -> None:
        """xgb.train(params, dtrain, ..., evals=[(dvalid, 'v')]) / lgb.train(..., valid_sets=[...])."""
        dtrain = A.arg(node, 1, "dtrain", "train_set")
        evals = A.kwarg(node, "evals") or A.kwarg(node, "valid_sets")
        if evals is None and len(node.args) > 3:
            evals = node.args[3]
        target = self.targets.get(id(node))
        fit = _Fit(node.lineno, self.scope, "booster", A.names_in(dtrain))
        fit.selection = A.names_in(evals) - A.names_in(dtrain)
        fit.receiver = target.id if isinstance(target, ast.Name) else None
        fit.loop = self.loop
        fit.early_stop = any(_set(A.kwarg(node, k)) for k in EARLY_STOP_KWARGS)
        fit.quiet = _quiet(node)
        self.fits.append(fit)

    def _descendants(self, names: set[str]) -> set[str]:
        """Names computed (directly or not) from these, ignoring line order."""
        out = set(names)
        changed = True
        while changed:
            changed = False
            for name, assigns in self.deps.items():
                if name not in out and any(src & out for _, src in assigns):
                    out.add(name)
                    changed = True
        return out

    def finish(self) -> None:
        """After the walk: tie calls of user functions to the splits inside them."""
        by_name = {name: (scope, params) for scope, (name, params) in self.func_scopes.items()}
        for name, line, scope, call in self.raw_calls:
            if name not in by_name:
                continue
            self.func_calls.setdefault(name, []).append((line, scope))
            _, params = by_name[name]
            for i, a in enumerate(call.args):
                if i < len(params):
                    self.param_deps.setdefault(params[i], []).append((line, A.names_in(a)))
            for kw in call.keywords:
                if kw.arg in params:
                    self.param_deps.setdefault(kw.arg, []).append((line, A.names_in(kw.value)))
        for scope, names in self.returns.items():
            fname = self.func_scopes[scope][0]
            self.dep(fname, names, 0)  # x = f(df): x comes from what f returns
        calls_by_func: dict[str, list[tuple[int, int, ast.Call]]] = {}
        for name, line, scope, call in self.raw_calls:
            if name in by_name:
                calls_by_func.setdefault(name, []).append((line, scope, call))
        for fit in list(self.fits):
            if fit.scope not in self.func_scopes or fit.kind not in ("stat", "fit") or fit.loop is not None:
                continue
            if fit.kind == "fit" and not fit.transformer:
                continue
            fname, params = self.func_scopes[fit.scope]
            produced = fit.writes | fit.outputs
            returned = self.returns.get(fit.scope, set())
            to_return = bool(produced & returned) or bool(returned & self._descendants(produced))
            to_param = bool(produced & set(params))
            if not (to_return or to_param):
                continue  # used inside the function only (dtype choice, a plot): nothing reaches the data
            for line, scope, call in calls_by_func.get(fname, []):
                args = {n for a in [*call.args, *(k.value for k in call.keywords)] for n in A.names_in(a)}
                # fit on a parameter, on the call's arguments, or on data the function loads and returns
                if not (fit.data & set(params)) and not (fit.data & args) and not to_return:
                    continue
                written = set(self.call_targets.get(id(call), set())) if to_return else set()
                if to_param:
                    written |= args  # f(df) writes into df
                if not written:
                    continue
                rows = fit.data | args
                if not (fit.data & set(params)) and not (fit.data & args):
                    rows |= written  # X = load(): what the loader returns are the rows it fit on
                proj = _Fit(line, scope, fit.cls, rows, kind=fit.kind, writes=fit.writes | written)
                proj.transformer, proj.outputs = fit.transformer, written if fit.transformer else set()
                self.fits.append(proj)
        for s in list(self.splits):
            if s.scope in self.func_scopes and not s.via_call:
                for line, scope in self.func_calls.get(self.func_scopes[s.scope][0], []):
                    self.splits.append(
                        _Split(line, scope, s.inputs, set(s.train_out), set(s.test_out), via_call=True)
                    )
        for loop, lists in self.loop_appends.items():
            if lists & self.picked:
                self.best_loops.add(loop)
        for scope, recv, line, outputs in self.transforms:
            for fit in self.fits:
                if fit.receiver == recv and fit.scope == scope and fit.line <= line:
                    fit.outputs |= outputs

    def _split(self, node: ast.Call) -> None:
        self.split_lines.add(node.lineno)
        arrays = [a for a in node.args if not isinstance(a, ast.Starred)]
        split = _Split(
            node.lineno, self.scope, set().union(*(A.names_in(a) for a in arrays)) if arrays else set()
        )
        state = A.kwarg(node, "random_state")
        if isinstance(state, ast.Constant):
            size = A.kwarg(node, "test_size") or A.kwarg(node, "train_size")
            split.seed = f"{state.value!r}|{ast.dump(size) if size is not None else ''}"
        target = self.targets.get(id(node))
        if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2 * len(arrays):
            for i, elt in enumerate(target.elts):
                names = A.target_names(elt)
                (split.train_out if i % 2 == 0 else split.test_out).update(names)
                for n in names:
                    self.dep(n, split.inputs, node.lineno)
        self.splits.append(split)


def _quiet(call: ast.Call) -> bool:
    """verbose=False / verbose_eval=0 / silent=True: the per-round eval scores aren't printed."""
    for k in ("verbose", "verbose_eval"):
        v = A.kwarg(call, k)
        if isinstance(v, ast.Constant) and v.value in (False, 0, None) and v.value is not True:
            return True
    return False


def _set(node: ast.expr | None) -> bool:
    """A keyword passed with a real value (not None / False / 0 / [])."""
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, (ast.List, ast.Tuple)):
        return bool(node.elts)
    return True


def _const_strs(node: ast.expr | None) -> list[str]:
    if node is None:
        return []
    if (v := A.const_str(node)) is not None:
        return [v]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [v for e in node.elts if (v := A.const_str(e)) is not None]
    return []


def _group_keys(call: ast.Call) -> set[str]:
    """groupby('a') / groupby(['a', 'b']) / groupby(df['a'])."""
    by = A.arg(call, 0, "by")
    keys = set(_const_strs(by))
    if by is not None and not keys:
        keys = {k for n in ast.walk(by) if isinstance(n, ast.Subscript) for k in _const_strs(n.slice)}
    return keys


def _encoder_slot(receiver: ast.expr) -> bool:
    """encoders[col].fit_transform(...) / lbl_enc[c].fit(...): a dict of label encoders."""
    return isinstance(receiver, ast.Subscript) and bool(
        re.search(r"enc|lbl|label|le$", A.dotted(receiver.value) or "", re.IGNORECASE)
    )


def _ends_in_model(call: ast.Call) -> bool:
    """make_pipeline(scaler, model) / Pipeline([..., ('m', model)]): fitting it trains a model."""
    ctors = [A.last_segment(n) for n in ast.walk(call) if isinstance(n, ast.Call) and n is not call]
    ctors = [c for c in ctors if c and c[:1].isupper()]
    return bool(ctors) and ctors[-1] not in FIT_BEFORE_SPLIT | VOCAB_ONLY


def _is_datetime(node: ast.expr) -> bool:
    """df['Date'].dt.date / pd.to_datetime(...) / df['date']: dates, by accessor or column name."""
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr == "dt":
            return True
        if isinstance(n, ast.Call) and A.last_segment(n) == "to_datetime":
            return True
    key = node.slice if isinstance(node, ast.Subscript) else None
    return bool(re.search(r"date|time", A.const_str(key) or "", re.IGNORECASE)) if key is not None else False


def _row_wise(call: ast.Call) -> bool:
    axis = A.kwarg(call, "axis")
    if axis is None and len(call.args) >= 2 and A.last_segment(call) in ("max", "min", "mean", "std", "sum"):
        axis = call.args[1]  # np.max(x, 1)
    return isinstance(axis, ast.Constant) and axis.value in (1, "columns")


def _chain_has(node: ast.expr, methods: frozenset[str]) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in methods
        for n in ast.walk(node)
    )


def _sweeps(node: ast.For | ast.AsyncFor) -> bool:
    """The loop variable is a hyperparameter: Model(n_estimators=n) / m.set_params(max_depth=d)."""
    if not isinstance(node.target, ast.Name):
        return False
    var = node.target.id
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        seg = A.last_segment(n) or ""
        if not (seg == "set_params" or seg[:1].isupper()) or seg in NOT_MODELS:
            continue
        if any(kw.arg and isinstance(kw.value, ast.Name) and kw.value.id == var for kw in n.keywords):
            return True
    return False


def _basename(path: str | None) -> str:
    return re.split(r"[\\/]", path)[-1] if path else ""


def _one_split(call: ast.Call) -> bool:
    """ShuffleSplit(n_splits=1) / StratifiedShuffleSplit(1, test_size=0.2)."""
    if A.last_segment(call) not in ("ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit"):
        return False
    n = A.arg(call, 0, "n_splits")
    return isinstance(n, ast.Constant) and n.value == 1


def _frame_length(node: ast.expr) -> bool:
    """len(train) / y_train.shape[0] / ntrain: where a concatenated train+test frame splits."""
    if isinstance(node, ast.Call) and A.last_segment(node) == "len":
        return True
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "shape"
    ):
        return True
    return isinstance(node, ast.Name) and bool(
        re.fullmatch(r"n_?train|len_?train|train_?len", node.id, re.IGNORECASE)
    )


def _index_role(name: str) -> str | None:
    """idx_tr / train_index -> head (training rows); idx_va / val_idx -> tail (held-out rows)."""
    if not re.search(r"idx|index|ind$|rows", name, re.IGNORECASE):
        return None
    if re.search(r"(^|_)tr(_|$)|train", name, re.IGNORECASE):
        return "head"
    if re.search(r"(^|_)(va|te)(_|$)", name, re.IGNORECASE) or HELD_OUT_NAME.search(name):
        return "tail"
    return None


def _unwrap(node: ast.expr) -> ast.expr:
    """df[...].values / .to_numpy() / .copy() / .reset_index(): the same rows."""
    while True:
        if isinstance(node, ast.Attribute) and node.attr == "values":
            node = node.value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("to_numpy", "copy", "reset_index", "astype")
        ):
            node = node.func.value
        else:
            return node


def _ctor(call: ast.Call) -> str | None:
    seg = A.last_segment(call)
    return seg if seg in NOT_MODELS else None


def _test_names(names: set[str]) -> set[str]:
    return {n for n in names if not VALIDATION_NAME.search(n) and not TRAIN_NAME.search(n)}


def random_holdouts(src: SourceFile) -> list[int]:
    """Lines of single random holdouts cut by hand: df.sample + drop, data.iloc[idx_tr] / [idx_va]."""
    tree = src.tree
    if tree is None:
        return []
    c = _Collector()
    c.visit(tree)
    by_splitter = any(  # train_ind, test_ind = next(StratifiedShuffleSplit(...).split(X, y))
        isinstance(n, ast.Call) and A.last_segment(n) == "split" and names & c.index_names
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.For))
        for names in [{t for tt in _targets_of(node) for t in A.target_names(tt)}]
        for n in ast.walk(node.value if isinstance(node, ast.Assign) else node.iter)
    )
    return sorted(
        s.line for s in c.splits if s.test_out and (s.how == "sample" or (s.how == "idx" and not by_splitter))
    )


def _targets_of(node: ast.Assign | ast.For) -> list[ast.expr]:
    return node.targets if isinstance(node, ast.Assign) else [node.target]


class LeakageDetector:
    name = "leakage"
    version = "0.3"

    def __init__(self, group_keys: Iterable[str] = ()) -> None:
        # entity keys of the dataset (from a profile): per-entity aggregates aren't cross-row statistics
        self.group_keys = frozenset(group_keys)

    def detect(self, src: SourceFile) -> list[Finding]:
        tree = src.tree
        if tree is None:
            return []
        c = _Collector()
        c.group_keys = self.group_keys
        c.visit(tree)
        c.finish()
        memo: dict[tuple[str, int, frozenset[str]], set[str]] = {}

        def ancestors(names: set[str], at: int = 10**9, block: frozenset[str] = frozenset()) -> set[str]:
            """Names these were computed from, following only assignments made before line `at`
            (and never through a name in `block`)."""
            out: set[str] = set()
            for n in names:
                key = (n, at, block)
                if key not in memo:
                    seen, stack, visited = {n}, [(n, at)], set()
                    while stack:
                        name, t = stack.pop()
                        if name in block:
                            continue
                        before = [line for line, _ in c.deps.get(name, ()) if line <= t]
                        if not before:
                            for call_line, args in c.param_deps.get(name, ()):
                                for d in args:  # a function parameter: what each call passes in
                                    if (d, call_line) not in visited:
                                        visited.add((d, call_line))
                                        seen.add(d)
                                        stack.append((d, call_line))
                            continue
                        latest = max(before)  # the assignment in effect at line t
                        for line, sources in c.deps.get(name, ()):
                            if line != latest:
                                continue
                            for d in sources:
                                if (d, line) not in visited:
                                    visited.add((d, line))
                                    seen.add(d)
                                    stack.append((d, line - 1 if d == name else line))
                    memo[key] = seen
                out |= memo[key]
            return out

        splits = [
            s for s in c.splits if not (ancestors(s.inputs, s.line) & c.toy) and not _discarded_piece(c, s)
        ]  # toy data isn't the dataset
        if not splits:
            return []

        def reaches(names: set[str], split: _Split) -> bool:
            return bool(names & (split.inputs | ancestors(split.inputs, split.line)))

        def nested(split: _Split) -> bool:
            """A validation split cut from the training part of an earlier split (train/val/test)."""
            inputs = ancestors(split.inputs, split.line)
            return any(
                o is not split and o.test_out and o.line < split.line and inputs & o.train_out for o in splits
            )

        found: dict[tuple[int, str], Finding] = {}

        def add(f: Finding) -> None:
            found.setdefault((f.location.line_start or 0, f.type), f)

        for fit in sorted(c.fits, key=lambda f: f.line):
            if not fit.data:
                continue
            fit_anc = ancestors(fit.data, fit.line)
            same = [s for s in splits if s.scope == fit.scope]
            earlier = [s for s in same if s.line < fit.line]
            if fit.scope in c.func_scopes:
                for call_line, call_scope in c.func_calls.get(c.func_scopes[fit.scope][0], []):
                    earlier += [s for s in splits if s.scope == call_scope and s.line < call_line]
            later = [s for s in same if s.line > fit.line]
            on_train = any(fit_anc & s.train_out for s in earlier)

            if fit.kind == "fit":
                held = next(
                    (s for s in reversed(earlier) if s.test_out and _held_out(fit_anc & s.test_out)), None
                )
                if held is not None and not _trainish(fit.data):
                    if fit.transformer and fit_anc & held.train_out:
                        add(self._before_split(src, fit, held, combined=True))  # train + held-out rows
                    elif not fit.transformer and fit.receiver and _evaluated_after(c, fit, held, ancestors):
                        add(self._test_in_training(src, fit, held, _held_out(fit_anc & held.test_out)))
                    # a transformer refit on the held-out rows alone is a bug, not leakage
                    continue
                if fit.selection and fit.early_stop:
                    for s in earlier:
                        used = ancestors(fit.selection, fit.line) & s.test_out
                        scored = not fit.quiet or _evaluated_after(c, fit, s, ancestors)
                        if used and scored and not nested(s):
                            add(
                                self._tuning(
                                    src, fit.line, s, "early stopping / best-checkpoint selection", used
                                )
                            )
                            break
                if fit.receiver and not fit.transformer:
                    replaced = _replaced_by_all_rows(c, fit, earlier, ancestors)
                    if replaced is not None and _evaluated_after(c, fit, replaced, ancestors):
                        add(self._model_on_all_data(src, fit, replaced))
            if on_train:
                continue  # fit on a training subset from an earlier split: correct
            if fit.transformer and fit.outputs:
                reach = fit.outputs
            elif fit.kind == "stat":
                reach = fit.writes
            else:
                reach = fit.data
            leaked = next(
                (s for s in later if reaches(fit.data, s) and reaches(reach, s) and _held_out_read(c, s)),
                None,
            )
            if leaked is None:
                continue
            if fit.kind in ("stat", "resample") or fit.transformer:
                add(self._before_split(src, fit, leaked))
            elif fit.receiver and _evaluated_on_test(c, fit, leaked, ancestors):
                add(self._model_on_all_data(src, fit, leaked))

        for f in self._selection_loops(src, c, splits, ancestors, nested):
            add(f)
        for f in self._importance_selection(src, c, splits, ancestors):
            add(f)
        return sorted(found.values(), key=lambda f: f.location.line_start or 0)

    def _selection_loops(
        self,
        src: SourceFile,
        c: _Collector,
        splits: list[_Split],
        ancestors: Ancestors,
        nested: Callable[[_Split], bool],
    ) -> list[Finding]:
        """A loop that refits or reweights and keeps the best score on the held-out rows."""
        out = []
        refit_loops = {f.loop for f in c.fits if f.loop is not None and f.kind == "fit"}
        candidates = [(e.line, e.loop, e.data, e.loop in refit_loops) for e in c.evals if e.loop is not None]
        candidates += [(line, loop, data, True) for line, loop, data in c.loop_metrics]
        for s in splits:
            if not s.test_out or nested(s):
                continue
            for line, loop, data, refits in sorted(candidates, key=lambda x: x[0]):
                used = ancestors(data, line) & s.test_out
                picks = loop in c.best_loops or loop in c.sweep_loops
                if picks and refits and line > s.line and used:
                    out.append(self._tuning(src, line, s, "picking the best result of a loop", used))
                    break
        return out

    def _importance_selection(
        self, src: SourceFile, c: _Collector, splits: list[_Split], ancestors: Ancestors
    ) -> list[Finding]:
        """Columns kept by importance from a model fit before the data is split or cross-validated."""
        out = []
        for line, models, written in c.importances:
            if not written:
                continue
            names = set(models.split(","))
            fits = [f for f in c.fits if f.receiver in names and f.line < line and f.kind == "fit"]
            for f in fits[-1:]:
                fit_anc = ancestors(f.data, f.line)
                chosen = c._descendants(written)
                fit_on = [e for e in splits if e.line < f.line and e.test_out and fit_anc & e.train_out]
                for s in splits:
                    if fit_anc & s.train_out:
                        continue
                    rows = frozenset(names | written)  # follow the rows, not the model that chose the columns
                    s_anc = s.inputs | ancestors(s.inputs, s.line, rows)
                    if any(s_anc & e.train_out or (e.seed and e.seed == s.seed) for e in fit_on):
                        continue  # chosen on training rows, then those rows (or the same partition) are used
                    before = s.line > line and written & ancestors(s.inputs, s.line)
                    # split first, importances from a model on all rows, then those columns on the held-out rows
                    after = (
                        s.line < f.line
                        and s.test_out
                        and fit_anc & (s.inputs | ancestors(s.inputs, s.line))
                        and _used_together(c, chosen, s.test_out, line)
                    )
                    if before or after:
                        out.append(self._importance_finding(src, line, f, s))
                        break
        return out

    # -- findings
    def _importance_finding(self, src: SourceFile, line: int, fit: _Fit, split: _Split) -> Finding:
        return make_finding(
            self,
            src,
            type="data.leakage.feature_selection_full",
            title="Features chosen by importance before the split",
            severity=Severity.high,
            confidence=0.7,
            line=line,
            rule_id="leakage.importance_selection_before_split",
            explanation=f"Columns are kept by the importances of a model fit at line {fit.line} on rows that are "
            f"split or cross-validated at line {split.line}, so the held-out rows helped choose the features.",
            metrics={"split_line": split.line},
            fix="Select features inside each training fold (or on a training split only).",
        )

    def _before_split(self, src: SourceFile, fit: _Fit, split: _Split, combined: bool = False) -> Finding:
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
            explanation=(
                f"{noun} at line {fit.line} is fit on data that includes the held-out rows of the split at "
                f"line {split.line}, so evaluation is optimistic."
                if combined
                else f"{noun} at line {fit.line} uses data that is split at line {split.line}, so it "
                "includes the test rows and evaluation is optimistic."
            ),
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
        if ancestors(e.data, e.line) & split.test_out:
            return not any(r.line < e.line for r in refits)
    return False


def _used_together(c: _Collector, chosen: set[str], held: set[str], after: int) -> bool:
    """Some later assignment reads both the chosen columns and the held-out rows: val_x[important]."""
    return any(
        line > after and sources & chosen and sources & held
        for assigns in c.deps.values()
        for line, sources in assigns
    )


def _discarded_piece(c: _Collector, split: _Split) -> bool:
    """train1 = train[:n]; train = train[n:m] and train1 never used: a subset of rows, not a holdout."""
    if split.subset_of is None:
        return False
    other = (split.train_out | split.test_out) - {split.subset_of}
    return not any(line > split.line for n in other for line in c.loads.get(n, ()))


def _held_out_read(c: _Collector, split: _Split) -> bool:
    """False when every held-out name is overwritten before anything reads it: a dead split."""
    if not split.test_out or split.via_call or split.scope != 0:
        return True
    for name in split.test_out:
        reassigned = [line for line, _ in c.deps.get(name, ()) if line > split.line]
        if not reassigned:
            return True
        if any(split.line < line <= min(reassigned) for line in c.loads.get(name, ())):
            return True
    return False


def _held_out(names: set[str]) -> set[str]:
    return {n for n in names if not TRAIN_NAME.search(n)}


def _trainish(names: set[str]) -> bool:
    """X_train, train_inputs: named as training data (and nothing named as held-out)."""
    return any(TRAIN_NAME.search(n) for n in names) and not any(HELD_OUT_NAME.search(n) for n in names)


def _evaluated_after(c: _Collector, fit: _Fit, split: _Split, ancestors: Ancestors) -> bool:
    return any(
        e.receiver == fit.receiver
        and e.scope == fit.scope
        and e.line > fit.line
        and ancestors(e.data, e.line) & split.test_out
        for e in c.evals
    )


def _replaced_by_all_rows(
    c: _Collector, fit: _Fit, earlier: list[_Split], ancestors: Ancestors
) -> _Split | None:
    """X_train = X after `X_train, X_test = split(X)`: the training set is all rows again."""
    for s in earlier:
        if not s.test_out:
            continue
        full = s.inputs | ancestors(s.inputs, s.line)
        for name in fit.data & s.train_out:
            last = [
                (line, src)
                for n, line, src in c.assigns
                if n == name and s.line < line < fit.line and line not in c.split_lines
            ]
            if last and last[-1][1] & full and not last[-1][1] & (s.train_out | s.test_out):
                return s
    return None
