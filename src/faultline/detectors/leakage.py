"""Leakage in ML code: test data reaching the model or its preprocessing during training.

Static, per scope (module/notebook or function), ordered by line. Tracks which names flow
into which (flow-insensitive, transitive) and where data is split, then flags:
  - preprocessing fit on data that is split later       -> preprocessing_before_split
  - feature selection fit on data that is split later   -> feature_selection_full
  - anything fit on the test side of a split             -> test_in_training
Modeled on the leakage categories of Yang et al., ASE'22 ("Data Leakage in Notebooks").
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

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
FIT_METHODS = frozenset({"fit", "fit_transform", "fit_resample"})


@dataclass
class _Fit:
    line: int
    scope: int
    cls: str | None
    data: set[str]  # names in X (and y) arguments


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
        self.deps: dict[str, set[str]] = {}
        self.objects: dict[tuple[int, str], str] = {}  # (scope, name) -> estimator class
        self.targets: dict[int, ast.expr] = {}  # id(call) -> assignment target
        self.fits: list[_Fit] = []
        self.splits: list[_Split] = []

    @property
    def scope(self) -> int:
        return self.scopes[-1]

    def _enter(self, node: ast.AST) -> None:
        self.n_scopes += 1
        self.scopes.append(self.n_scopes)
        self.generic_visit(node)
        self.scopes.pop()

    visit_FunctionDef = _enter
    visit_AsyncFunctionDef = _enter

    def _bind(self, targets: list[ast.expr], value: ast.expr) -> None:
        sources = A.names_in(value)
        for t in targets:
            for name in A.target_names(t):
                self.deps.setdefault(name, set()).update(sources)
        if isinstance(value, ast.Call):
            self.targets[id(value)] = targets[0]
            cls = _ctor(value)
            if cls and len(targets) == 1 and isinstance(targets[0], ast.Name):
                self.objects[(self.scope, targets[0].id)] = cls

    def visit_Assign(self, node: ast.Assign) -> None:
        self._bind(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._bind([node.target], node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._bind([node.target], node.value)
        self.generic_visit(node)

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
            data = A.names_in(node.args[0]) if node.args else set()
            if len(node.args) > 1:
                data |= A.names_in(node.args[1])
            if seg == "split" and cls in CV_SPLITTERS or seg in FIT_METHODS and cls in CV_SEARCH:
                self.splits.append(_Split(node.lineno, self.scope, data))
            elif seg in FIT_METHODS:
                x_only = A.names_in(node.args[0]) if node.args else set()
                is_prep = cls in FIT_BEFORE_SPLIT
                self.fits.append(_Fit(node.lineno, self.scope, cls, x_only if is_prep else data))
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
    known = FIT_BEFORE_SPLIT | CV_SEARCH | CV_SPLITTERS
    return seg if seg in known else None


class LeakageDetector:
    name = "leakage"
    version = "0.1"

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

        out: list[Finding] = []
        for fit in sorted(c.fits, key=lambda f: f.line):
            if not fit.data:
                continue
            fit_anc = ancestors(fit.data)
            earlier = [s for s in c.splits if s.scope == fit.scope and s.line < fit.line]
            later = [s for s in c.splits if s.scope == fit.scope and s.line > fit.line]
            on_test = [s for s in earlier if s.test_out and fit_anc & s.test_out]
            on_train = any(fit_anc & s.train_out for s in earlier)
            if on_test and not (fit_anc & on_test[-1].train_out and fit.data & on_test[-1].train_out):
                out.append(self._test_in_training(src, fit, on_test[-1], fit.data & on_test[-1].test_out))
                continue
            if fit.cls not in FIT_BEFORE_SPLIT:
                continue
            if on_train:
                continue  # fit on a training subset from an earlier split: correct
            leaked = next((s for s in later if fit.data & (s.inputs | ancestors(s.inputs))), None)
            if leaked:
                out.append(self._before_split(src, fit, leaked))
        return out

    def _before_split(self, src: SourceFile, fit: _Fit, split: _Split) -> Finding:
        selection = fit.cls in SELECTORS
        kind = "feature selection" if selection else "preprocessing"
        return make_finding(
            self,
            src,
            type="data.leakage.feature_selection_full"
            if selection
            else "data.leakage.preprocessing_before_split",
            title=f"{fit.cls} fit on data before the train/test split",
            severity=Severity.high if selection else Severity.medium,
            confidence=0.8,
            line=fit.line,
            rule_id=f"leakage.{'selection' if selection else 'preprocessing'}_before_split",
            explanation=f"{fit.cls} is fit at line {fit.line} on data that is split at line {split.line}, "
            f"so {kind} statistics include the test rows and evaluation is optimistic.",
            metrics={"split_line": split.line},
            fix=f"Split first, fit {fit.cls} on the training split only, then transform the test split "
            "(or wrap it in an sklearn Pipeline so cross-validation refits it per fold).",
        )

    def _test_in_training(self, src: SourceFile, fit: _Fit, split: _Split, names: set[str]) -> Finding:
        direct = bool(names)
        what = fit.cls or "an estimator"
        return make_finding(
            self,
            src,
            type="data.leakage.test_in_training",
            title=f"{what} fit on test data",
            severity=Severity.high,
            confidence=0.8 if direct else 0.6,
            line=fit.line,
            rule_id="leakage.fit_on_test",
            explanation=f"{what} is fit at line {fit.line} on data derived from the test side of the "
            f"split at line {split.line}; the model or its preprocessing has seen the test set.",
            metrics={"split_line": split.line},
            fix="Fit only on the training split; on the test split call transform/predict, never fit.",
        )
