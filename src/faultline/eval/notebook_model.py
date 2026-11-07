"""Phase 4 v1: a LightGBM classifier for pipeline issues, trained on the per-notebook labels.

One row per (notebook, issue type). Features are what the rules already collect (findings per
type and rule, with confidence), cheap code facts (which splitters, transforms and evaluation calls
the notebook uses) and the competition label (what the data requires). The target is whether the
notebook is labeled with that issue type.

Training uses the train split only. A threshold per type is picked on out-of-fold predictions
(folds grouped by competition, like the corpus splits), then the model is scored once on dev, next
to the rules on the same rows. Test needs final=True.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from faultline.eval.notebooks import LabeledNotebook, TypeScore, rule_findings
from faultline.schema import Finding

# Issue types the model predicts: the labeled data issues with enough train examples to learn from.
MODEL_TYPES = (
    "data.leakage.preprocessing_before_split",
    "data.leakage.tuning_on_test",
    "data.leakage.temporal_split",
    "data.leakage.feature_selection_full",
    "data.leakage.group_split",
    "data.leakage.target_derived",
    "data.leakage.test_in_training",
    "data.leakage.resampling_before_split",
    "data.fit.unstratified_split",
    "data.fit.single_split_small",
    "data.fit.metric_mismatch",
    "data.fit.id_as_feature",
)
CODE_FACTS: dict[str, str] = {
    "train_test_split": r"\btrain_test_split\(",
    "shuffle_false": r"shuffle\s*=\s*False",
    "stratify_kw": r"\bstratify\s*=",
    "kfold": r"\bKFold\(",
    "stratified_kfold": r"\bStratified(Shuffle|K)\w*\(|RepeatedStratifiedKFold\(",
    "group_kfold": r"\bGroup\w*(KFold|Split)\(|LeaveOneGroupOut\(",
    "time_series_split": r"\bTimeSeriesSplit\(",
    "positional_slice": r"\[\s*:\s*-?\s*\w+\s*\]|\.iloc\[\s*:\s*-?\w+|\[\s*-?\w+\s*:\s*\]",
    "date_cutoff": r"(date|time|day|Date)\w*\s*[<>]=?\s*['\"\d]",
    "cv_func": r"\bcross_val_(score|predict)\(|\bcross_validate\(|\bxgb\.cv\(|\blgb\.cv\(",
    "search": r"\b(GridSearchCV|RandomizedSearchCV|BayesSearchCV)\(|optuna",
    "eval_set": r"\beval_set\s*=|\bvalid_sets\s*=|\bvalidation_data\s*=|\bevals\s*=",
    "early_stopping": r"early_stopping|EarlyStopping\(|use_best_model",
    "checkpoint": r"ModelCheckpoint\(|save_best_only|load_weights\(|best_state",
    "fit_transform": r"\.fit_transform\(",
    "scaler": r"\b(Standard|MinMax|Robust|MaxAbs)Scaler\(|\bnormalize\(",
    "imputer": r"\b(Simple|KNN|Iterative)Imputer\(|fillna\([^)]*\.(mean|median|mode)\(",
    "decomposition": r"\b(PCA|TruncatedSVD|FastICA|KMeans|GaussianMixture)\(",
    "tfidf": r"\bTfidfVectorizer\(",
    "group_transform": r"groupby\([^)]*\)[^\n]*\.(transform|agg|mean)\(",
    "value_counts_map": r"value_counts\([^)]*\)[^\n]*|\.map\(",
    "target_groupby": r"groupby\([^)]*\)\[['\"]?(target|y|label|Target|SalePrice|Sales|sales)",
    "resampler": r"\b(SMOTE|ADASYN|RandomOverSampler|RandomUnderSampler|resample)\(",
    "selector": r"\b(SelectKBest|SelectFromModel|RFE|RFECV|VarianceThreshold)\(|feature_importances_|get_fscore",
    "concat_train_test": r"concat\(\s*\[\s*train\w*\s*,\s*test",
    "accuracy": r"\baccuracy_score\(|['\"]accuracy['\"]|val_acc",
    "auc": r"\broc_auc_score\(|['\"]auc['\"]",
    "classifier": r"Classifier\(|LogisticRegression\(|SVC\(",
    "regressor": r"Regressor\(|LinearRegression\(|Ridge\(|Lasso\(",
    "neural_net": r"\bSequential\(|nn\.Module|keras",
    "lag_features": r"\.shift\(|\.rolling\(|\.ewm\(|\.expanding\(",
    "id_columns": r"['\"](id|Id|ID|\w+_id|\w+Id)['\"]",
    "drop_target": r"\.drop\([^)]*(target|Target|label|SalePrice|Sales)",
    "test_size": r"\btest_size\s*=",
}
_COMPILED = {k: re.compile(v) for k, v in CODE_FACTS.items()}
SPLIT_TYPES = ("random", "group", "temporal")
TASKS = ("binary", "multiclass", "multilabel", "ordinal", "regression")


def feature_names(rule_ids: Sequence[str]) -> list[str]:
    return [
        *(f"type={t}" for t in MODEL_TYPES),
        "rule_hits_this_type",
        "rule_max_conf_this_type",
        *(f"rule={r}" for r in rule_ids),
        *(f"code={k}" for k in CODE_FACTS),
        "code_lines",
        *(f"split={s}" for s in SPLIT_TYPES),
        *(f"task={t}" for t in TASKS),
        "stratify",
        "small",
    ]


def notebook_rows(nb: LabeledNotebook, findings: list[Finding], rule_ids: Sequence[str]) -> np.ndarray:
    """One feature row per MODEL_TYPES entry for this notebook."""
    code = nb.src.code
    facts = [float(len(rx.findall(code))) for rx in _COMPILED.values()]
    by_rule = [float(sum(f.evidence.rule_id == r for f in findings)) for r in rule_ids]
    lab = nb.competition_label
    comp = [
        *(float(lab is not None and lab.split_type == s) for s in SPLIT_TYPES),
        *(float(lab is not None and lab.task == t) for t in TASKS),
        float(lab is not None and lab.stratify),
        float(lab is not None and lab.small),
    ]
    n_lines = float(sum(1 for line in nb.src.lines if line.strip()))
    rows = []
    for t in MODEL_TYPES:
        hits = [f for f in findings if f.type == t]
        rows.append(
            [
                *(float(t == u) for u in MODEL_TYPES),
                float(len(hits)),
                max((f.confidence for f in hits), default=0.0),
                *by_rule,
                *facts,
                n_lines,
                *comp,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


class Dataset(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    X: Any  # np.ndarray (n_notebooks * len(MODEL_TYPES), n_features)
    y: Any  # np.ndarray of 0/1
    groups: list[str]  # competition per row
    rule_pred: Any  # np.ndarray of 0/1: the rules flag this type in this notebook
    notebooks: list[str]


def build_dataset(
    notebooks: Sequence[LabeledNotebook], rule_ids: Sequence[str] | None = None
) -> tuple[Dataset, list[str]]:
    scanned = [(nb, rule_findings(nb, context=True)) for nb in notebooks]
    if rule_ids is None:  # fixed by the train split; dev/test reuse it
        rule_ids = sorted({f.evidence.rule_id for _, fs in scanned for f in fs})
    xs, ys, groups, rule_pred, names = [], [], [], [], []
    for nb, findings in scanned:
        xs.append(notebook_rows(nb, findings, rule_ids))
        labeled = {i.type for i in nb.label.issues}
        found = {f.type for f in findings}
        for t in MODEL_TYPES:
            ys.append(int(t in labeled))
            rule_pred.append(int(t in found))
            groups.append(nb.competition)
            names.append(f"{nb.competition}/{nb.rel}::{t}")
    width = len(feature_names(rule_ids))
    X = np.vstack(xs) if xs else np.zeros((0, width))
    return (
        Dataset(X=X, y=np.asarray(ys), groups=groups, rule_pred=np.asarray(rule_pred), notebooks=names),
        list(rule_ids),
    )


PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "n_estimators": 300,
    "num_leaves": 15,
    "min_child_samples": 10,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}


def _model(seed: int) -> Any:
    import lightgbm as lgb  # the ml extra

    return lgb.LGBMClassifier(**PARAMS, random_state=seed)


def oof_predictions(ds: Dataset, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold probabilities, folds grouped by competition."""
    from sklearn.model_selection import GroupKFold

    oof = np.zeros(len(ds.y))
    for trn, val in GroupKFold(n_splits=n_folds).split(ds.X, ds.y, ds.groups):
        m = _model(seed).fit(ds.X[trn], ds.y[trn])
        oof[val] = m.predict_proba(ds.X[val])[:, 1]
    return oof


def type_index(n_rows: int) -> np.ndarray:
    return np.tile(np.arange(len(MODEL_TYPES)), n_rows // len(MODEL_TYPES))


def pick_thresholds(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Per type, the threshold with the best F1 on out-of-fold predictions (0.5 if no positives)."""
    idx = type_index(len(y))
    out: dict[str, float] = {}
    for k, t in enumerate(MODEL_TYPES):
        yt, pt = y[idx == k], prob[idx == k]
        best, best_f1 = 0.5, -1.0
        for th in np.arange(0.05, 0.96, 0.05):
            pred = pt >= th
            tp = int((pred & (yt == 1)).sum())
            fp = int((pred & (yt == 0)).sum())
            fn = int((~pred & (yt == 1)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best_f1:
                best, best_f1 = float(th), f1
        out[t] = round(best, 2)
    return out


def score_by_type(y: np.ndarray, pred: np.ndarray) -> dict[str, TypeScore]:
    idx = type_index(len(y))
    return {
        t: TypeScore(
            tp=int(((pred == 1) & (y == 1) & (idx == k)).sum()),
            fp=int(((pred == 1) & (y == 0) & (idx == k)).sum()),
            fn=int(((pred == 0) & (y == 1) & (idx == k)).sum()),
        )
        for k, t in enumerate(MODEL_TYPES)
    }


class ModelReport(BaseModel):
    train_notebooks: int
    eval_split: str
    eval_notebooks: int
    thresholds: dict[str, float]
    oof_model: dict[str, TypeScore]  # train, out of fold
    oof_rules: dict[str, TypeScore]
    eval_model: dict[str, TypeScore]
    eval_rules: dict[str, TypeScore]
    top_features: list[tuple[str, float]]


def train_and_evaluate(
    train: Sequence[LabeledNotebook],
    evaluate: Sequence[LabeledNotebook],
    eval_split: str,
    out_model: Path | None = None,
    seed: int = 0,
) -> ModelReport:
    tr, rule_ids = build_dataset(train)
    ev, _ = build_dataset(evaluate, rule_ids)
    oof = oof_predictions(tr, seed=seed)
    thresholds = pick_thresholds(tr.y, oof)
    th_tr = np.array([thresholds[MODEL_TYPES[k]] for k in type_index(len(tr.y))])
    model = _model(seed).fit(tr.X, tr.y)
    prob = model.predict_proba(ev.X)[:, 1] if len(ev.y) else np.zeros(0)
    th_ev = np.array([thresholds[MODEL_TYPES[k]] for k in type_index(len(ev.y))])
    names = feature_names(rule_ids)
    gains = model.booster_.feature_importance(importance_type="gain")
    top = sorted(zip(names, (float(g) for g in gains), strict=True), key=lambda x: -x[1])[:15]
    if out_model is not None:
        out_model.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(out_model))
    return ModelReport(
        train_notebooks=len(tr.y) // len(MODEL_TYPES),
        eval_split=eval_split,
        eval_notebooks=len(ev.y) // len(MODEL_TYPES),
        thresholds=thresholds,
        oof_model=score_by_type(tr.y, (oof >= th_tr).astype(int)),
        oof_rules=score_by_type(tr.y, tr.rule_pred),
        eval_model=score_by_type(ev.y, (prob >= th_ev).astype(int)),
        eval_rules=score_by_type(ev.y, ev.rule_pred),
        top_features=top,
    )


def _micro(scores: dict[str, TypeScore]) -> TypeScore:
    return TypeScore(
        tp=sum(s.tp for s in scores.values()),
        fp=sum(s.fp for s in scores.values()),
        fn=sum(s.fn for s in scores.values()),
    )


def _f(v: float | None) -> str:
    return "  –  " if v is None else f"{v:5.1%}"


def summarize_model(r: ModelReport) -> str:
    def block(title: str, model: dict[str, TypeScore], rules: dict[str, TypeScore]) -> list[str]:
        rows = [title, f"{'type':42} {'rules P/R/F1':>21}   {'model P/R/F1':>21}  {'n':>4}"]
        for t in MODEL_TYPES:
            m, ru = model[t], rules[t]
            n = m.tp + m.fn
            rows.append(
                f"{t:42} {_f(ru.precision)} {_f(ru.recall)} {_f(ru.f1)}   {_f(m.precision)} {_f(m.recall)} {_f(m.f1)}  {n:4d}"
            )
        mm, mr = _micro(model), _micro(rules)
        rows.append(
            f"{'micro':42} {_f(mr.precision)} {_f(mr.recall)} {_f(mr.f1)}   {_f(mm.precision)} {_f(mm.recall)} {_f(mm.f1)}"
        )
        return rows

    out = [
        (
            f"LightGBM v1: trained on {r.train_notebooks} train notebooks, evaluated on {r.eval_notebooks} "
            f"{r.eval_split} notebooks (thresholds from out-of-fold train predictions)"
        ),
        "",
        *block(f"{r.eval_split} (held out)", r.eval_model, r.eval_rules),
        "",
        *block("train, out of fold (folds grouped by competition)", r.oof_model, r.oof_rules),
        "",
        "top features (gain): " + ", ".join(f"{n} {g:.0f}" for n, g in r.top_features[:10]),
    ]
    return "\n".join(out)
