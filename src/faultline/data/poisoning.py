"""Poisoning signals in data a model is about to be trained on.

- **Label flips**, by confident learning (Northcutt et al., JAIR 2021). Out-of-fold class
  probabilities come from a gradient-boosted model, each class gets a confidence threshold (the
  mean probability of that class over the rows labeled with it), and every row whose confident
  class differs from its label is an issue. About 30 lines here, so no cleanlab dependency. Every
  real dataset has some label noise, so the alarm compares the current issue rate with a
  trusted reference; without a reference, only a high absolute rate is reported (low severity).
- **Injected rows**, by two scores fit on trusted reference rows: an isolation forest and a
  Mahalanobis distance (robust-scaled features, Ledoit-Wolf covariance). A forest scores values
  beyond the reference range no higher than the reference's own extremes, and a few odd columns
  get diluted among many; the distance catches rows whose values don't fit together (an extreme
  but consistent row stays close). A row is flagged when either score is high. Both are
  cross-fitted: each model scores reference rows it never saw, and current rows are scored by a
  model picked at random, so clean current rows and held-out reference rows are exchangeable,
  and the false-alarm rate is measured before any current row is judged.
- **Duplicate floods**: many identical rows (features and label), which push a model cheaply.

Known limit: rows crafted to look normal with a plausible label (clean-label poisoning) pass all
three. Thresholds are defaults to be calibrated on dev datasets (rule 3).
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from faultline.data.frame import (
    as_str_columns,
    data_finding,
    encode,
    feature_columns,
    is_identifier,
    row_hashes,
    rows_preview,
)
from faultline.data.stats import increase_p_value
from faultline.schema import Finding, Severity, Tier


@dataclass(frozen=True)
class PoisoningConfig:
    folds: int = 5
    alpha: float = 0.01
    min_rows: int = 50
    max_classes: int = 20
    label_issue_delta: float = 0.03  # rise over the reference issue rate
    label_issue_rate: float = 0.10  # absolute rate, when there is no reference
    outlier_quantile: float = 0.98  # held-out reference rows flagged by either score: ~2-4%
    outlier_ratio: float = 2.0  # current outlier rate must be at least this multiple
    min_flagged: int = 3
    flood_min_copies: int = 5
    flood_min_share: float = 0.005
    flood_ratio: float = 5.0  # a group's share must be this multiple of its reference share


@dataclass(frozen=True)
class LabelIssues:
    rows: np.ndarray  # positions, least self-confident first
    given: np.ndarray
    suggested: np.ndarray
    self_confidence: np.ndarray


def out_of_fold_proba(x: np.ndarray, y: np.ndarray, *, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Class probabilities for every row from a model that never saw that row."""
    splits = max(2, min(folds, int(np.bincount(y).min())))
    model = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    return np.asarray(cross_val_predict(model, x, y, cv=cv, method="predict_proba"))


def confident_label_issues(proba: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows whose confident class differs from their label: (rows, believed class, p(label))."""
    k = proba.shape[1]
    thresholds = np.array([proba[y == j, j].mean() for j in range(k)])
    above = proba >= thresholds
    believed = np.where(above, proba, -1.0).argmax(axis=1)
    rows = np.flatnonzero(above.any(axis=1) & (believed != y))
    self_conf = proba[rows, y[rows]]
    order = np.argsort(self_conf, kind="stable")
    return rows[order], believed[rows[order]], self_conf[order]


def find_label_issues(x: np.ndarray, labels: np.ndarray, *, folds: int = 5, seed: int = 0) -> LabelIssues:
    classes, y = np.unique(labels, return_inverse=True)
    y = y.ravel()
    keep = np.flatnonzero(np.bincount(y)[y] >= 2)  # a class needs rows on both sides of a fold
    kept, yk = np.unique(y[keep], return_inverse=True)
    yk = yk.ravel()
    if len(kept) < 2:
        empty = np.array([], dtype=int)
        return LabelIssues(empty, labels[empty], labels[empty], np.array([], dtype=float))
    proba = out_of_fold_proba(x[keep], yk, folds=folds, seed=seed)
    rel, believed, self_conf = confident_label_issues(proba, yk)
    names = classes[kept]
    return LabelIssues(
        rows=keep[rel], given=names[yk[rel]], suggested=names[believed], self_confidence=self_conf
    )


def outlier_scores(
    ref_x: np.ndarray, cur_x: np.ndarray, *, folds: int = 5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Anomaly scores (higher = stranger), shape (rows, 2): isolation forest and Mahalanobis
    distance, for held-out reference rows and for current rows, from the same cross-fitted
    models so the two are comparable."""
    rng = np.random.default_rng(seed)
    ref_fold = rng.permutation(len(ref_x)) % folds
    cur_fold = rng.integers(folds, size=len(cur_x))
    ref_scores, cur_scores = np.empty((len(ref_x), 2)), np.empty((len(cur_x), 2))
    for k in range(folds):
        train, held, in_fold = ref_x[ref_fold != k], ref_fold == k, cur_fold == k
        forest = IsolationForest(n_estimators=200, random_state=seed + k).fit(train)
        center = np.median(train, axis=0)
        q25, q75 = np.percentile(train, [25, 75], axis=0)
        scale = np.where(q75 - q25 > 0, q75 - q25, 1.0)
        cov = LedoitWolf().fit((train - center) / scale)
        for x, out, rows in ((ref_x, ref_scores, held), (cur_x, cur_scores, in_fold)):
            if rows.any():
                out[rows, 0] = -forest.score_samples(x[rows])
                out[rows, 1] = cov.mahalanobis((x[rows] - center) / scale)
    return ref_scores, cur_scores


def _label(value: object) -> str:
    return str(value)[:40]


class PoisoningDetector:
    name = "data_poisoning"
    version = "0.1"

    def __init__(self, config: PoisoningConfig | None = None) -> None:
        self.config = config or PoisoningConfig()

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    def check(
        self,
        current: pd.DataFrame,
        *,
        dataset: str,
        target: str | None = None,
        reference: pd.DataFrame | None = None,
        seed: int = 0,
    ) -> list[Finding]:
        """Row numbers in findings are 0-based positions in `current`."""
        cfg = self.config
        cur = as_str_columns(current).reset_index(drop=True)
        ref = None if reference is None else as_str_columns(reference).reset_index(drop=True)
        if len(cur) < cfg.min_rows:
            return []
        if ref is not None and len(ref) < cfg.min_rows:
            ref = None
        frames = [cur] if ref is None else [cur, ref]
        features = feature_columns(frames, exclude=[target] if target else [])
        findings = self._floods(cur, ref, dataset)
        if target is not None and target in cur.columns and features:
            labeled_ref = ref if ref is not None and target in ref.columns else None
            findings += self._labels(cur, labeled_ref, features, target, dataset, seed)
        if ref is not None and features:
            findings += self._outliers(ref, cur, features, dataset, seed)
        return findings

    def _labels(
        self,
        cur: pd.DataFrame,
        ref: pd.DataFrame | None,
        features: list[str],
        target: str,
        dataset: str,
        seed: int,
    ) -> list[Finding]:
        cfg = self.config
        cols = [*features, target]
        n_ref = 0 if ref is None else len(ref)
        both = cur[cols] if ref is None else pd.concat([ref[cols], cur[cols]], ignore_index=True)
        labeled = both[target].notna().to_numpy()
        if not 2 <= both.loc[labeled, target].nunique() <= cfg.max_classes:
            return []
        x = encode([both], features)[0][labeled]
        labels = both.loc[labeled, target].astype("string").to_numpy(dtype=object)
        issues = find_label_issues(x, labels, folds=cfg.folds, seed=seed)
        positions = np.flatnonzero(labeled)[issues.rows]
        in_cur = positions >= n_ref
        cur_rows = positions[in_cur] - n_ref
        k_cur, n_cur = len(cur_rows), int(labeled[n_ref:].sum())
        rate_cur = k_cur / n_cur if n_cur else 0.0
        metrics: dict[str, float | int | str] = {
            "issues": k_cur,
            "labeled_rows": n_cur,
            "issue_rate": round(rate_cur, 4),
            "rows": rows_preview(cur_rows),
        }
        if k_cur:
            pairs = Counter(
                zip(issues.given[in_cur].tolist(), issues.suggested[in_cur].tolist(), strict=True)
            )
            (given, suggested), count = pairs.most_common(1)[0]
            metrics["top_flip"] = f"{_label(given)} -> {_label(suggested)}"
            metrics["top_flip_share"] = round(count / k_cur, 4)
        fix = (
            "Review the listed rows' labels against their source; if one batch or source accounts for "
            "them, quarantine it and retrain without it."
        )
        if ref is None:
            if rate_cur < cfg.label_issue_rate:
                return []
            return [
                data_finding(
                    detector=self.id,
                    type="data.poisoning.label_flip",
                    title=f"{rate_cur:.1%} of labels disagree with the data",
                    severity=Severity.low,
                    confidence=0.5,
                    dataset=dataset,
                    columns=[target],
                    rule_id="poisoning.confident_learning",
                    explanation=f"{k_cur} rows' labels disagree with what a model trained on the other "
                    "rows confidently predicts (confident learning). With no trusted reference this can't "
                    "tell ordinary label noise from flipped labels.",
                    fix=fix,
                    metrics=metrics,
                    tier=Tier.tier2,
                )
            ]
        k_ref, n_ref_l = int((~in_cur).sum()), int(labeled[:n_ref].sum())
        rate_ref = k_ref / n_ref_l if n_ref_l else 0.0
        p = increase_p_value(k_ref, n_ref_l, k_cur, n_cur)
        if rate_cur - rate_ref < cfg.label_issue_delta or p >= cfg.alpha:
            return []
        metrics |= {"issue_rate_ref": round(rate_ref, 4), "p_value": p}
        return [
            data_finding(
                detector=self.id,
                type="data.poisoning.label_flip",
                title=f"Label issues rose from {rate_ref:.1%} (reference) to {rate_cur:.1%}",
                severity=Severity.high if rate_cur - rate_ref >= 0.10 else Severity.medium,
                confidence=0.8,
                dataset=dataset,
                columns=[target],
                rule_id="poisoning.confident_learning",
                explanation=f"{k_cur} current rows have labels that disagree with what a model trained on "
                "the other rows confidently predicts (confident learning), well above the rate in the "
                "trusted reference. Flipped labels, a broken labeling step or a changed label definition "
                "look like this.",
                fix=fix,
                metrics=metrics,
                tier=Tier.tier2,
            )
        ]

    def _outliers(
        self, ref: pd.DataFrame, cur: pd.DataFrame, features: list[str], dataset: str, seed: int
    ) -> list[Finding]:
        cfg = self.config
        xr, xc = encode([ref, cur], features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            medians = np.nanmedian(xr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        xr, xc = np.where(np.isnan(xr), medians, xr), np.where(np.isnan(xc), medians, xc)
        ref_scores, cur_scores = outlier_scores(xr, xc, folds=cfg.folds, seed=seed)
        # each score's threshold leaves half the budget; the reference rate is then measured
        # with the same either-score rule, so the comparison below stays exact
        thresholds = np.quantile(ref_scores, 1 - (1 - cfg.outlier_quantile) / 2, axis=0)
        thresholds = np.maximum(thresholds, 1e-12)
        ref_strangeness = (ref_scores / thresholds).max(axis=1)
        cur_strangeness = (cur_scores / thresholds).max(axis=1)
        k_ref = int((ref_strangeness > 1).sum())
        flagged = np.flatnonzero(cur_strangeness > 1)
        rate_ref, rate_cur = k_ref / len(xr), len(flagged) / len(xc)
        p = increase_p_value(k_ref, len(xr), len(flagged), len(xc))
        if len(flagged) < cfg.min_flagged or rate_cur < cfg.outlier_ratio * rate_ref or p >= cfg.alpha:
            return []
        flagged = flagged[np.argsort(-cur_strangeness[flagged], kind="stable")]
        q25, q75 = np.percentile(xr, [25, 75], axis=0)
        spread = np.where(q75 - q25 > 0, q75 - q25, 1.0)
        how_far = np.median(np.abs(xc[flagged] - medians) / spread, axis=0)  # robust z per column
        columns = [features[i] for i in np.argsort(-how_far)[:3] if how_far[i] > 2]
        excess = len(flagged) - rate_ref * len(xc)
        return [
            data_finding(
                detector=self.id,
                type="data.poisoning.injected_rows",
                title=f"{rate_cur:.1%} of rows look unlike the reference (expected {rate_ref:.1%})",
                severity=Severity.high if rate_cur >= 4 * max(rate_ref, 0.01) else Severity.medium,
                confidence=0.75,
                dataset=dataset,
                columns=columns,
                rule_id="poisoning.outlier_rows",
                explanation=f"An isolation forest and a Mahalanobis distance fit on the trusted reference "
                f"flag {len(flagged)} current rows as anomalous, about {max(excess, 0):.0f} more than "
                "chance. Injected or corrupted rows look like this; the columns listed are where the "
                "flagged rows are most extreme.",
                fix="Trace the listed rows to their source or batch; quarantine them before training and "
                "add a range check on the listed columns at ingestion.",
                metrics={
                    "flagged": len(flagged),
                    "rate_cur": round(rate_cur, 4),
                    "rate_ref": round(rate_ref, 4),
                    "p_value": p,
                    "rows": rows_preview(flagged),
                },
                tier=Tier.tier2,
            )
        ]

    def _floods(self, cur: pd.DataFrame, ref: pd.DataFrame | None, dataset: str) -> list[Finding]:
        cfg = self.config
        cols = [c for c in cur.columns if not is_identifier(cur[c])]
        if not cols:
            return []
        hashes = row_hashes(cur, cols)
        groups = pd.Series(hashes).value_counts()
        need = max(cfg.flood_min_copies, math.ceil(cfg.flood_min_share * len(cur)))
        big = groups[groups >= need]
        if ref is not None and not big.empty and all(c in ref.columns for c in cols):
            ref_share = pd.Series(row_hashes(ref, cols)).value_counts().reindex(
                big.index, fill_value=0
            ) / len(ref)
            big = big[big / len(cur) >= cfg.flood_ratio * ref_share]
        if big.empty:
            return []
        rows = np.flatnonzero(np.isin(hashes, big.index.to_numpy()))
        share = len(rows) / len(cur)
        return [
            data_finding(
                detector=self.id,
                type="data.poisoning.injected_rows",
                title=f"{len(big)} row(s) repeated {int(big.max())}+ times ({share:.1%} of the data)",
                severity=Severity.high if share >= 0.05 else Severity.medium,
                confidence=0.8 if ref is not None else 0.6,
                dataset=dataset,
                rule_id="poisoning.duplicate_flood",
                explanation="Groups of identical rows (all columns except ids)"
                + (" that the reference data doesn't have" if ref is not None else "")
                + ". Repeating a row is a cheap way to pull a model toward it, and also a common "
                "result of a re-run ingestion job.",
                fix="Deduplicate before training (drop_duplicates) and find the job or source that wrote "
                "the copies.",
                metrics={
                    "groups": len(big),
                    "flagged": len(rows),
                    "largest": int(big.max()),
                    "rows": rows_preview(rows),
                },
                tier=Tier.tier1,
            )
        ]
