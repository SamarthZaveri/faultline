"""Drift between reference data (what the pipeline was built and validated on) and current data (a
new batch, a later snapshot, serving data).

- Per column, a shift is reported only when it is both large (PSI, an effect size) and clear (KS
  for numbers, chi-squared for categories; Bonferroni-corrected over columns), so big samples
  don't turn every wobble into a finding.
- A classifier two-sample test (a model trained to tell reference rows from current rows) catches
  joint shifts that no single column shows, such as a correlation flipping, and names the columns
  the model relied on (permutation importance).
- With a target column: label-prior shift, and concept drift (a model fit on the reference makes
  more mistakes on current rows than on held-out reference rows).

Thresholds are defaults to be calibrated on dev datasets (rule 3); confidences are not yet
calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from faultline.data.frame import as_str_columns, column_kind, data_finding, encode, is_identifier
from faultline.data.stats import auc_p_value, increase_p_value, psi, psi_from_counts
from faultline.schema import Finding, Severity, Tier

SCALE_FACTORS = (10.0, 100.0, 1000.0, 1e6)


@dataclass(frozen=True)
class DriftConfig:
    psi_threshold: float = 0.2
    alpha: float = 0.01  # family-wise over the columns tested
    bins: int = 10
    min_rows: int = 50
    missing_delta: float = 0.05  # absolute rise in the missing rate
    new_category_share: float = 0.01  # share of current rows holding values never seen before
    c2st_margin: float = 0.05  # AUC must clear 0.5 by this much (effect size) and be significant
    c2st_max_rows: int = 5000  # per side
    concept_delta: float = 0.05  # absolute rise in the error rate
    max_classes: int = 20


def scale_ratio(ref: np.ndarray, cur: np.ndarray) -> float | None:
    """The power-of-ten factor between the two medians (a unit change such as kg -> g), if any."""
    mr, mc = float(np.median(ref)), float(np.median(cur))
    if mr <= 0 or mc <= 0:
        return None
    ratio = mc / mr
    for factor in SCALE_FACTORS:
        for candidate in (factor, 1 / factor):
            if abs(ratio / candidate - 1) < 0.15:
                return candidate
    return None


def _confidence(value: float, threshold: float) -> float:
    return 0.9 if value >= 2 * threshold else 0.75


class DriftDetector:
    name = "data_drift"
    version = "0.1"

    def __init__(self, config: DriftConfig | None = None) -> None:
        self.config = config or DriftConfig()

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    def compare(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        *,
        dataset: str,
        target: str | None = None,
        seed: int = 0,
    ) -> list[Finding]:
        cfg = self.config
        ref, cur = as_str_columns(reference), as_str_columns(current)
        if len(ref) < cfg.min_rows or len(cur) < cfg.min_rows:
            return []
        findings = self._schema(ref, cur, dataset)
        shared = [c for c in ref.columns if c in cur.columns]
        missing_alpha = cfg.alpha / max(1, len(shared))
        for c in shared:
            findings += self._missing(ref[c], cur[c], c, dataset, missing_alpha)
        tested = [
            c
            for c in shared
            if column_kind(ref[c]) == column_kind(cur[c])
            and column_kind(ref[c]) in ("numeric", "categorical")
            and not is_identifier(ref[c])
        ]
        alpha = cfg.alpha / max(1, len(tested))
        flagged: set[str] = set()
        for c in tested:
            hits = self._distribution(ref[c], cur[c], c, dataset, alpha, is_target=c == target)
            if any(f.type == "data.drift.covariate" for f in hits):
                flagged.add(c)
            findings += hits
        features = [c for c in tested if c != target]
        findings += self._joint(ref, cur, features, flagged, dataset, seed)
        if target is not None and target in tested:
            drifted = any(f.type == "data.drift.covariate" for f in findings)
            findings += self._concept(ref, cur, features, target, dataset, seed, drifted)
        return findings

    def _schema(self, ref: pd.DataFrame, cur: pd.DataFrame, dataset: str) -> list[Finding]:
        out = []
        for c in ref.columns:
            if c not in cur.columns:
                out.append(
                    data_finding(
                        detector=self.id,
                        type="data.drift.schema",
                        title=f"Column {c!r} is missing from the current data",
                        severity=Severity.high,
                        confidence=0.95,
                        dataset=dataset,
                        columns=[c],
                        rule_id="drift.schema.missing_column",
                        explanation="The reference data has this column and the current data doesn't. "
                        "Code that selects it fails, or a default silently fills it in.",
                        fix="Restore the column upstream, or change the pipeline and retrain without it.",
                        tier=Tier.tier2,
                    )
                )
            elif cur[c].notna().any() and (rk := column_kind(ref[c])) != (ck := column_kind(cur[c])):
                out.append(
                    data_finding(
                        detector=self.id,
                        type="data.drift.schema",
                        title=f"Column {c!r} changed type from {rk} to {ck}",
                        severity=Severity.high,
                        confidence=0.9,
                        dataset=dataset,
                        columns=[c],
                        rule_id="drift.schema.type_change",
                        explanation=f"The column was {rk} in the reference data and is {ck} now "
                        f"(dtype {ref[c].dtype} -> {cur[c].dtype}). Encoders and arithmetic on it break "
                        "or silently coerce values.",
                        fix="Fix the type upstream, or cast and validate it at ingestion with a schema check.",
                        tier=Tier.tier2,
                    )
                )
        for c in cur.columns:
            if c not in ref.columns:
                out.append(
                    data_finding(
                        detector=self.id,
                        type="data.drift.schema",
                        title=f"New column {c!r} in the current data",
                        severity=Severity.low,
                        confidence=0.95,
                        dataset=dataset,
                        columns=[c],
                        rule_id="drift.schema.new_column",
                        explanation="The current data has a column the reference data doesn't. "
                        "Code that selects all columns picks it up as a feature.",
                        fix="Select features explicitly, or add the column to the expected schema.",
                    )
                )
        return out

    def _missing(
        self, ref: pd.Series, cur: pd.Series, column: str, dataset: str, alpha: float
    ) -> list[Finding]:
        k_ref, k_cur = int(ref.isna().sum()), int(cur.isna().sum())
        r_rate, c_rate = k_ref / len(ref), k_cur / len(cur)
        if c_rate - r_rate < self.config.missing_delta:
            return []
        p = increase_p_value(k_ref, len(ref), k_cur, len(cur))
        if p >= alpha:
            return []
        return [
            data_finding(
                detector=self.id,
                type="data.drift.missing_spike",
                title=f"Missing values in {column!r} rose from {r_rate:.1%} to {c_rate:.1%}",
                severity=Severity.high if c_rate - r_rate >= 0.25 else Severity.medium,
                confidence=0.9,
                dataset=dataset,
                columns=[column],
                rule_id="drift.missing",
                explanation="Much more of this column is missing than in the reference data. Imputation "
                "then fills it with a constant, which shifts the feature and hides the upstream break.",
                fix="Find the upstream source that stopped filling the column; alert on its missing rate.",
                metrics={"missing_ref": round(r_rate, 4), "missing_cur": round(c_rate, 4), "p_value": p},
                tier=Tier.tier2,
            )
        ]

    def _distribution(
        self, ref: pd.Series, cur: pd.Series, column: str, dataset: str, alpha: float, *, is_target: bool
    ) -> list[Finding]:
        if column_kind(ref) == "numeric":
            return self._numeric(ref, cur, column, dataset, alpha, is_target=is_target)
        return self._categorical(ref, cur, column, dataset, alpha, is_target=is_target)

    def _numeric(
        self, ref: pd.Series, cur: pd.Series, column: str, dataset: str, alpha: float, *, is_target: bool
    ) -> list[Finding]:
        cfg = self.config
        r = pd.to_numeric(ref, errors="coerce").dropna().to_numpy(dtype=float)
        c = pd.to_numeric(cur, errors="coerce").dropna().to_numpy(dtype=float)
        if len(r) < cfg.min_rows or len(c) < cfg.min_rows:
            return []
        value = psi(r, c, cfg.bins)
        p = float(stats.ks_2samp(r, c).pvalue)
        if value < cfg.psi_threshold or p >= alpha:
            return []
        metrics: dict[str, float | int | str] = {
            "psi": round(value, 4),
            "ks_p_value": p,
            "median_ref": round(float(np.median(r)), 6),
            "median_cur": round(float(np.median(c)), 6),
        }
        explanation = f"The distribution moved (PSI {value:.2f}; KS p={p:.1e})."
        if (factor := scale_ratio(r, c)) is not None:
            metrics["scale_ratio"] = factor
            explanation += f" Values are about x{factor:g} the reference: this looks like a unit change."
        return [self._shift_finding(column, dataset, value, metrics, explanation, is_target, "drift.psi_ks")]

    def _categorical(
        self, ref: pd.Series, cur: pd.Series, column: str, dataset: str, alpha: float, *, is_target: bool
    ) -> list[Finding]:
        cfg = self.config
        r, c = ref.dropna().astype("string"), cur.dropna().astype("string")
        if len(r) < cfg.min_rows or len(c) < cfg.min_rows:
            return []
        rc, cc = r.value_counts(), c.value_counts()
        out = []
        new = cc.index.difference(rc.index)
        new_share = float(cc[new].sum()) / len(c) if len(new) else 0.0
        if new_share >= cfg.new_category_share:
            out.append(
                data_finding(
                    detector=self.id,
                    type="data.drift.new_category",
                    title=f"{len(new)} unseen value(s) in {column!r} cover {new_share:.1%} of current rows",
                    severity=Severity.medium,
                    confidence=0.9,
                    dataset=dataset,
                    columns=[column],
                    rule_id="drift.new_category",
                    explanation="Values never seen in the reference data. Encoders fitted on the reference "
                    "either fail or map them all to one unknown bucket.",
                    fix="Decide how unknown values are handled (handle_unknown, an 'other' bucket) and "
                    "check upstream whether the new values are legitimate.",
                    metrics={"new_values": len(new), "new_share": round(new_share, 4)},
                    tier=Tier.tier2,
                )
            )
        known = rc.index
        if len(known) < 2:
            return out
        rv = rc.reindex(known, fill_value=0).to_numpy()
        cv = cc.reindex(known, fill_value=0).to_numpy()  # shift among known values only
        if cv.sum() == 0:
            return out
        value = psi_from_counts(rv, cv)
        table = np.vstack([rv, cv])
        p = float(stats.chi2_contingency(table[:, table.sum(axis=0) > 0]).pvalue)
        if value >= cfg.psi_threshold and p < alpha:
            explanation = f"The mix of values moved (PSI {value:.2f}; chi-squared p={p:.1e})."
            metrics: dict[str, float | int | str] = {"psi": round(value, 4), "chi2_p_value": p}
            out.append(
                self._shift_finding(column, dataset, value, metrics, explanation, is_target, "drift.psi_chi2")
            )
        return out

    def _shift_finding(
        self,
        column: str,
        dataset: str,
        value: float,
        metrics: dict[str, float | int | str],
        explanation: str,
        is_target: bool,
        rule_id: str,
    ) -> Finding:
        cfg = self.config
        if is_target:
            return data_finding(
                detector=self.id,
                type="data.drift.label_prior",
                title=f"Label {column!r} is distributed differently than in the reference",
                severity=Severity.medium,
                confidence=_confidence(value, cfg.psi_threshold),
                dataset=dataset,
                columns=[column],
                rule_id=rule_id,
                explanation=explanation + " A model calibrated on the old class balance is now "
                "miscalibrated, and thresholds tuned on it are off.",
                fix="Recalibrate or reweight for the new class balance; check whether labeling changed.",
                metrics=metrics,
                tier=Tier.tier2,
            )
        return data_finding(
            detector=self.id,
            type="data.drift.covariate",
            title=f"Column {column!r} drifted from the reference data",
            severity=Severity.high if value >= 2.5 * cfg.psi_threshold else Severity.medium,
            confidence=_confidence(value, cfg.psi_threshold),
            dataset=dataset,
            columns=[column],
            rule_id=rule_id,
            explanation=explanation,
            fix="Check the upstream source for a change (units, encoding, population); add a range or "
            "distribution check at ingestion; retrain if the change is real.",
            metrics=metrics,
            tier=Tier.tier2,
        )

    def _joint(
        self,
        ref: pd.DataFrame,
        cur: pd.DataFrame,
        features: list[str],
        flagged: set[str],
        dataset: str,
        seed: int,
    ) -> list[Finding]:
        cfg = self.config
        if not features:
            return []
        xr, xc = encode([ref, cur], features)
        rng = np.random.default_rng(seed)
        n = min(len(xr), len(xc), cfg.c2st_max_rows)
        x = np.vstack([xr[rng.choice(len(xr), n, replace=False)], xc[rng.choice(len(xc), n, replace=False)]])
        y = np.r_[np.zeros(n), np.ones(n)]
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, stratify=y, random_state=seed)
        model = HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(x_tr, y_tr)
        auc = float(roc_auc_score(y_te, model.predict_proba(x_te)[:, 1]))
        n1 = int(y_te.sum())
        p = auc_p_value(auc, len(y_te) - n1, n1)
        if auc < 0.5 + cfg.c2st_margin or p >= cfg.alpha:
            return []
        importance = permutation_importance(
            model, x_te, y_te, scoring="roc_auc", n_repeats=5, random_state=seed
        ).importances_mean
        top = [features[i] for i in np.argsort(-importance)[:5] if importance[i] > 0.005]
        if top and set(top[:3]) <= flagged:
            return []  # the per-column findings already explain the shift
        return [
            data_finding(
                detector=self.id,
                type="data.drift.covariate",
                title="Current rows can be told apart from reference rows",
                severity=Severity.high if auc >= 0.8 else Severity.medium,
                confidence=0.9 if auc >= 0.65 else 0.75,
                dataset=dataset,
                columns=top,
                rule_id="drift.c2st",
                explanation=f"A classifier separated current rows from reference rows with AUC {auc:.2f} "
                "(0.5 = indistinguishable). The columns listed are the ones it relied on; together they "
                "changed even where no single column did (e.g. a relationship between columns).",
                fix="Compare the listed columns jointly between the two datasets; check for an upstream "
                "change in how they're produced; retrain if the change is real.",
                metrics={"auc": round(auc, 4), "p_value": p, "rows_per_side": n},
                tier=Tier.tier2,
            )
        ]

    def _concept(
        self,
        ref: pd.DataFrame,
        cur: pd.DataFrame,
        features: list[str],
        target: str,
        dataset: str,
        seed: int,
        drifted: bool,
    ) -> list[Finding]:
        cfg = self.config
        ref_l, cur_l = ref[ref[target].notna()], cur[cur[target].notna()]
        if (
            not features
            or not 2 <= ref_l[target].nunique() <= cfg.max_classes
            or len(ref_l) < cfg.min_rows
            or len(cur_l) < cfg.min_rows
        ):
            return []  # regression targets are not covered yet
        xr, xc = encode([ref_l, cur_l], features)
        yr = ref_l[target].astype("string").to_numpy(dtype=object)
        yc = cur_l[target].astype("string").to_numpy(dtype=object)
        x_tr, x_ho, y_tr, y_ho = train_test_split(xr, yr, test_size=0.3, random_state=seed)
        model = HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(x_tr, y_tr)
        if len(xc) > cfg.c2st_max_rows:
            idx = np.random.default_rng(seed).choice(len(xc), cfg.c2st_max_rows, replace=False)
            xc, yc = xc[idx], yc[idx]
        k_ref = int((model.predict(x_ho) != y_ho).sum())
        k_cur = int((model.predict(xc) != yc).sum())
        e_ref, e_cur = k_ref / len(y_ho), k_cur / len(yc)
        p = increase_p_value(k_ref, len(y_ho), k_cur, len(yc))
        if e_cur - e_ref < cfg.concept_delta or p >= cfg.alpha:
            return []
        explanation = (
            f"A model fit on the reference data errs on {e_cur:.1%} of current rows vs {e_ref:.1%} of "
            "held-out reference rows: the relationship between features and label changed."
        )
        if drifted:
            explanation += " Covariate drift was also found, so part of the rise may come from inputs "
            explanation += "the model never saw rather than a changed relationship."
        return [
            data_finding(
                detector=self.id,
                type="data.drift.concept",
                title=f"Label {target!r} relates to the features differently than before",
                severity=Severity.high,
                confidence=0.6 if drifted else 0.85,
                dataset=dataset,
                columns=[target],
                rule_id="drift.concept",
                explanation=explanation,
                fix="Check whether the labeling process or definition changed; if the change is real, "
                "retrain on recent data and monitor the error rate on labeled samples.",
                metrics={"error_ref": round(e_ref, 4), "error_cur": round(e_cur, 4), "p_value": p},
                tier=Tier.tier2,
            )
        ]
