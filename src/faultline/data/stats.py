"""Small statistics shared by the data detectors."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

EPS = 1e-4  # floor for empty bins, so PSI stays finite


def psi_from_counts(ref: np.ndarray, cur: np.ndarray) -> float:
    r = np.clip(ref / max(float(ref.sum()), 1.0), EPS, None)
    c = np.clip(cur / max(float(cur.sum()), 1.0), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population stability index of a numeric column on the reference's quantile bins (plus one
    bin above the reference maximum). Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 large.
    Callers drop NaN first."""
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    cuts = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)[1:]))
    k = len(cuts) + 1
    r = np.bincount(np.searchsorted(cuts, ref, side="left"), minlength=k)
    c = np.bincount(np.searchsorted(cuts, cur, side="left"), minlength=k)
    return psi_from_counts(r, c)


def increase_p_value(k_ref: int, n_ref: int, k_cur: int, n_cur: int) -> float:
    """One-sided two-proportion z-test: is the current rate k_cur/n_cur above k_ref/n_ref?"""
    if n_ref == 0 or n_cur == 0:
        return 1.0
    pooled = (k_ref + k_cur) / (n_ref + n_cur)
    if pooled in (0.0, 1.0):
        return 1.0
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_ref + 1 / n_cur))
    return float(stats.norm.sf((k_cur / n_cur - k_ref / n_ref) / se))


def auc_p_value(auc: float, n0: int, n1: int) -> float:
    """One-sided p-value of an AUC above 0.5 (normal approximation of Mann-Whitney U under H0)."""
    se = math.sqrt((n0 + n1 + 1) / (12 * n0 * n1))
    return float(stats.norm.sf((auc - 0.5) / se))
