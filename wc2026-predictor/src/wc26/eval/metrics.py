"""Forecast quality metrics: log-loss, RPS, Brier, calibration table.

Conventions: probs is (n, 3) ordered [home win, draw, home loss];
outcome is (n,) in {0, 1, 2} with the same ordering.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def log_loss(probs: np.ndarray, outcome: np.ndarray) -> float:
    probs = np.asarray(probs, float)
    outcome = np.asarray(outcome, int)
    return -float(np.mean(np.log(probs[np.arange(len(outcome)), outcome] + _EPS)))


def rps(probs: np.ndarray, outcome: np.ndarray) -> float:
    """Ranked probability score for the ordered W/D/L outcome (lower = better)."""
    probs = np.asarray(probs, float)
    n, k = probs.shape
    obs = np.zeros_like(probs)
    obs[np.arange(n), np.asarray(outcome, int)] = 1.0
    cum_diff = np.cumsum(probs, axis=1) - np.cumsum(obs, axis=1)
    return float(np.mean(np.sum(cum_diff[:, : k - 1] ** 2, axis=1) / (k - 1)))


def brier(probs: np.ndarray, outcome: np.ndarray) -> float:
    probs = np.asarray(probs, float)
    n = len(outcome)
    obs = np.zeros_like(probs)
    obs[np.arange(n), np.asarray(outcome, int)] = 1.0
    return float(np.mean(np.sum((probs - obs) ** 2, axis=1)))


def accuracy(probs: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == np.asarray(outcome, int)))


def reliability_table(probs: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Calibration bins for the *predicted favorite* class.

    Returns rows {bin_lo, bin_hi, n, mean_pred, hit_rate}; a calibrated model
    has mean_pred ~= hit_rate per bin. Serialized into calibration_report.json.
    """
    probs = np.asarray(probs, float)
    fav = np.argmax(probs, axis=1)
    p_fav = probs[np.arange(len(fav)), fav]
    hit = (fav == np.asarray(outcome, int)).astype(float)
    edges = np.linspace(1 / 3, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_fav >= lo) & (p_fav < hi)
        if m.sum() == 0:
            continue
        rows.append(
            {"bin_lo": float(lo), "bin_hi": float(hi), "n": int(m.sum()),
             "mean_pred": float(p_fav[m].mean()), "hit_rate": float(hit[m].mean())}
        )
    return rows
