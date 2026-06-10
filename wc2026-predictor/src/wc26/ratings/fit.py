"""Fit Elo constants by out-of-sample predictive likelihood.

Folk constants (K=60/50/40/30/20, HA=100, lambda=0.8) are starting values
only; we optimize (k_scale, home_advantage, margin_exponent, rtm_weight) by
Nelder-Mead on the W/D/L log-loss of an ordered logit evaluated on matches
after ``eval_start`` (Hvattum & Arntzen 2010 protocol).

k_scale multiplies all tier Ks jointly: it controls update *speed*; the tier
*ratios* are kept (separately fitting 5 Ks adds variance for little gain on
the flat loss surface reported by opisthokonta).
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc26.models.ordered_logit import OrderedLogit
from wc26.ratings.elo import EloParams, run_elo


def _objective(matches: pd.DataFrame, params: EloParams, eval_start: pd.Timestamp) -> float:
    res = run_elo(matches, params)
    hist = res.history
    dr = (hist["elo_home"] - hist["elo_away"]).to_numpy()
    home = (~matches["neutral"]).to_numpy(float)
    gd = matches["home_goals"].to_numpy() - matches["away_goals"].to_numpy()
    outcome = np.where(gd > 0, 0, np.where(gd == 0, 1, 2))
    is_eval = (matches["date"] >= eval_start).to_numpy()
    if is_eval.sum() < 50:
        raise ValueError("eval window too small")
    ol = OrderedLogit().fit(dr[~is_eval], home[~is_eval], outcome[~is_eval])
    p = ol.predict(dr[is_eval], home[is_eval])
    return -float(np.mean(np.log(p[np.arange(is_eval.sum()), outcome[is_eval]] + 1e-12)))


def fit_elo_params(
    matches: pd.DataFrame,
    eval_start: str | pd.Timestamp = "2010-01-01",
    base: EloParams | None = None,
    out_path: Path | None = None,
    maxfev: int = 120,
) -> tuple[EloParams, float]:
    """Return (fitted params, eval log-loss). Optionally persist to JSON."""
    base = base or EloParams()
    eval_start = pd.Timestamp(eval_start)
    base_k = dict(base.k_by_tier)

    def make(theta: np.ndarray) -> EloParams:
        k_scale, ha, lam, rtm = theta
        k_scale = float(np.clip(k_scale, 0.1, 4.0))
        return replace(
            base,
            k_by_tier={t: v * k_scale for t, v in base_k.items()},
            home_advantage=float(np.clip(ha, 0.0, 300.0)),
            margin_exponent=float(np.clip(lam, 0.0, 2.0)),
            rtm_weight=float(np.clip(rtm, 0.5, 1.0)),
        )

    def f(theta: np.ndarray) -> float:
        return _objective(matches, make(theta), eval_start)

    x0 = np.array([1.0, base.home_advantage, base.margin_exponent, 1.0])
    res = minimize(f, x0, method="Nelder-Mead",
                   options={"xatol": 0.02, "fatol": 1e-4, "maxfev": maxfev})
    fitted = make(res.x)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"params": asdict(fitted), "eval_logloss": float(res.fun)}, indent=2))
    return fitted, float(res.fun)
