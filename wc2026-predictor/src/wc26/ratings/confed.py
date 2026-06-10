"""Cross-confederation strength calibration (iteration-5 fix).

Problem observed on real data: results-only ratings inflate confederations
whose internal games are competitive (CONMEBOL) and deflate those whose
non-tournament games are low-stakes (UEFA friendlies/rotated squads). Within-
confederation evidence cannot identify the cross-confederation scale; only
inter-confederation matches can.

Fix: with the fitted Poisson model FROZEN, estimate one strength shift s_c
per confederation (UEFA = 0 reference) by maximizing the time-decayed Poisson
likelihood of INTER-confederation matches only. A team's shift enters
symmetrically: att_i += s/2, def_i += s/2 (scores more, concedes less).

The same shifts are exposed in Elo points (~ goals * 400 / 0.65 elasticity)
so the ratings head can use them too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc26.models.poisson import PoissonModel

CONFS = ("UEFA", "CONMEBOL", "CONCACAF", "AFC", "CAF", "OFC")
_EPS = 1e-10
GOALS_PER_ELO = 0.65 / 400.0  # elasticity used only for the Elo-points view


def fit_confed_shifts(
    matches: pd.DataFrame,
    model: PoissonModel,
    confed_of: dict[str, str],
    as_of: pd.Timestamp,
    lookback_years: float = 12.0,
    half_life_days: float = 1500.0,
    l2: float = 2.0,
) -> dict[str, float]:
    """Per-confederation strength shifts (goal-log scale), UEFA = 0.

    Uses only matches where the two sides belong to different confederations,
    within `lookback_years` of `as_of`. Ridge keeps OFC-style tiny samples sane.
    """
    df = matches[
        (matches["date"] < as_of)
        & (matches["date"] >= as_of - pd.Timedelta(days=int(lookback_years * 365.25)))
    ]
    rows = []
    for _, r in df.iterrows():
        ch = confed_of.get(r["home"])
        ca = confed_of.get(r["away"])
        if ch and ca and ch != ca and r["home"] in model.index and r["away"] in model.index:
            lam, mu = model.rates(r["home"], r["away"], neutral=bool(r["neutral"]))
            w = 0.5 ** ((as_of - r["date"]).days / half_life_days)
            rows.append((CONFS.index(ch), CONFS.index(ca), lam, mu,
                         int(r["home_goals"]), int(r["away_goals"]), w))
    if len(rows) < 50:
        return {c: 0.0 for c in CONFS}

    ch = np.array([r[0] for r in rows]); ca = np.array([r[1] for r in rows])
    lam0 = np.array([r[2] for r in rows]); mu0 = np.array([r[3] for r in rows])
    x = np.array([r[4] for r in rows], float); y = np.array([r[5] for r in rows], float)
    w = np.array([r[6] for r in rows])

    def nll(s5: np.ndarray) -> float:
        s = np.concatenate([[0.0], s5])  # UEFA reference
        d = s[ch] - s[ca]                # net shift, home minus away
        # att_i += s_i and def_i += s_i  =>  lam_h *= exp(s_h - s_a), lam_a *= exp(s_a - s_h)
        lam = lam0 * np.exp(d)
        mu = mu0 * np.exp(-d)
        ll = w * (x * np.log(lam + _EPS) - lam + y * np.log(mu + _EPS) - mu)
        return -float(ll.sum()) + l2 * float(s5 @ s5)

    res = minimize(nll, np.zeros(5), method="L-BFGS-B")
    s = np.concatenate([[0.0], res.x])
    return {c: float(v) for c, v in zip(CONFS, s)}


def apply_shifts(model: PoissonModel, shifts: dict[str, float],
                 confed_of: dict[str, str]) -> PoissonModel:
    """Return a copy of the model with att/def adjusted by the shifts."""
    att = model.att.copy()
    deff = model.def_.copy()
    for t, i in model.index.items():
        s = shifts.get(confed_of.get(t, ""), 0.0)
        att[i] += s
        deff[i] += s
    return PoissonModel(teams=list(model.teams), att=att, def_=deff, mu=model.mu,
                        ha=model.ha, rho=model.rho, params=model.params)


def shifts_in_elo_points(shifts: dict[str, float]) -> dict[str, float]:
    return {c: v / GOALS_PER_ELO for c, v in shifts.items()}
