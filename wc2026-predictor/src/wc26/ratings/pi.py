"""Pi-ratings, single-rating international variant.

Constantinou & Fenton (2013, JQAS): ratings live in GOAL units; prediction
error is log-dampened so blowouts don't dominate. The original maintains
separate home/away ratings per team; for internationals (mostly neutral
venues) we collapse to one rating plus a learned venue offset.

  expected goal diff   g_hat = R_h - R_a + ha * (not neutral)
  error                e = g_obs - g_hat
  dampened             psi(e) = sign(e) * c * log10(1 + |e|)     # Eq. (3)-(4)
  update               R_h += lr * psi(e);  R_a -= lr * psi(e)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PiParams:
    learning_rate: float = 0.06
    dampening_c: float = 3.0
    home_advantage_goals: float = 0.3


@dataclass
class PiResult:
    ratings: dict[str, float]
    history: pd.DataFrame  # pre-match pi_home, pi_away per row


def _psi(err: float, c: float) -> float:
    return float(np.sign(err) * c * np.log10(1.0 + abs(err)))


def run_pi(matches: pd.DataFrame, params: PiParams | None = None) -> PiResult:
    p = params or PiParams()
    ratings: dict[str, float] = {}
    n = len(matches)
    ph = np.empty(n); pa = np.empty(n)

    homes = matches["home"].to_numpy()
    aways = matches["away"].to_numpy()
    hg = matches["home_goals"].to_numpy()
    ag = matches["away_goals"].to_numpy()
    neutral = matches["neutral"].to_numpy()

    for i in range(n):
        h, a = homes[i], aways[i]
        r_h = ratings.setdefault(h, 0.0)
        r_a = ratings.setdefault(a, 0.0)
        ph[i], pa[i] = r_h, r_a
        ha = 0.0 if neutral[i] else p.home_advantage_goals
        g_hat = r_h - r_a + ha
        err = (int(hg[i]) - int(ag[i])) - g_hat
        upd = p.learning_rate * _psi(err, p.dampening_c)
        ratings[h] = r_h + upd
        ratings[a] = r_a - upd

    history = pd.DataFrame(
        {"date": matches["date"].to_numpy(), "home": homes, "away": aways,
         "pi_home": ph, "pi_away": pa}
    )
    return PiResult(ratings=ratings, history=history)
