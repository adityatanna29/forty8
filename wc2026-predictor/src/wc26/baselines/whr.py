"""WHR-lite: Whole-History Rating with draws (Coulom 2008, simplified).

THE honest baseline for the cross-confederation anchoring claim: unlike Elo
(a forward filter), WHR is a SMOOTHER — a 2014 cross-confederation result
revises a team's 2010 rating, so information propagates through the sparse
match graph in both time directions. A neural model that beats Elo but not
WHR has not demonstrated anything about anchoring (see docs/RESEARCH_PROGRAM).

Simplifications vs full WHR: piecewise-constant skill per (team, year)
instead of per match-day; Davidson (1970) draw model instead of win/loss
Bradley-Terry; single global home advantage; MAP via L-BFGS with analytic
gradients instead of Newton per player.

Model:
  pi_i = exp(s_{i, year}); effective home skill adds h on non-neutral.
  D = pi_i + pi_j + nu * sqrt(pi_i * pi_j)
  P(i wins) = pi_i / D;  P(draw) = nu * sqrt(pi_i pi_j) / D            # Davidson
  prior: (s_{t,y+1} - s_{t,y}) ~ N(0, sigma2_drift) per team (Wiener)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_EPS = 1e-12


@dataclass
class WHRParams:
    sigma2_drift: float = 0.04   # skill variance per year
    init_var: float = 1.0        # weak prior tying each team's first year to 0
    maxiter: int = 300


@dataclass
class WHRModel:
    params: WHRParams
    team_years: dict[str, np.ndarray]      # team -> sorted array of active years
    offsets: dict[str, int]                # team -> first index into s
    s: np.ndarray                          # flattened (team,year) skills
    nu: float                              # Davidson draw parameter
    h: float                               # home advantage (log-skill)
    index: dict[tuple[str, int], int] = field(default_factory=dict)

    def skill(self, team: str, year: int | None = None) -> float:
        ys = self.team_years.get(team)
        if ys is None:
            return 0.0
        if year is None:
            year = int(ys[-1])
        pos = int(np.searchsorted(ys, year, side="right")) - 1
        pos = max(0, min(pos, len(ys) - 1))
        return float(self.s[self.offsets[team] + pos])

    def predict(self, home: str, away: str, neutral: bool = True,
                year: int | None = None) -> np.ndarray:
        si = self.skill(home, year) + (0.0 if neutral else self.h)
        sj = self.skill(away, year)
        pi, pj = np.exp(si), np.exp(sj)
        g = np.sqrt(pi * pj)
        d = pi + pj + self.nu * g
        return np.array([pi / d, self.nu * g / d, pj / d])


def fit_whr(matches: pd.DataFrame, params: WHRParams | None = None) -> WHRModel:
    p = params or WHRParams()
    df = matches
    years = df["date"].dt.year.to_numpy()

    # (team, year) parameter layout
    team_years: dict[str, np.ndarray] = {}
    for team_col in ("home", "away"):
        for t, y in zip(df[team_col], years):
            team_years.setdefault(t, set()).add(int(y))  # type: ignore[arg-type]
    team_years = {t: np.array(sorted(ys)) for t, ys in team_years.items()}
    offsets: dict[str, int] = {}
    n_params = 0
    for t, ys in team_years.items():
        offsets[t] = n_params
        n_params += len(ys)

    def pidx(team: str, year: int) -> int:
        ys = team_years[team]
        return offsets[team] + int(np.searchsorted(ys, year))

    i_idx = np.array([pidx(t, int(y)) for t, y in zip(df["home"], years)])
    j_idx = np.array([pidx(t, int(y)) for t, y in zip(df["away"], years)])
    gd = df["home_goals"].to_numpy() - df["away_goals"].to_numpy()
    outcome = np.where(gd > 0, 0, np.where(gd == 0, 1, 2))
    home_flag = (~df["neutral"]).to_numpy(float)

    # drift prior pairs: consecutive (team, year) params, variance ~ gap years
    prev_idx, next_idx, gap = [], [], []
    for t, ys in team_years.items():
        for k in range(len(ys) - 1):
            prev_idx.append(offsets[t] + k)
            next_idx.append(offsets[t] + k + 1)
            gap.append(max(1, int(ys[k + 1] - ys[k])))
    prev_idx = np.array(prev_idx, int); next_idx = np.array(next_idx, int)
    gap = np.array(gap, float)

    is_w = outcome == 0; is_d = outcome == 1; is_l = outcome == 2

    def nll_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        s = theta[:n_params]
        log_nu, h = theta[n_params], theta[n_params + 1]
        nu = np.exp(log_nu)

        si = s[i_idx] + h * home_flag
        sj = s[j_idx]
        pi = np.exp(np.clip(si, -20, 20)); pj = np.exp(np.clip(sj, -20, 20))
        g = np.sqrt(pi * pj)
        d = pi + pj + nu * g

        ll = np.where(is_w, np.log(pi + _EPS),
             np.where(is_d, np.log(nu * g + _EPS), np.log(pj + _EPS))) - np.log(d)

        # d log P / d si  (and sj):  win_i: 1 - dd_i; draw: 1/2 - dd_i; loss: -dd_i
        dd_i = (pi + nu * g / 2.0) / d
        dd_j = (pj + nu * g / 2.0) / d
        gi = np.where(is_w, 1.0, np.where(is_d, 0.5, 0.0)) - dd_i
        gj = np.where(is_l, 1.0, np.where(is_d, 0.5, 0.0)) - dd_j
        g_nu = np.where(is_d, 1.0, 0.0) - nu * g / d   # wrt log nu

        grad = np.zeros_like(theta)
        np.add.at(grad, i_idx, gi)
        np.add.at(grad, j_idx, gj)
        grad[n_params] = g_nu.sum()
        grad[n_params + 1] = (gi * home_flag).sum()

        # Wiener drift prior + weak anchor on every parameter (identifiability)
        diff = s[next_idx] - s[prev_idx]
        pen = float(np.sum(diff * diff / (2 * p.sigma2_drift * gap)))
        pen += float(s @ s) / (2 * p.init_var * 10)
        gpen = np.zeros(n_params)
        np.add.at(gpen, next_idx, diff / (p.sigma2_drift * gap))
        np.add.at(gpen, prev_idx, -diff / (p.sigma2_drift * gap))
        gpen += s / (p.init_var * 10)
        grad[:n_params] -= gpen

        return -(float(ll.sum()) - pen), -grad

    theta0 = np.zeros(n_params + 2)
    theta0[n_params] = np.log(max(_EPS, (is_d.mean() * 2.5) + 0.2))
    res = minimize(nll_grad, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": p.maxiter})
    s = res.x[:n_params]
    model = WHRModel(params=p, team_years=team_years, offsets=offsets, s=s,
                     nu=float(np.exp(res.x[n_params])), h=float(res.x[n_params + 1]))
    return model
