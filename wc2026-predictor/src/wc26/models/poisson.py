"""Time-decayed attack/defence Poisson goal model with Dixon-Coles correction.

Ley, Van de Wiele & Van Eetvelde (2019): weighted MLE with exponential decay
  w(t) = 0.5 ** (delta_days / half_life)                      # Eq. (decay)
won their 10-model national-team comparison on RPS. We add:
  - Dixon-Coles low-score correction tau (Dixon & Coles 1997, Eq. 4.3)
  - ridge shrinkage + shrinkage toward confederation means (hierarchical-
    prior stand-in; identifies cross-confederation gaps from sparse data)

Rates:
  log lam_home = mu + ha * (not neutral) + att[h] - def_[a]
  log lam_away = mu               + att[a] - def_[h]

Likelihood per match (weights w_i, factorials dropped):
  ll_i = w_i * (x*log(lam) - lam + y*log(mu_g) - mu_g + log(tau(x, y)))
  tau(0,0) = 1 - lam*mu_g*rho   tau(1,0) = 1 + mu_g*rho
  tau(0,1) = 1 + lam*rho        tau(1,1) = 1 - rho        else tau = 1

Fit by L-BFGS-B with analytic gradients (numeric gradient is intractable at
~500 params; analytic fit takes ~1 s on 50k matches).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist

_EPS = 1e-10


@dataclass
class PoissonParams:
    half_life_days: float = 540.0
    l2: float = 1e-3
    confed_shrink: float = 0.5
    max_goals: int = 10
    rho_bounds: tuple[float, float] = (-0.2, 0.2)


@dataclass
class PoissonModel:
    teams: list[str]
    att: np.ndarray          # (T,)
    def_: np.ndarray         # (T,)
    mu: float
    ha: float
    rho: float
    params: PoissonParams
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {t: i for i, t in enumerate(self.teams)}

    # ---------- prediction ----------
    def rates(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        ih, ia = self.index[home], self.index[away]
        lam_h = float(np.exp(self.mu + (0.0 if neutral else self.ha) + self.att[ih] - self.def_[ia]))
        lam_a = float(np.exp(self.mu + self.att[ia] - self.def_[ih]))
        return lam_h, lam_a

    def score_grid(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """(G+1, G+1) joint scoreline probabilities, DC-corrected, sums to 1."""
        g = self.params.max_goals
        lam, mu_g = self.rates(home, away, neutral)
        k = np.arange(g + 1)
        ph = poisson_dist.pmf(k, lam)
        pa = poisson_dist.pmf(k, mu_g)
        grid = np.outer(ph, pa)
        # Dixon-Coles tau on the four low-score cells  # Eq. 4.3
        grid[0, 0] *= max(_EPS, 1.0 - lam * mu_g * self.rho)
        grid[1, 0] *= max(_EPS, 1.0 + mu_g * self.rho)
        grid[0, 1] *= max(_EPS, 1.0 + lam * self.rho)
        grid[1, 1] *= max(_EPS, 1.0 - self.rho)
        return grid / grid.sum()

    def outcome_probs(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """[P(home win), P(draw), P(home loss)] from the score grid."""
        grid = self.score_grid(home, away, neutral)
        return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])


def fit_poisson(
    matches: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    params: PoissonParams | None = None,
    confed_of: dict[str, str] | None = None,
) -> PoissonModel:
    """Weighted MLE on matches strictly before ``as_of`` (leakage guard)."""
    p = params or PoissonParams()
    df = matches if as_of is None else matches[matches["date"] < as_of]
    if as_of is None:
        as_of = df["date"].max() + pd.Timedelta(days=1)

    teams = sorted(set(df["home"]).union(df["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)

    h = df["home"].map(idx).to_numpy()
    a = df["away"].map(idx).to_numpy()
    x = df["home_goals"].to_numpy(float)
    y = df["away_goals"].to_numpy(float)
    home_flag = (~df["neutral"]).to_numpy(float)
    dt_days = (as_of - df["date"]).dt.days.to_numpy(float)
    w = 0.5 ** (dt_days / p.half_life_days)

    # Confederation shrink groups (teams without a confed entry form their own no-op group).
    groups: list[np.ndarray] = []
    if confed_of:
        by_conf: dict[str, list[int]] = {}
        for t, i in idx.items():
            by_conf.setdefault(confed_of.get(t, f"_solo_{t}"), []).append(i)
        groups = [np.array(v) for v in by_conf.values() if len(v) > 1]

    is00 = (x == 0) & (y == 0); is10 = (x == 1) & (y == 0)
    is01 = (x == 0) & (y == 1); is11 = (x == 1) & (y == 1)

    def unpack(theta: np.ndarray):
        att = theta[:T]; deff = theta[T:2 * T]
        mu, ha, rho = theta[2 * T], theta[2 * T + 1], theta[2 * T + 2]
        return att, deff, mu, ha, rho

    def nll_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        att, deff, mu, ha, rho = unpack(theta)
        eta_h = mu + ha * home_flag + att[h] - deff[a]
        eta_a = mu + att[a] - deff[h]
        lam = np.exp(np.clip(eta_h, -30, 5))
        mug = np.exp(np.clip(eta_a, -30, 5))

        tau = np.ones_like(lam)
        tau[is00] = np.maximum(_EPS, 1.0 - lam[is00] * mug[is00] * rho)
        tau[is10] = np.maximum(_EPS, 1.0 + mug[is10] * rho)
        tau[is01] = np.maximum(_EPS, 1.0 + lam[is01] * rho)
        tau[is11] = np.maximum(_EPS, 1.0 - rho)

        ll = w * (x * np.log(lam + _EPS) - lam + y * np.log(mug + _EPS) - mug + np.log(tau))

        # d ll / d eta_h and eta_a (Poisson part + DC chain rule)
        g_h = x - lam
        g_a = y - mug
        g_h[is00] += -lam[is00] * mug[is00] * rho / tau[is00]
        g_a[is00] += -lam[is00] * mug[is00] * rho / tau[is00]
        g_a[is10] += mug[is10] * rho / tau[is10]
        g_h[is01] += lam[is01] * rho / tau[is01]
        g_h *= w
        g_a *= w

        g_rho = np.zeros_like(lam)
        g_rho[is00] = -lam[is00] * mug[is00] / tau[is00]
        g_rho[is10] = mug[is10] / tau[is10]
        g_rho[is01] = lam[is01] / tau[is01]
        g_rho[is11] = -1.0 / tau[is11]
        g_rho *= w

        grad = np.zeros_like(theta)
        np.add.at(grad, h, g_h)                 # att[h]
        np.add.at(grad, T + a, -g_h)            # -def_[a]
        np.add.at(grad, a, g_a)                 # att[a]
        np.add.at(grad, T + h, -g_a)            # -def_[h]
        grad[2 * T] = g_h.sum() + g_a.sum()     # mu
        grad[2 * T + 1] = (g_h * home_flag).sum()
        grad[2 * T + 2] = g_rho.sum()

        # penalties: ridge + sum-to-zero + confederation shrink
        pen = p.l2 * (att @ att + deff @ deff) + 10.0 * (att.sum() ** 2 + deff.sum() ** 2)
        grad[:T] -= 2 * p.l2 * att + 20.0 * att.sum()
        grad[T:2 * T] -= 2 * p.l2 * deff + 20.0 * deff.sum()
        for g_idx in groups:
            for vec, off in ((att, 0), (deff, T)):
                m = vec[g_idx].mean()
                pen += p.confed_shrink * ((vec[g_idx] - m) ** 2).sum()
                grad[off + g_idx] -= 2 * p.confed_shrink * (vec[g_idx] - m) * (1 - 1 / len(g_idx))

        return -(ll.sum() - pen), -grad

    theta0 = np.zeros(2 * T + 3)
    theta0[2 * T] = np.log(max(0.5, (x.mean() + y.mean()) / 2))
    bounds = [(None, None)] * (2 * T + 2) + [p.rho_bounds]
    res = minimize(nll_grad, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500})
    att, deff, mu, ha, rho = unpack(res.x)
    return PoissonModel(teams=teams, att=att, def_=deff, mu=float(mu), ha=float(ha),
                        rho=float(rho), params=p)
