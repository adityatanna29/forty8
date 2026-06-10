"""Ordered logit W/D/L head on rating difference (Robberechts & Davis 2018).

  P(home win)  = sigma(beta * dr + ha * home - c2)
  P(home win or draw) = sigma(beta * dr + ha * home - c1),  c1 < c2
  P(draw) = difference. Three free parameters (beta, c1, c2) + optional ha.

The proven WC-grade W/D/L calibrator: with Elo diff as single covariate it
beat bivariate Poisson and bookmakers on log-loss at WC 2002-2014.
Fit by MLE (scipy L-BFGS-B); closed-form gradient is unnecessary at this size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-12


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


@dataclass
class OrderedLogit:
    beta: float = 1.0 / 200.0   # per Elo point; scale set by fit
    ha: float = 0.3             # home (non-neutral) shift in latent units
    c1: float = -0.45
    c2: float = 0.45

    def predict(self, dr: np.ndarray, home: np.ndarray) -> np.ndarray:
        """(n,) rating diffs, (n,) home flags -> (n, 3) [P_win, P_draw, P_loss]."""
        dr = np.asarray(dr, dtype=float)
        home = np.asarray(home, dtype=float)
        z = self.beta * dr + self.ha * home
        p_w = _sigmoid(z - self.c2)
        p_wd = _sigmoid(z - self.c1)
        p_d = np.clip(p_wd - p_w, _EPS, 1.0)
        p_l = np.clip(1.0 - p_wd, _EPS, 1.0)
        out = np.stack([p_w, p_d, p_l], axis=1)
        return out / out.sum(axis=1, keepdims=True)

    def fit(self, dr: np.ndarray, home: np.ndarray, outcome: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "OrderedLogit":
        """outcome: 0 = home win, 1 = draw, 2 = home loss.

        sample_weight: optional per-match weights — use to emphasize
        tournament-grade matches so the head matches WC conditions
        (iteration-5 fix: friendlies were polluting the fit).
        """
        dr = np.asarray(dr, float); home = np.asarray(home, float)
        y = np.asarray(outcome, int)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        w = w / w.mean()

        def nll(theta: np.ndarray) -> float:
            beta, ha, c1, gap = theta
            m = OrderedLogit(beta=beta, ha=ha, c1=c1, c2=c1 + abs(gap) + 1e-6)
            p = m.predict(dr, home)
            return -float(np.mean(w * np.log(p[np.arange(len(y)), y] + _EPS)))

        x0 = np.array([self.beta, self.ha, self.c1, self.c2 - self.c1])
        res = minimize(nll, x0, method="L-BFGS-B")
        beta, ha, c1, gap = res.x
        self.beta, self.ha, self.c1, self.c2 = float(beta), float(ha), float(c1), float(c1 + abs(gap) + 1e-6)
        return self
