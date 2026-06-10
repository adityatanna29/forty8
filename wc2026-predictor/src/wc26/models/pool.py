"""Log-opinion pool over model heads + temperature scaling.

Log pool (weights on the simplex):
  p_pool(c) ∝ prod_m p_m(c)^{w_m}                       # Eq. (LogOP)
Equivalent to logistic stacking on component log-odds; sharper than the
linear pool and externally Bayesian (Ranjan & Gneiting 2010 show linear
pools of calibrated forecasts are miscalibrated).

Weights are fit by minimizing validation log-loss over the simplex
(softmax reparameterization, L-BFGS-B). Temperature scaling:
  p_T(c) ∝ p(c)^{1/T}
with scalar T fit by NLL on a held-out window (Guo et al. 2017 adapted to
probability space; preserves the argmax).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-12


def log_pool(prob_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """prob_stack: (M, n, 3) head probabilities; weights: (M,) on simplex."""
    logp = np.log(np.clip(prob_stack, _EPS, 1.0))
    pooled = np.einsum("m,mnc->nc", weights, logp)
    pooled -= pooled.max(axis=1, keepdims=True)
    p = np.exp(pooled)
    return p / p.sum(axis=1, keepdims=True)


def fit_pool_weights(prob_stack: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Simplex weights minimizing log-loss of the pooled forecast."""
    m = prob_stack.shape[0]
    y = np.asarray(outcome, int)

    def nll(z: np.ndarray) -> float:
        w = np.exp(z - z.max())
        w = w / w.sum()
        p = log_pool(prob_stack, w)
        return -float(np.mean(np.log(p[np.arange(len(y)), y] + _EPS)))

    res = minimize(nll, np.zeros(m), method="L-BFGS-B")
    w = np.exp(res.x - res.x.max())
    return w / w.sum()


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(probs, float), _EPS, 1.0) ** (1.0 / max(temperature, 1e-3))
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, outcome: np.ndarray) -> float:
    y = np.asarray(outcome, int)

    def nll(log_t: np.ndarray) -> float:
        p = apply_temperature(probs, float(np.exp(log_t[0])))
        return -float(np.mean(np.log(p[np.arange(len(y)), y] + _EPS)))

    res = minimize(nll, np.array([0.0]), method="L-BFGS-B", bounds=[(-1.5, 1.5)])
    return float(np.exp(res.x[0]))
