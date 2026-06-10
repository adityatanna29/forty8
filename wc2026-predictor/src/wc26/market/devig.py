"""Remove bookmaker overround from quoted odds.

Three methods; Shin is the default (best calibrated for sharp books;
Clarke et al. 2017). Input: decimal odds. Output: probabilities summing to 1.

Shin (1991/93) insider-trading model: with z = insider proportion and
pi_k = 1/o_k, B = sum(pi),
  p_k = ( sqrt(z^2 + 4 (1 - z) pi_k^2 / B) - z ) / (2 (1 - z))   # Eq. (Shin)
z is solved so that sum_k p_k = 1 (bisection; z in [0, 0.2] in practice).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def proportional(odds: np.ndarray) -> np.ndarray:
    pi = 1.0 / np.asarray(odds, float)
    return pi / pi.sum()


def power(odds: np.ndarray) -> np.ndarray:
    """p_k = pi_k^gamma with gamma solved so probabilities sum to 1."""
    pi = 1.0 / np.asarray(odds, float)

    def f(gamma: float) -> float:
        return np.power(pi, gamma).sum() - 1.0

    gamma = brentq(f, 0.5, 3.0, xtol=1e-12)
    return np.power(pi, gamma)


def shin(odds: np.ndarray) -> np.ndarray:
    pi = 1.0 / np.asarray(odds, float)
    b = pi.sum()
    if b <= 1.0 + 1e-12:  # no overround (already fair)
        return pi / b

    def p_of(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / b) - z) / (2.0 * (1.0 - z))

    def f(z: float) -> float:
        return p_of(z).sum() - 1.0

    z = brentq(f, 0.0, 0.5, xtol=1e-12)
    p = p_of(z)
    return p / p.sum()


METHODS = {"proportional": proportional, "power": power, "shin": shin}


def devig(odds: np.ndarray, method: str = "shin") -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"unknown de-vig method {method!r}; use one of {sorted(METHODS)}")
    return METHODS[method](np.asarray(odds, float))
