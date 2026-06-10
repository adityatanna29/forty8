"""Vectorized knockout match resolution: 90' -> extra time -> shootout.

90' scorelines come from the full DC-corrected grid (cdf). Extra time is a
~1/3-length match: a separate cdf built from rates lambda/3 (cdf_et). If
still level, the shootout is Bernoulli with
  P(home wins) = clip(0.5 + kappa * elo_diff / 400, 0.35, 0.65)
kappa defaults to 0 (the literature's defensible coin flip); a fitted kappa
can be passed in (framework v2 example C: test it on the 49 historical WC
shootouts before trusting it).
"""

from __future__ import annotations

import numpy as np

from wc26.sim.group_stage import sample_scorelines_gather


def resolve_knockout(
    cdf: np.ndarray,
    cdf_et: np.ndarray,
    h: np.ndarray,
    a: np.ndarray,
    rng: np.random.Generator,
    elo: np.ndarray | None = None,
    kappa: float = 0.0,
) -> np.ndarray:
    """Winner team index per sim for one knockout tie. h, a: (N,) team ids."""
    hg, ag = sample_scorelines_gather(cdf, h, a, rng)
    winner = np.where(hg > ag, h, a)
    level = hg == ag

    if level.any():
        he, ae = sample_scorelines_gather(cdf_et, h[level], a[level], rng)
        sub = np.where(he > ae, h[level], a[level])
        still = he == ae
        if still.any():
            hh = h[level][still]
            aa = a[level][still]
            p_home = np.full(len(hh), 0.5)
            if elo is not None and kappa != 0.0:
                p_home = np.clip(0.5 + kappa * (elo[hh] - elo[aa]) / 400.0, 0.35, 0.65)
            shoot = rng.random(len(hh)) < p_home
            sub[still] = np.where(shoot, hh, aa)
        winner[level] = sub
    return winner
