"""Synthetic international-football generator with known ground truth.

Teams get true log-strengths; goals are Poisson with attack/defence structure.
Used by integration tests: a correct model must (a) recover the strength
ordering and (b) beat a uniform baseline on held-out log-loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = [f"T{i:02d}" for i in range(20)]


def make_matches(n: int = 4000, seed: int = 7) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    strength = {t: s for t, s in zip(TEAMS, np.linspace(0.8, -0.8, len(TEAMS)))}
    rows = []
    dates = pd.date_range("2012-01-01", periods=n, freq="12h")
    tiers = np.array(["friendly", "qualifier", "tournament", "wc_finals"])
    for i in range(n):
        h, a = rng.choice(TEAMS, size=2, replace=False)
        neutral = bool(rng.random() < 0.3)
        ha = 0.0 if neutral else 0.25  # log-scale home boost
        lam_h = np.exp(0.1 + 0.65 * (strength[h] - strength[a]) + ha)
        lam_a = np.exp(0.1 + 0.65 * (strength[a] - strength[h]))
        rows.append(
            {
                "date": dates[i],
                "home": h,
                "away": a,
                "home_goals": int(rng.poisson(lam_h)),
                "away_goals": int(rng.poisson(lam_a)),
                "tournament": "synthetic",
                "neutral": neutral,
                "tier": str(rng.choice(tiers)),
            }
        )
    return pd.DataFrame(rows), strength
