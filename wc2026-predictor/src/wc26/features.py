"""Leakage-free per-match feature assembly.

Joins the PRE-match rating snapshots (elo.history / pi.history rows are
pre-update by construction) into one model-ready table. Player-feature
diffs are merged in when a squad table is supplied (2026 only; historical
squads are out of scope for the backtest).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wc26.ratings.elo import EloResult
from wc26.ratings.pi import PiResult

FEATURES = ["elo_diff", "pi_diff", "rd_sum", "home"]


def build_features(matches: pd.DataFrame, elo: EloResult, pi: PiResult) -> pd.DataFrame:
    """One row per match: features + outcome (0 home win / 1 draw / 2 loss).

    All inputs must be row-aligned to ``matches`` (same chronological order),
    which run_elo / run_pi guarantee.
    """
    if not (len(matches) == len(elo.history) == len(pi.history)):
        raise ValueError("matches / elo / pi histories are not aligned")
    gd = matches["home_goals"].to_numpy() - matches["away_goals"].to_numpy()
    out = pd.DataFrame(
        {
            "date": matches["date"].to_numpy(),
            "home_team": matches["home"].to_numpy(),
            "away_team": matches["away"].to_numpy(),
            "tournament": matches["tournament"].to_numpy(),
            "tier": matches["tier"].to_numpy(),
            "elo_diff": (elo.history["elo_home"] - elo.history["elo_away"]).to_numpy(),
            "pi_diff": (pi.history["pi_home"] - pi.history["pi_away"]).to_numpy(),
            "rd_sum": (elo.history["rd_home"] + elo.history["rd_away"]).to_numpy(),
            "home": (~matches["neutral"]).to_numpy(float),
            "outcome": np.where(gd > 0, 0, np.where(gd == 0, 1, 2)),
        }
    )
    return out
