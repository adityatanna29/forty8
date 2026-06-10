"""Squad-level player features (framework v2 §3 — the evidence-backed set).

Input: a squad table with one row per player:
  team (canonical), name, position in {GK, DEF, MID, ATT}, mv_eur (market
  value), ea_overall (0-100 game rating), status in {fit, doubt, out},
  minutes_2526 (club minutes this season; optional).

Output per team: the <=8-feature family the EURO-2024 ablation supports.
Player-vs-player dyad features are deliberately absent (overfitting; see
framework v2 §1). Positional unit ratings are provided so the tree model can
learn ATT-vs-DEF interactions itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

UNITS = ("GK", "DEF", "MID", "ATT")
_TOP_K = {"GK": 1, "DEF": 4, "MID": 4, "ATT": 3}
_DOUBT_WEIGHT = 0.5  # a doubtful player counts half in availability


@dataclass(frozen=True)
class TeamFeatures:
    team: str
    log_mv: float                 # log total squad market value
    ea_mean: float                # squad mean EA rating (plus-minus proxy)
    unit_rating: dict[str, float] # top-k mean EA rating per unit
    availability_mult: float      # available-XI value / full-strength value
    minutes_load: float           # mean club minutes of likely XI (0 if absent)


def _top_k_mean(s: pd.Series, k: int) -> float:
    if s.empty:
        return float("nan")
    return float(s.nlargest(min(k, len(s))).mean())


def team_features(squad: pd.DataFrame) -> TeamFeatures:
    """Features for one team's squad table (already filtered to that team)."""
    team = str(squad["team"].iloc[0])
    if not set(squad["position"]).issubset(UNITS):
        bad = set(squad["position"]) - set(UNITS)
        raise ValueError(f"{team}: unknown positions {bad}")

    mv = squad["mv_eur"].fillna(squad["mv_eur"].median())
    log_mv = float(np.log(max(mv.sum(), 1.0)))
    ea_mean = float(squad["ea_overall"].mean())

    unit_rating = {
        u: _top_k_mean(squad.loc[squad["position"] == u, "ea_overall"], _TOP_K[u])
        for u in UNITS
    }

    # availability: best-XI value with everyone vs only available players
    weight = squad["status"].map({"fit": 1.0, "doubt": _DOUBT_WEIGHT, "out": 0.0}).fillna(1.0)
    full_xi = _best_xi_value(squad, np.ones(len(squad)))
    avail_xi = _best_xi_value(squad, weight.to_numpy())
    availability_mult = float(avail_xi / full_xi) if full_xi > 0 else 1.0

    minutes_load = 0.0
    if "minutes_2526" in squad.columns and squad["minutes_2526"].notna().any():
        minutes_load = float(squad.nlargest(11, "mv_eur")["minutes_2526"].fillna(0).mean())

    return TeamFeatures(team, log_mv, ea_mean, unit_rating, availability_mult, minutes_load)


def _best_xi_value(squad: pd.DataFrame, weight: np.ndarray) -> float:
    """Value of the best XI (1 GK + top 10 outfield by weighted market value)."""
    wv = squad["mv_eur"].fillna(0).to_numpy() * weight
    gk = squad["position"].to_numpy() == "GK"
    gk_val = wv[gk].max() if gk.any() else 0.0
    outfield = np.sort(wv[~gk])[::-1][:10]
    return float(gk_val + outfield.sum())


def match_feature_diffs(fa: TeamFeatures, fb: TeamFeatures) -> dict[str, float]:
    """Difference features (A minus B) for the hybrid model."""
    d = {
        "log_mv_diff": fa.log_mv - fb.log_mv,
        "ea_mean_diff": fa.ea_mean - fb.ea_mean,
        "availability_diff": fa.availability_mult - fb.availability_mult,
    }
    for u in UNITS:
        d[f"{u.lower()}_unit_diff"] = fa.unit_rating[u] - fb.unit_rating[u]
    return d
