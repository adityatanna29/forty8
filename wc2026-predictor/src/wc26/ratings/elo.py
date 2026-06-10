"""Elo engine with fitted constants (framework v2 §2, v3 §B.1 ratings head).

Rules follow eloratings.net (Wikipedia: World Football Elo Ratings):
  expectancy  W_e = 1 / (1 + 10^(-(R_h + HA*home - R_a)/400))      # Eq. (Elo)
  update      R' = R + K_tier * G(d) * (S - W_e)
  margin      G(d) = (1 + d)^lambda  with d = |goal diff| - 1, G(0 or 1 diff)=1
  shootouts   counted as draws (S = 0.5) -- martj42 scores are 90'/ET scores.

Differences from the folk implementation (deliberate, evidence-backed):
- K per tier, HA, and lambda are *parameters*, refit by out-of-sample log-loss
  via ``wc26.ratings.fit`` (Hvattum & Arntzen 2010; opisthokonta).
- Optional between-cycle regression toward each team's trailing mean
  (538-style RTM) applied at season boundaries (July 31).
- A Glicko-style rating deviation (RD) per team: grows with inactivity,
  shrinks with matches played. Exposed as an uncertainty *feature* only;
  it never alters the Elo update itself.

The chronological loop is a plain Python loop over ~49k rows (sequential by
construction); with O(1) dict lookups it runs in well under a second, so no
further optimization is warranted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_DAY = np.timedelta64(1, "D")


@dataclass
class EloParams:
    initial_rating: float = 1500.0
    k_by_tier: dict[str, float] = field(
        default_factory=lambda: {
            "wc_finals": 60.0,
            "continental_finals": 50.0,
            "qualifier": 40.0,
            "tournament": 30.0,
            "friendly": 20.0,
        }
    )
    home_advantage: float = 100.0
    margin_exponent: float = 0.8
    rtm_weight: float = 1.0          # 1.0 = RTM disabled
    rtm_window_years: float = 4.0
    rd_init: float = 200.0           # rating-deviation feature
    rd_min: float = 40.0
    rd_growth_per_year: float = 30.0
    rd_shrink_per_match: float = 8.0


def expected_score(r_home: float, r_away: float, home_adv: float) -> float:
    """Two-outcome Elo expectancy. # Eq. (Elo), eloratings.net"""
    return 1.0 / (1.0 + 10.0 ** (-(r_home + home_adv - r_away) / 400.0))


def margin_multiplier(goal_diff: int, lam: float) -> float:
    """G(d): 1 for <=1-goal margins, grows sub-linearly beyond."""
    d = abs(int(goal_diff))
    if d <= 1:
        return 1.0
    return float((1.0 + (d - 1)) ** lam)


@dataclass
class EloResult:
    """Final state + per-match pre-update snapshots (leakage-free features)."""

    ratings: dict[str, float]
    rd: dict[str, float]
    history: pd.DataFrame  # date, home, away, elo_home, elo_away, rd_home, rd_away, exp_home


def run_elo(matches: pd.DataFrame, params: EloParams | None = None) -> EloResult:
    """Chronological Elo over a canonical match table (see normalize.load_matches).

    history rows carry PRE-match ratings -> usable directly as features for
    the match on that row without leakage.
    """
    p = params or EloParams()
    ratings: dict[str, float] = {}
    rd: dict[str, float] = {}
    last_seen: dict[str, np.datetime64] = {}
    last_rtm_year = -1
    trailing: dict[str, list[tuple[np.datetime64, float]]] = {}

    n = len(matches)
    eh = np.empty(n); ea = np.empty(n); rh_ = np.empty(n); ra_ = np.empty(n); ex = np.empty(n)

    dates = matches["date"].to_numpy(dtype="datetime64[D]")
    homes = matches["home"].to_numpy()
    aways = matches["away"].to_numpy()
    hg = matches["home_goals"].to_numpy()
    ag = matches["away_goals"].to_numpy()
    tiers = matches["tier"].to_numpy()
    neutral = matches["neutral"].to_numpy()

    for i in range(n):
        h, a, date = homes[i], aways[i], dates[i]

        # Between-cycle RTM at season boundary (July 31 of each year).
        year = date.astype("datetime64[Y]").astype(int) + 1970
        if p.rtm_weight < 1.0 and year != last_rtm_year:
            cutoff = date - int(p.rtm_window_years * 365.25) * _DAY
            for t in list(ratings):
                hist = [r for (d0, r) in trailing.get(t, []) if d0 >= cutoff]
                if hist:
                    ratings[t] = p.rtm_weight * ratings[t] + (1 - p.rtm_weight) * float(np.mean(hist))
            last_rtm_year = year

        r_h = ratings.setdefault(h, p.initial_rating)
        r_a = ratings.setdefault(a, p.initial_rating)
        rd_h = rd.setdefault(h, p.rd_init)
        rd_a = rd.setdefault(a, p.rd_init)

        # RD: grow with inactivity since last match (uncertainty feature).
        for team, r_d in ((h, rd_h), (a, rd_a)):
            if team in last_seen:
                idle_years = float((date - last_seen[team]) / _DAY) / 365.25
                rd[team] = min(p.rd_init, r_d + p.rd_growth_per_year * idle_years)
        rd_h, rd_a = rd[h], rd[a]

        ha = 0.0 if neutral[i] else p.home_advantage
        e = expected_score(r_h, r_a, ha)

        eh[i], ea[i], rh_[i], ra_[i], ex[i] = r_h, r_a, rd_h, rd_a, e

        gd = int(hg[i]) - int(ag[i])
        s = 0.5 if gd == 0 else (1.0 if gd > 0 else 0.0)
        k = p.k_by_tier.get(str(tiers[i]), 30.0)
        delta = k * margin_multiplier(gd, p.margin_exponent) * (s - e)
        ratings[h] = r_h + delta
        ratings[a] = r_a - delta

        rd[h] = max(p.rd_min, rd_h - p.rd_shrink_per_match)
        rd[a] = max(p.rd_min, rd_a - p.rd_shrink_per_match)
        last_seen[h] = date
        last_seen[a] = date
        trailing.setdefault(h, []).append((date, ratings[h]))
        trailing.setdefault(a, []).append((date, ratings[a]))

    history = pd.DataFrame(
        {
            "date": matches["date"].to_numpy(),
            "home": homes,
            "away": aways,
            "elo_home": eh,
            "elo_away": ea,
            "rd_home": rh_,
            "rd_away": ra_,
            "exp_home": ex,
        }
    )
    return EloResult(ratings=ratings, rd=rd, history=history)
