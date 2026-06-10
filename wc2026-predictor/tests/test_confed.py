"""Confederation-shift calibration must recover an injected cross-conf bias."""

import numpy as np
import pandas as pd

from wc26.models.poisson import PoissonParams, fit_poisson
from wc26.ratings.confed import apply_shifts, fit_confed_shifts, shifts_in_elo_points


def _league(rng, teams, n, start, conf_boost):
    """Matches where conf B teams are secretly conf_boost better in cross games."""
    rows = []
    dates = pd.date_range(start, periods=n, freq="17h")
    for i in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        boost_h = conf_boost if h.startswith("B") else 0.0
        boost_a = conf_boost if a.startswith("B") else 0.0
        # within-conf games: boosts cancel; cross-conf: B is genuinely better
        lam_h = np.exp(0.15 + boost_h - boost_a)
        lam_a = np.exp(0.15 + boost_a - boost_h)
        rows.append({"date": dates[i], "home": h, "away": a,
                     "home_goals": int(rng.poisson(lam_h)),
                     "away_goals": int(rng.poisson(lam_a)),
                     "tournament": "x", "neutral": True, "tier": "qualifier"})
    return rows


def test_confed_shift_recovery():
    rng = np.random.default_rng(4)
    conf_a = [f"A{i}" for i in range(8)]
    conf_b = [f"B{i}" for i in range(8)]
    confed_of = {t: "UEFA" for t in conf_a} | {t: "CONMEBOL" for t in conf_b}

    # Intra-confederation play only: within each confederation the gap is
    # invisible, so a model trained on THIS data cannot anchor the cross-
    # confederation scale (sum-to-zero forces equal means -> the real-world
    # bias). This is exactly the situation the confed module corrects.
    intra = pd.DataFrame(
        _league(rng, conf_a, 1500, "2018-01-01", 0.0)
        + _league(rng, conf_b, 1500, "2018-01-02", 0.0)
    ).sort_values("date").reset_index(drop=True)

    # Sparse cross-confederation matches reveal B's true edge (0.35 log-rate)
    cross = []
    dates = pd.date_range("2019-01-01", periods=400, freq="71h")
    for i in range(400):
        h = rng.choice(conf_a if i % 2 else conf_b)
        a = rng.choice(conf_b if i % 2 else conf_a)
        edge = 0.35 if h.startswith("B") else -0.35
        cross.append({"date": dates[i], "home": h, "away": a,
                      "home_goals": int(rng.poisson(np.exp(0.15 + edge))),
                      "away_goals": int(rng.poisson(np.exp(0.15 - edge))),
                      "tournament": "x", "neutral": True, "tier": "qualifier"})
    matches = pd.concat([intra, pd.DataFrame(cross)]).sort_values("date").reset_index(drop=True)

    # model trained WITHOUT the cross evidence (the unidentified case)
    model = fit_poisson(intra, params=PoissonParams(half_life_days=100_000))
    as_of = matches["date"].max() + pd.Timedelta(days=1)
    shifts = fit_confed_shifts(matches, model, confed_of, as_of,
                               lookback_years=20, half_life_days=100_000, l2=0.5)

    # CONMEBOL (= B) must come out positive vs UEFA reference
    assert shifts["UEFA"] == 0.0
    assert shifts["CONMEBOL"] > 0.10, f"shift {shifts['CONMEBOL']:.3f} too small"

    # applying shifts must raise B-team scoring rates against A teams
    adj = apply_shifts(model, shifts, confed_of)
    lam_before, _ = model.rates("B0", "A0")
    lam_after, _ = adj.rates("B0", "A0")
    assert lam_after > lam_before

    pts = shifts_in_elo_points(shifts)
    assert pts["CONMEBOL"] > 0
