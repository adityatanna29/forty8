import numpy as np
import pandas as pd
import pytest

from synthetic import make_matches
from wc26.models.ordered_logit import OrderedLogit
from wc26.models.poisson import PoissonParams, fit_poisson
from wc26.ratings.pi import run_pi


@pytest.fixture(scope="module")
def synth():
    return make_matches(n=4000, seed=7)


def test_pi_recovers_ordering(synth):
    matches, strength = synth
    res = run_pi(matches)
    best = max(strength, key=strength.get)
    worst = min(strength, key=strength.get)
    assert res.ratings[best] > res.ratings[worst]
    # rank correlation with truth should be strong
    teams = list(strength)
    true = np.array([strength[t] for t in teams])
    est = np.array([res.ratings[t] for t in teams])
    rho = np.corrcoef(np.argsort(np.argsort(true)), np.argsort(np.argsort(est)))[0, 1]
    assert rho > 0.8


def test_ordered_logit_probs_valid_and_monotone():
    ol = OrderedLogit()
    dr = np.array([-400.0, 0.0, 400.0])
    home = np.zeros(3)
    p = ol.predict(dr, home)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()
    # win prob increases with rating diff
    assert p[2, 0] > p[1, 0] > p[0, 0]
    # loss prob decreases
    assert p[2, 2] < p[1, 2] < p[0, 2]


def test_ordered_logit_fit_beats_uniform(synth):
    matches, _ = synth
    # cheap rating diff: pi ratings (pre-match)
    pires = run_pi(matches)
    h = pires.history
    dr = (h["pi_home"] - h["pi_away"]).to_numpy() * 400  # arbitrary scale
    home = (~matches["neutral"]).to_numpy(float)
    gd = matches["home_goals"].to_numpy() - matches["away_goals"].to_numpy()
    y = np.where(gd > 0, 0, np.where(gd == 0, 1, 2))
    cut = len(matches) // 2
    ol = OrderedLogit().fit(dr[:cut], home[:cut], y[:cut])
    p = ol.predict(dr[cut:], home[cut:])
    ll = -np.mean(np.log(p[np.arange(len(y) - cut), y[cut:]] + 1e-12))
    assert ll < np.log(3.0), f"log-loss {ll:.3f} not better than uniform"


def test_poisson_fit_and_grid(synth):
    matches, strength = synth
    model = fit_poisson(matches, params=PoissonParams(half_life_days=10_000, l2=1e-3))
    best = max(strength, key=strength.get)
    worst = min(strength, key=strength.get)
    # attack of best > attack of worst; defence likewise better (higher def_)
    assert model.att[model.index[best]] > model.att[model.index[worst]]
    grid = model.score_grid(best, worst)
    assert grid.shape == (11, 11)
    assert abs(grid.sum() - 1.0) < 1e-9
    p = model.outcome_probs(best, worst)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[0] > 0.5, "strongest team should be a clear favorite"
    # home advantage parameter learned positive (synthetic ha = 0.25)
    assert model.ha > 0.05


def test_poisson_time_decay_prefers_recent_form():
    # team Z is weak early, strong late; short half-life must rate Z higher
    rng = np.random.default_rng(3)
    rows = []
    dates = pd.date_range("2015-01-01", periods=1200, freq="36h")
    for i, d in enumerate(dates):
        z_strong = i > 600
        lam_z = 2.2 if z_strong else 0.5
        rows.append({"date": d, "home": "Z", "away": f"O{i % 6}",
                     "home_goals": int(rng.poisson(lam_z)),
                     "away_goals": int(rng.poisson(1.0)),
                     "tournament": "s", "neutral": True, "tier": "friendly"})
    m = pd.DataFrame(rows)
    short = fit_poisson(m, params=PoissonParams(half_life_days=180))
    long = fit_poisson(m, params=PoissonParams(half_life_days=100_000))
    iz_s, iz_l = short.index["Z"], long.index["Z"]
    assert short.att[iz_s] > long.att[iz_l]


def test_poisson_leakage_guard(synth):
    matches, _ = synth
    cutoff = matches["date"].iloc[2000]
    model = fit_poisson(matches, as_of=cutoff)
    # model must not know teams that only appear after the cutoff: simulate by
    # checking the training set size implied by teams list
    assert set(model.teams) <= set(matches[matches["date"] < cutoff]["home"]).union(
        matches[matches["date"] < cutoff]["away"]
    )
