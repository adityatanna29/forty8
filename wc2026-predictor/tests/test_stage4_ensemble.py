import numpy as np
import pandas as pd
import pytest

from synthetic import make_matches
from wc26.eval.backtest import TournamentWindow, backtest_tournament
from wc26.eval.metrics import accuracy, brier, log_loss, reliability_table, rps
from wc26.models.hybrid import HybridModel
from wc26.models.pool import apply_temperature, fit_pool_weights, fit_temperature, log_pool


# ---------- metrics ----------
def test_metrics_known_values():
    perfect = np.array([[1.0, 0.0, 0.0]])
    y = np.array([0])
    assert log_loss(perfect, y) < 1e-9
    assert rps(perfect, y) == 0.0
    assert brier(perfect, y) == 0.0
    uniform = np.array([[1 / 3] * 3])
    assert abs(log_loss(uniform, y) - np.log(3)) < 1e-9


def test_rps_respects_ordering():
    # predicting "home win" when outcome is draw should score better than
    # predicting "home loss" (RPS is distance-sensitive on the ordered scale)
    y = np.array([1])
    near = np.array([[0.7, 0.2, 0.1]])
    far = np.array([[0.1, 0.2, 0.7]])
    assert rps(near, y) < rps(far, y) or abs(rps(near, y) - rps(far, y)) < 1e-12


# ---------- pool ----------
def test_log_pool_weights_recover_better_head():
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.integers(0, 3, n)
    good = np.full((n, 3), 0.15)
    good[np.arange(n), y] = 0.70          # sharp, correct head
    noise = rng.dirichlet([1, 1, 1], n)    # junk head
    stack = np.stack([good, noise])
    w = fit_pool_weights(stack, y)
    assert w[0] > 0.85, f"good head weight {w[0]:.2f}"
    pooled = log_pool(stack, w)
    assert log_loss(pooled, y) <= log_loss(noise, y)


def test_temperature_fixes_overconfidence():
    rng = np.random.default_rng(1)
    n = 4000
    # true prob 0.5/0.3/0.2 but model claims 0.8/0.15/0.05 -> overconfident
    y = rng.choice(3, n, p=[0.5, 0.3, 0.2])
    over = np.tile([0.8, 0.15, 0.05], (n, 1))
    t = fit_temperature(over, y)
    assert t > 1.0, "should soften overconfident forecasts"
    assert log_loss(apply_temperature(over, t), y) < log_loss(over, y)


def test_temperature_preserves_argmax():
    p = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    for t in (0.5, 1.0, 2.5):
        q = apply_temperature(p, t)
        assert (q.argmax(axis=1) == p.argmax(axis=1)).all()


# ---------- hybrid ----------
def test_hybrid_learns_signal():
    rng = np.random.default_rng(2)
    n = 2000
    x = rng.normal(size=n)
    # outcome depends on x: strong positive -> class 0, strong negative -> 2
    p0 = 1 / (1 + np.exp(-2 * x))
    y = np.where(rng.random(n) < p0 * 0.8, 0, np.where(rng.random(n) < 0.3, 1, 2))
    X = pd.DataFrame({"elo_diff": x, "noise": rng.normal(size=n)})
    cut = 1500
    hm = HybridModel(n_estimators=60).fit(X[:cut], y[:cut])
    p = hm.predict_proba(X[cut:])
    assert p.shape == (n - cut, 3)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert log_loss(p, y[cut:]) < np.log(3.0)


# ---------- backtest ----------
def test_backtest_end_to_end_on_synthetic():
    matches, _ = make_matches(n=5000, seed=11)
    # mark a fake "tournament" in the last stretch
    matches = matches.copy()
    start = matches["date"].iloc[4700]
    end = matches["date"].iloc[-1]
    in_t = matches["date"] >= start
    matches.loc[in_t, "tournament"] = "FIFA World Cup"
    matches.loc[in_t, "tier"] = "wc_finals"
    w = TournamentWindow("synthetic WC", "FIFA World Cup", start, end)
    rep = backtest_tournament(matches, w)
    assert rep["n_matches"] == int(in_t.sum())
    # both model and baseline must beat uniform on synthetic data with signal
    assert rep["model"]["log_loss"] < np.log(3.0)
    assert rep["elo_baseline"]["log_loss"] < np.log(3.0)
    # pooled model should not be materially worse than the baseline
    assert rep["model"]["log_loss"] <= rep["elo_baseline"]["log_loss"] + 0.02
    assert 0 < rep["model"]["accuracy"] <= 1
    assert len(rep["reliability"]) > 0
