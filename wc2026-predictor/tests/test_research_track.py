"""Tests for the research track: shootout fit, WHR-lite, neural data builder
(numpy parts always run; torch-dependent model tests skip when torch absent)."""

import numpy as np
import pandas as pd
import pytest

from synthetic import make_matches
from wc26.baselines.whr import WHRParams, fit_whr
from wc26.models.shootout import fit_kappa
from wc26.neural.data import TOKEN_DIM, build_pack, chronological_splits
from wc26.ratings.elo import run_elo


# ---------- shootout ----------
def test_shootout_kappa_detects_real_effect():
    rng = np.random.default_rng(0)
    dr = rng.normal(0, 200, 4000)
    p_true = np.clip(0.5 + 0.25 * dr / 400, 0.35, 0.65)
    won = rng.random(4000) < p_true
    fit = fit_kappa(dr, won)
    assert abs(fit.kappa - 0.25) < 0.08
    assert fit.significant
    assert fit.recommended_kappa != 0.0


def test_shootout_kappa_null_recommends_zero():
    rng = np.random.default_rng(1)
    dr = rng.normal(0, 200, 300)        # realistic small-n shootout sample
    won = rng.random(300) < 0.5         # true coin flip
    fit = fit_kappa(dr, won)
    assert fit.recommended_kappa == 0.0 or abs(fit.kappa) < 0.15


# ---------- WHR-lite ----------
def test_whr_recovers_ordering_and_smooths():
    matches, strength = make_matches(n=3000, seed=13)
    model = fit_whr(matches, WHRParams(maxiter=200))
    teams = sorted(strength)
    true = np.array([strength[t] for t in teams])
    est = np.array([model.skill(t) for t in teams])
    rho = np.corrcoef(np.argsort(np.argsort(true)), np.argsort(np.argsort(est)))[0, 1]
    assert rho > 0.85, f"rank corr {rho:.2f}"
    p = model.predict(teams[0], teams[-1])
    assert abs(p.sum() - 1.0) < 1e-9
    # best vs worst must be a clear favorite
    best = max(strength, key=strength.get); worst = min(strength, key=strength.get)
    assert model.predict(best, worst)[0] > 0.5
    assert model.nu > 0 and 0 < model.predict(best, worst)[1] < 0.5


def test_whr_anchors_sparse_cross_groups():
    """Two leagues, B truly stronger, only a handful of cross matches:
    the smoother must place B's teams above A's on average."""
    rng = np.random.default_rng(5)
    a = [f"A{i}" for i in range(6)]; b = [f"B{i}" for i in range(6)]
    rows = []
    dates = pd.date_range("2015-01-01", periods=2400, freq="9h")
    k = 0
    for i in range(2300):  # intra-league
        grp = a if i % 2 else b
        h, w = rng.choice(grp, 2, replace=False)
        rows.append({"date": dates[k], "home": h, "away": w,
                     "home_goals": int(rng.poisson(1.2)), "away_goals": int(rng.poisson(1.2)),
                     "tournament": "s", "neutral": True, "tier": "qualifier"}); k += 1
    for i in range(100):   # sparse cross: B wins a lot
        h, w = rng.choice(a), rng.choice(b)
        rows.append({"date": dates[k], "home": h, "away": w,
                     "home_goals": int(rng.poisson(0.6)), "away_goals": int(rng.poisson(2.0)),
                     "tournament": "s", "neutral": True, "tier": "qualifier"}); k += 1
    m = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    model = fit_whr(m, WHRParams(maxiter=200))
    mean_a = np.mean([model.skill(t) for t in a])
    mean_b = np.mean([model.skill(t) for t in b])
    assert mean_b > mean_a + 0.15


# ---------- neural data builder (numpy only) ----------
@pytest.fixture(scope="module")
def packed():
    matches, strength = make_matches(n=1200, seed=21)
    elo = run_elo(matches)
    tr, va, te = chronological_splits(matches, "2013-06-01", "2013-10-01")
    pack = build_pack(matches, elo.history, va[:50], max_len=16)
    return matches, pack, va[:50]


def test_pack_shapes_and_masks(packed):
    matches, pack, idx = packed
    n = len(idx)
    assert pack.hist.shape == (n, 2, 16, TOKEN_DIM)
    assert pack.mask.shape == (n, 2, 16)
    assert pack.mask.any(axis=2).all(), "every side must have some history"
    assert set(np.unique(pack.y)) <= {0, 1, 2}
    # tokens beyond mask must be zero
    assert np.allclose(pack.hist[~pack.mask], 0.0)


def test_pack_no_leakage():
    """The sample match itself must never appear in either side's history.

    Construction: X-Y play daily; the FINAL match has a unique sentinel score
    (9-0). If any token encodes gf=9/4 from X's perspective (or ga=9/4 from
    Y's), the builder leaked the label.
    """
    rows = []
    for i, d in enumerate(pd.date_range("2020-01-01", periods=10, freq="D")):
        rows.append({"date": d, "home": "X", "away": "Y",
                     "home_goals": 1, "away_goals": 1,
                     "tournament": "t", "neutral": True, "tier": "friendly"})
    rows[-1]["home_goals"], rows[-1]["away_goals"] = 9, 0  # sentinel on sample
    m = pd.DataFrame(rows)
    elo = run_elo(m)
    pack = build_pack(m, elo.history, np.array([9]), max_len=16)
    toks = pack.hist[0][pack.mask[0]]
    assert not np.any(np.isclose(toks[:, 3], 9 / 4)), "label leaked into history"
    assert pack.mask[0].sum() == 18  # 9 prior matches per side
    assert pack.y[0] == 0 and pack.goals[0, 0] == 8  # clipped sentinel target


def test_chronological_splits_disjoint():
    matches, _ = make_matches(n=1000, seed=3)
    tr, va, te = chronological_splits(matches, "2013-01-01", "2013-06-01")
    assert len(set(tr) & set(va)) == 0 and len(set(va) & set(te)) == 0
    if len(tr) and len(va):
        assert matches["date"].iloc[tr.max()] < matches["date"].iloc[va.min()]


# ---------- torch-dependent ----------
def test_forty8net_forward_and_loss():
    torch = pytest.importorskip("torch")
    from wc26.neural.model import Forty8Net, NetConfig

    torch.manual_seed(0)
    model = Forty8Net(NetConfig(d_model=32, n_layers=1))
    b, t = 4, 8
    hist = torch.randn(b, 2, t, TOKEN_DIM)
    mask = torch.ones(b, 2, t, dtype=torch.bool)
    mask[:, :, 5:] = False
    ctx = torch.zeros(b, 6); ctx[:, 0] = 1
    logits, rates = model(hist, mask, ctx)
    assert logits.shape == (b, 3) and rates.shape == (b, 2)
    assert (rates > 0).all()
    p = model.predict_proba(hist, mask, ctx)
    assert torch.allclose(p.sum(dim=1), torch.ones(b), atol=1e-5)
    y = torch.tensor([0, 1, 2, 0]); goals = torch.rand(b, 2) * 3
    loss = model.loss(logits, rates, y, goals)
    loss.backward()
    assert torch.isfinite(loss)
