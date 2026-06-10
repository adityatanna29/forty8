import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wc26.market.consensus import consensus_probs
from wc26.market.devig import devig, shin
from wc26.market.inverse_sim import bt_match_probs, invert_outrights, simulate_champion_probs
from wc26.players.features import match_feature_diffs, team_features

ROOT = Path(__file__).resolve().parents[1]


# ---------- de-vig ----------
def test_devig_sums_to_one_all_methods():
    odds = np.array([1.5, 4.5, 7.0])  # overround ~ 1.03
    for m in ("proportional", "power", "shin"):
        p = devig(odds, m)
        assert abs(p.sum() - 1.0) < 1e-9
        assert (p > 0).all()
        # ordering preserved: shortest odds -> highest prob
        assert p[0] > p[1] > p[2]


def test_shin_reduces_longshot_bias_vs_proportional():
    # classic favorite-longshot: shin should give the longshot LESS than proportional
    odds = np.array([1.3, 5.0, 21.0])
    p_prop = devig(odds, "proportional")
    p_shin = shin(odds)
    assert p_shin[2] < p_prop[2]
    assert p_shin[0] > p_prop[0]


def test_devig_fair_odds_passthrough():
    p_true = np.array([0.5, 0.3, 0.2])
    odds = 1.0 / p_true
    assert np.allclose(shin(odds), p_true, atol=1e-9)


# ---------- consensus ----------
def test_consensus_logit_average():
    idx = ["A", "B", "C"]
    book1 = pd.Series([2.0, 3.0, 8.0], index=idx)
    book2 = pd.Series([2.1, 2.9, 7.5], index=idx)
    df = pd.DataFrame({"b1": book1, "b2": book2})
    p = consensus_probs(df)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p["A"] > p["B"] > p["C"]


# ---------- inverse simulation ----------
@pytest.fixture(scope="module")
def groups():
    with open(ROOT / "data/static/groups_2026.json") as f:
        return json.load(f)["groups"]


def test_simulator_probs_valid(groups):
    teams = [t for g in "ABCDEFGHIJKL" for t in groups[g]]
    gi = np.array([[teams.index(t) for t in groups[g]] for g in "ABCDEFGHIJKL"])
    s = np.ones(48)
    p = simulate_champion_probs(s, gi, n_sims=2000)
    assert abs(p.sum() - 1.0) < 1e-9
    # equal strengths: nobody should dominate
    assert p.max() < 0.10


def test_inverse_sim_recovers_target(groups):
    teams = [t for g in "ABCDEFGHIJKL" for t in groups[g]]
    # synthetic target: Spain heavy favorite, France second, flat tail
    target = pd.Series(0.5 / 46, index=teams)
    target["Spain"] = 0.30
    target["France"] = 0.20
    target /= target.sum()
    a = invert_outrights(target, groups, n_iter=25, n_sims=3000)
    assert a["Spain"] == a.max()
    assert a["France"] > a["Brazil"]
    # abilities positive and normalized
    assert (a > 0).all()
    assert abs(a.mean() - 1.0) < 1e-6


def test_bt_match_probs_valid():
    p = bt_match_probs(2.0, 1.0)
    assert abs(p.sum() - 1.0) < 1e-12
    assert p[0] > p[2]
    # equal abilities -> symmetric, draw maximal
    q = bt_match_probs(1.0, 1.0)
    assert abs(q[0] - q[2]) < 1e-12
    assert q[1] > p[1]


# ---------- player features ----------
def _squad(team: str, star_status: str = "fit") -> pd.DataFrame:
    rows = []
    for i in range(26):
        pos = "GK" if i < 3 else "DEF" if i < 11 else "MID" if i < 19 else "ATT"
        rows.append({"team": team, "name": f"{team}_p{i}", "position": pos,
                     "mv_eur": 5e6 + i * 3e6, "ea_overall": 70 + (i % 20),
                     "status": "fit", "minutes_2526": 2000 + 50 * i})
    rows[-1]["mv_eur"] = 1.5e8  # star striker
    rows[-1]["ea_overall"] = 91
    rows[-1]["status"] = star_status
    return pd.DataFrame(rows)


def test_team_features_shape():
    f = team_features(_squad("Spain"))
    assert f.availability_mult == 1.0
    assert set(f.unit_rating) == {"GK", "DEF", "MID", "ATT"}
    assert f.log_mv > 0


def test_star_out_reduces_availability():
    fit = team_features(_squad("Spain", "fit"))
    out = team_features(_squad("Spain", "out"))
    doubt = team_features(_squad("Spain", "doubt"))
    assert out.availability_mult < doubt.availability_mult < fit.availability_mult
    assert out.availability_mult < 0.9  # a 150M striker missing must bite


def test_match_feature_diffs_antisymmetric():
    fa = team_features(_squad("Spain"))
    fb = team_features(_squad("Uruguay", "out"))
    d_ab = match_feature_diffs(fa, fb)
    d_ba = match_feature_diffs(fb, fa)
    for k in d_ab:
        assert abs(d_ab[k] + d_ba[k]) < 1e-9
    assert d_ab["availability_diff"] > 0
