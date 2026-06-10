import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic import make_matches
from wc26.models.poisson import PoissonParams, fit_poisson
from wc26.sim.bracket import allocate_thirds, load_bracket, validate_bracket
from wc26.sim.group_stage import play_groups, rank_groups, sample_scorelines
from wc26.sim.tournament import build_score_cdfs, run_tournament

ROOT = Path(__file__).resolve().parents[1]
GROUPS = "ABCDEFGHIJKL"


@pytest.fixture(scope="module")
def bracket():
    return load_bracket(ROOT / "data/static")


def test_bracket_invariants(bracket):
    validate_bracket(bracket)  # raises on violation
    assert len(bracket.r32) == 16
    assert len(bracket.rounds["r16"]) == 8
    assert len(bracket.rounds["final"]) == 1


def test_allocate_thirds_basic(bracket):
    qualified = list("ABCDEFGH")
    alloc = allocate_thirds(qualified, bracket.r32)
    assert alloc is not None
    assert sorted(alloc.keys()) == [f"T{i}" for i in range(1, 9)]
    assert sorted(alloc.values()) == qualified
    # pool constraints respected
    pools = {m["away"]: m["pool"] for m in bracket.r32 if m["away"].startswith("T")}
    for slot, grp in alloc.items():
        assert grp in pools[slot]


def test_allocate_thirds_all_495_subsets(bracket):
    ok = 0
    for combo in itertools.combinations(GROUPS, 8):
        alloc = allocate_thirds(list(combo), bracket.r32)
        if alloc is not None:
            assert sorted(alloc.values()) == sorted(combo)
            ok += 1
    # with 6-group pools, virtually all subsets must admit a matching
    assert ok >= 490, f"only {ok}/495 subsets matched — pools too restrictive"


def test_sample_scorelines_inverse_cdf():
    grid = np.zeros((11, 11))
    grid[2, 1] = 0.7
    grid[0, 0] = 0.3
    cdf = grid.ravel().cumsum()
    rng = np.random.default_rng(0)
    hg, ag = sample_scorelines(cdf, rng.random(20000))
    frac_21 = np.mean((hg == 2) & (ag == 1))
    assert abs(frac_21 - 0.7) < 0.02


@pytest.fixture(scope="module")
def sim_setup():
    """Poisson model over the real 48 teams from synthetic history."""
    with open(ROOT / "data/static/groups_2026.json") as f:
        gdata = json.load(f)
    groups = gdata["groups"]
    teams = [t for g in GROUPS for t in groups[g]]
    rng = np.random.default_rng(8)
    raw = np.linspace(0.9, -0.9, 48)
    rng.shuffle(raw)
    strength = dict(zip(teams, raw))
    rows = []
    dates = pd.date_range("2019-01-01", periods=7000, freq="9h")
    for i in range(7000):
        h, a = rng.choice(teams, 2, replace=False)
        rows.append({"date": dates[i], "home": h, "away": a,
                     "home_goals": int(rng.poisson(np.exp(0.1 + 0.6 * (strength[h] - strength[a])))),
                     "away_goals": int(rng.poisson(np.exp(0.1 + 0.6 * (strength[a] - strength[h])))),
                     "tournament": "Friendly", "neutral": True, "tier": "friendly"})
    matches = pd.DataFrame(rows)
    model = fit_poisson(matches, params=PoissonParams(half_life_days=100_000))
    group_index = np.array([[teams.index(t) for t in groups[g]] for g in GROUPS])
    fixtures = [(teams.index(x), teams.index(y))
                for g in GROUPS for x, y in itertools.combinations(groups[g], 2)]
    cdf, cdf_et = build_score_cdfs(model, teams, hosts=set(gdata["hosts"]))
    return teams, strength, group_index, fixtures, cdf, cdf_et


def test_group_stage_conservation(sim_setup):
    teams, _, group_index, fixtures, cdf, _ = sim_setup
    rng = np.random.default_rng(1)
    pts, gd, gf = play_groups(cdf, fixtures, 500, rng)
    # goal difference sums to zero, points per group in [_min,_max]
    assert (gd.sum(axis=1) == 0).all()
    for g in range(12):
        gpts = pts[:, group_index[g]].sum(axis=1)
        assert (gpts >= 12).all() and (gpts <= 18).all(), "6 matches: 12 (all draws) to 18 pts"
    w, r, t3, _ = rank_groups(pts, gd, gf, group_index, rng)
    # winner/runner/third are distinct members of the group
    for g in range(12):
        assert not (w[:, g] == r[:, g]).any()
        assert np.isin(w[:, g], group_index[g]).all()


def test_full_tournament(sim_setup, bracket=None):
    from wc26.sim.bracket import load_bracket
    teams, strength, group_index, fixtures, cdf, cdf_et = sim_setup
    b = load_bracket(ROOT / "data/static")
    res = run_tournament(teams, group_index, fixtures, b, cdf, cdf_et,
                         n_sims=8000, chunk=4000, seed=3)
    champ = res.reach["champion"]
    assert abs(champ.sum() - 1.0) < 1e-9
    # exactly 32 reach R32, 16 reach R16, ... in every sim (probability mass)
    assert abs(res.reach["r32"].sum() - 32.0) < 1e-9
    assert abs(res.reach["r16"].sum() - 16.0) < 1e-9
    assert abs(res.reach["qf"].sum() - 8.0) < 1e-9
    assert abs(res.reach["sf"].sum() - 4.0) < 1e-9
    assert abs(res.reach["final"].sum() - 2.0) < 1e-9
    # monotonicity per team: P(r32) >= P(r16) >= ... >= P(champion)
    order = ["r32", "r16", "qf", "sf", "final", "champion"]
    for i in range(48):
        seq = [res.reach[r][i] for r in order]
        assert all(seq[j] >= seq[j + 1] - 1e-12 for j in range(5))
    # strongest synthetic team should be among top champions
    best = max(strength, key=strength.get)
    top5 = np.argsort(-champ)[:5]
    assert teams.index(best) in top5
    # third qualification only for non-winners; total = 8 per sim
    assert abs(res.third_qualify.sum() - 8.0) < 1e-9
    assert res.allocation_fallbacks == 0
    # CI sanity
    lo, hi = res.champion_ci[best]
    assert 0 <= lo <= hi <= 1
