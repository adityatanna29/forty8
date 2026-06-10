"""End-to-end: synthetic history over the REAL 48 teams -> full 2026 artifact."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wc26.config import load_config
from wc26.pipeline import predict_2026

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_team_history():
    with open(ROOT / "data/static/groups_2026.json") as f:
        groups = json.load(f)["groups"]
    teams = [t for g in groups.values() for t in g]
    rng = np.random.default_rng(5)
    # (strengths assigned after shuffle below)
    raw = np.linspace(1.0, -1.0, len(teams))
    rng.shuffle(raw)  # shuffle so within-group strength gaps are large
    strength = {t: s for t, s in zip(teams, raw)}
    rows = []
    dates = pd.date_range("2018-01-01", periods=6000, freq="11h")
    for i in range(6000):
        h, a = rng.choice(teams, 2, replace=False)
        lam_h = np.exp(0.1 + 0.6 * (strength[h] - strength[a]))
        lam_a = np.exp(0.1 + 0.6 * (strength[a] - strength[h]))
        rows.append({"date": dates[i], "home": h, "away": a,
                     "home_goals": int(rng.poisson(lam_h)), "away_goals": int(rng.poisson(lam_a)),
                     "tournament": "Friendly", "neutral": True, "tier": "friendly"})
    return pd.DataFrame(rows), strength


def test_predict_2026_artifact(real_team_history, tmp_path):
    matches, strength = real_team_history
    cfg = load_config()
    artifact = predict_2026(matches, cfg, ROOT / "data/static", tmp_path,
                            as_of=pd.Timestamp("2026-06-11"))

    fixtures = artifact["group_fixtures"]
    assert len(fixtures) == 72, "12 groups x 6 fixtures"
    for fx in fixtures:
        p = fx["probs"]
        total = p["win"] + p["draw"] + p["loss"]
        assert abs(total - 1.0) < 1e-6
        assert all(v > 0 for v in p.values())
        if fx["top_scorelines"]:
            assert len(fx["top_scorelines"]) == 5

    # 48 rated teams, all with elo and pi
    assert len(artifact["ratings"]) == 48
    assert all(v["elo"] is not None and v["pi"] is not None for v in artifact["ratings"].values())

    # sanity: in any fixture, the much stronger synthetic team is favored
    by_team = {t: strength[t] for t in strength}
    checked = 0
    for fx in fixtures:
        gap = by_team[fx["home"]] - by_team[fx["away"]]
        if gap > 0.8:
            assert fx["probs"]["win"] > fx["probs"]["loss"]
            checked += 1
    assert checked > 0

    # pool weights on simplex
    w = artifact["pool"]["weights"]
    assert abs(sum(w) - 1.0) < 1e-9
    assert artifact["pool"]["temperature"] > 0


def test_simulate_2026_artifact(real_team_history):
    from wc26.pipeline import simulate_2026

    matches, strength = real_team_history
    cfg = load_config()
    sim = simulate_2026(matches, cfg, ROOT / "data/static", n_sims=4000, seed=1)
    assert sim["n_sims"] == 4000
    assert sim["allocation_fallbacks"] == 0
    assert len(sim["teams"]) == 48
    champ_total = sum(d["champion"] for d in sim["teams"].values())
    assert abs(champ_total - 1.0) < 1e-9
    r32_total = sum(d["r32"] for d in sim["teams"].values())
    assert abs(r32_total - 32.0) < 1e-9
    # strongest synthetic team in top-5 title odds
    best = max(strength, key=strength.get)
    top5 = sorted(sim["teams"], key=lambda t: -sim["teams"][t]["champion"])[:5]
    assert best in top5
    # template flag surfaces so the site can warn
    assert "bracket_is_template" in sim
