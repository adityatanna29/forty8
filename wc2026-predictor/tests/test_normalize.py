import json
from pathlib import Path

import pandas as pd

from wc26.data.normalize import canonicalize, tournament_tier

ROOT = Path(__file__).resolve().parents[1]


def test_tournament_tiers():
    assert tournament_tier("FIFA World Cup") == "wc_finals"
    assert tournament_tier("FIFA World Cup qualification") == "qualifier"
    assert tournament_tier("UEFA Euro") == "continental_finals"
    assert tournament_tier("UEFA Nations League") == "tournament"
    assert tournament_tier("Friendly") == "friendly"
    assert tournament_tier("Copa América") == "continental_finals"
    assert tournament_tier("Something Unknown Cup") == "tournament"


def test_canonicalize_hops():
    aliases = {"USA": "United States", "Zaire": "DR Congo", "Congo DR": "DR Congo"}
    assert canonicalize("USA", aliases) == "United States"
    assert canonicalize("Zaire", aliases) == "DR Congo"
    assert canonicalize("France", aliases) == "France"


def test_static_groups_shape():
    with open(ROOT / "data/static/groups_2026.json") as f:
        data = json.load(f)
    groups = data["groups"]
    assert len(groups) == 12
    assert sorted(groups) == list("ABCDEFGHIJKL")
    teams = [t for g in groups.values() for t in g]
    assert len(teams) == 48
    assert len(set(teams)) == 48, "duplicate team in groups"
    for host in data["hosts"]:
        assert host in teams


def test_confederations_cover_all_48():
    with open(ROOT / "data/static/groups_2026.json") as f:
        teams = {t for g in json.load(f)["groups"].values() for t in g}
    with open(ROOT / "data/static/confederations.json") as f:
        conf = json.load(f)
    covered = {t for k, v in conf.items() if not k.startswith("_") for t in v}
    assert teams == covered, f"mismatch: {teams ^ covered}"
