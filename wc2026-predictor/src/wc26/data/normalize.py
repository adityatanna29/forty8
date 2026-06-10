"""Identity resolution + match-table normalization (framework §A.4).

Single canonical team key everywhere. Two layers:
1. former_names.csv (martj42): historical renames (e.g. Zaire -> DR Congo),
   applied with date awareness collapsed to "always map to current".
2. team_aliases.json (curated): cross-source spelling variants.

``load_matches`` returns the canonical match table used by every rating and
model module. Columns: date (datetime64), home, away (canonical str),
home_goals, away_goals (int), tournament (str), neutral (bool), tier (str).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wc26.config import REPO_ROOT, load_config

# Match-importance tier from martj42 `tournament` strings (drives Elo K).
_TIER_RULES: list[tuple[str, str]] = [
    ("FIFA World Cup qualification", "qualifier"),
    ("FIFA World Cup", "wc_finals"),
    ("UEFA Euro qualification", "qualifier"),
    ("UEFA Euro", "continental_finals"),
    ("Copa América", "continental_finals"),
    ("African Cup of Nations qualification", "qualifier"),
    ("African Cup of Nations", "continental_finals"),
    ("AFC Asian Cup qualification", "qualifier"),
    ("AFC Asian Cup", "continental_finals"),
    ("CONCACAF Championship qualification", "qualifier"),
    ("CONCACAF Championship", "continental_finals"),
    ("Gold Cup qualification", "qualifier"),
    ("Gold Cup", "continental_finals"),
    ("Oceania Nations Cup", "continental_finals"),
    ("UEFA Nations League", "tournament"),
    ("CONCACAF Nations League", "tournament"),
    ("Confederations Cup", "tournament"),
    ("qualification", "qualifier"),  # catch-all for other qualifiers
    ("Friendly", "friendly"),
]


def tournament_tier(name: str) -> str:
    for needle, tier in _TIER_RULES:
        if needle.lower() in name.lower():
            return tier
    return "tournament"


def load_alias_map(static_dir: Path | None = None, raw_dir: Path | None = None) -> dict[str, str]:
    """Merged alias map: curated aliases + martj42 former names."""
    cfg = load_config()
    static = static_dir or (REPO_ROOT / cfg["paths"]["static"])
    raw = raw_dir or (REPO_ROOT / cfg["paths"]["raw"])
    aliases: dict[str, str] = {}
    former = raw / "former_names.csv"
    if former.exists():
        fdf = pd.read_csv(former)
        # columns: current, former, start_date, end_date
        aliases.update(dict(zip(fdf["former"], fdf["current"])))
    with open(static / "team_aliases.json") as f:
        curated = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    aliases.update(curated)  # curated wins over former_names on conflict
    return aliases


def canonicalize(name: str, aliases: dict[str, str]) -> str:
    """Resolve to canonical, following at most 2 alias hops."""
    seen = name.strip()
    for _ in range(2):
        nxt = aliases.get(seen, seen)
        if nxt == seen:
            break
        seen = nxt
    return seen


def load_matches(raw_dir: Path | None = None, static_dir: Path | None = None) -> pd.DataFrame:
    """Canonical chronological match table from martj42 results.csv."""
    cfg = load_config()
    raw = raw_dir or (REPO_ROOT / cfg["paths"]["raw"])
    df = pd.read_csv(raw / "results.csv")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    aliases = load_alias_map(static_dir=static_dir, raw_dir=raw)
    df["home"] = df["home_team"].map(lambda t: canonicalize(t, aliases))
    df["away"] = df["away_team"].map(lambda t: canonicalize(t, aliases))
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "home": df["home"],
            "away": df["away"],
            "home_goals": df["home_score"].astype(int),
            "away_goals": df["away_score"].astype(int),
            "tournament": df["tournament"],
            "neutral": df["neutral"].astype(bool),
        }
    )
    out["tier"] = out["tournament"].map(tournament_tier)
    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    return out


def validate_wc2026_teams(matches: pd.DataFrame, static_dir: Path | None = None) -> list[str]:
    """Every 2026 team must appear in match history under its canonical name.

    Returns the list of missing teams (empty = pass). Loud failure beats a
    silently mis-keyed team whose Elo would be initial-rating garbage.
    """
    cfg = load_config()
    static = static_dir or (REPO_ROOT / cfg["paths"]["static"])
    with open(static / "groups_2026.json") as f:
        groups = json.load(f)["groups"]
    teams_2026 = {t for g in groups.values() for t in g}
    known = set(matches["home"]).union(matches["away"])
    return sorted(teams_2026 - known)
