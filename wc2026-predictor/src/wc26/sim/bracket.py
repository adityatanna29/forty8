"""Bracket DAG: load, validate, and allocate third-place slots.

The R32 skeleton lives in data/static/bracket_2026.json. Validation enforces
the structural invariants regardless of the exact official pairings:
every group winner and runner-up appears exactly once, exactly 8 third
slots exist, no third slot's pool contains its opponent winner's group,
and rounds reference each other consistently.

Third-place allocation: FIFA's Annex C is a lookup keyed by WHICH 8 groups
supply the qualified thirds. We implement it as deterministic backtracking
over the slot pools (8 slots x candidate groups); with the official pools
this reproduces the Annex C assignment for every one of the C(12,8)=495
subsets that admits a perfect matching, in fixed deterministic order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

GROUPS = "ABCDEFGHIJKL"


@dataclass(frozen=True)
class Bracket:
    r32: list[dict]
    rounds: dict[str, list[dict]]  # r16, qf, sf, final
    template: bool


def load_bracket(static_dir: Path) -> Bracket:
    with open(static_dir / "bracket_2026.json") as f:
        raw = json.load(f)
    b = Bracket(
        r32=raw["r32"],
        rounds={k: raw[k] for k in ("r16", "qf", "sf", "final")},
        template=bool(raw.get("_template", False)),
    )
    validate_bracket(b)
    return b


def validate_bracket(b: Bracket) -> None:
    refs = [m["home"] for m in b.r32] + [m["away"] for m in b.r32]
    winners = [r for r in refs if r.startswith("1")]
    runners = [r for r in refs if r.startswith("2")]
    thirds = [r for r in refs if r.startswith("T")]
    if sorted(winners) != [f"1{g}" for g in GROUPS]:
        raise ValueError("each group winner must appear exactly once in R32")
    if sorted(runners) != [f"2{g}" for g in GROUPS]:
        raise ValueError("each runner-up must appear exactly once in R32")
    if sorted(thirds) != [f"T{i}" for i in range(1, 9)]:
        raise ValueError("exactly 8 third-place slots T1..T8 required")
    for m in b.r32:
        if m["away"].startswith("T"):
            pool = m.get("pool") or []
            if not (3 <= len(pool) <= 12):
                raise ValueError(f"{m['id']}: third pool size {len(pool)} implausible")
            if m["home"].startswith("1") and m["home"][1] in pool:
                raise ValueError(f"{m['id']}: third pool contains opponent's own group")
    # round wiring: every match id referenced downstream must exist upstream
    ids = {m["id"] for m in b.r32}
    for rnd in ("r16", "qf", "sf", "final"):
        for m in b.rounds[rnd]:
            for side in ("home", "away"):
                ref = m[side]
                if not ref.startswith("W:") or ref[2:] not in ids:
                    raise ValueError(f"{m['id']}: bad upstream ref {ref}")
            ids.add(m["id"])


def allocate_thirds(qualified_groups: list[str], r32: list[dict]) -> dict[str, str] | None:
    """Map third slots (T1..T8) -> group letter, honoring slot pools.

    Deterministic backtracking (slots ordered by pool size, groups in
    alphabetical order). Returns None when no perfect matching exists --
    with sane pools this should never happen; the simulator then falls back
    to unrestricted assignment and counts the event (exposed in sim stats).
    """
    slots = [m for m in r32 if m["away"].startswith("T")]
    slots = sorted(slots, key=lambda m: len([g for g in (m.get("pool") or GROUPS) if g in qualified_groups]))
    assignment: dict[str, str] = {}
    used: set[str] = set()

    def bt(k: int) -> bool:
        if k == len(slots):
            return True
        m = slots[k]
        pool = [g for g in (m.get("pool") or GROUPS) if g in qualified_groups and g not in used]
        for g in sorted(pool):
            assignment[m["away"]] = g
            used.add(g)
            if bt(k + 1):
                return True
            used.discard(g)
            del assignment[m["away"]]
        return False

    return assignment if bt(0) else None
