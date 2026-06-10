"""Monte Carlo tournament orchestrator (framework v3 §B.1 simulation layer).

Vectorized across sims, chunked for memory. Per chunk:
  group stage -> FIFA ranking -> best 8 thirds -> cached third allocation
  (495 possible qualified-group subsets) -> bracket DAG -> ET/shootout.

Scoreline distributions are the goal model's DC-corrected grids per ordered
pair, precomputed once: cdf (48, 48, 121) and cdf_et (extra time, rates/3).
100k sims run in seconds on a laptop; no GPU required.

Optional strength uncertainty: per-chunk multiplicative noise on each
team's rates, sigma derived from the Elo RD feature -- this widens the
tails honestly (teams the data knows less about get wider outcomes) and
the chunk-to-chunk spread feeds the reported credible intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson as pd_

from wc26.models.poisson import PoissonModel
from wc26.sim.bracket import Bracket, allocate_thirds
from wc26.sim.group_stage import GOALS_DIM, best_thirds, play_groups, rank_groups
from wc26.sim.knockout import resolve_knockout

ROUNDS = ("r32", "r16", "qf", "sf", "final", "champion")
GROUPS = "ABCDEFGHIJKL"


@dataclass
class SimResult:
    teams: list[str]
    n_sims: int
    reach: dict[str, np.ndarray]      # round -> (48,) probability of reaching
    champion_ci: dict[str, tuple[float, float]]  # team -> 90% CI from chunk spread
    third_qualify: np.ndarray         # (48,) P(qualify as third)
    allocation_fallbacks: int


def build_score_cdfs(
    model: PoissonModel, teams: list[str], hosts: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute (T, T, 121) scoreline CDFs for 90' and extra time."""
    t = len(teams)
    cdf = np.zeros((t, t, GOALS_DIM * GOALS_DIM))
    cdf_et = np.zeros_like(cdf)
    et = ExtraTimeModel(model)
    for i, hteam in enumerate(teams):
        for j, ateam in enumerate(teams):
            if i == j:
                continue
            neutral = hteam not in hosts
            cdf[i, j] = model.score_grid(hteam, ateam, neutral=neutral).ravel().cumsum()
            cdf_et[i, j] = et.score_grid(hteam, ateam, neutral=neutral).ravel().cumsum()
    return cdf, cdf_et


class ExtraTimeModel:
    """Wrapper: same model, rates scaled to a 30-minute period (lambda/3)."""

    def __init__(self, base: PoissonModel, scale: float = 1.0 / 3.0):
        self.base = base
        self.scale = scale

    def score_grid(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        lam, mu = self.base.rates(home, away, neutral)
        k = np.arange(GOALS_DIM)
        ph = pd_.pmf(k, lam * self.scale)
        pa = pd_.pmf(k, mu * self.scale)
        grid = np.outer(ph, pa)
        return grid / grid.sum()


def _ref_team(ref: str, winners: np.ndarray, runners: np.ndarray,
              third_team_for_slot: dict[str, np.ndarray]) -> np.ndarray:
    if ref.startswith("1"):
        return winners[:, GROUPS.index(ref[1])]
    if ref.startswith("2"):
        return runners[:, GROUPS.index(ref[1])]
    return third_team_for_slot[ref]


def run_tournament(
    teams: list[str],
    group_index: np.ndarray,
    fixtures: list[tuple[int, int]],
    bracket: Bracket,
    cdf: np.ndarray,
    cdf_et: np.ndarray,
    n_sims: int = 100_000,
    chunk: int = 20_000,
    seed: int = 26,
    elo: np.ndarray | None = None,
    kappa: float = 0.0,
    strength_sd: np.ndarray | None = None,
    cdf_builder=None,
) -> SimResult:
    """strength_sd + cdf_builder enable parameter-uncertainty propagation:
    per chunk, team strengths are perturbed ~N(0, strength_sd) (goal-log
    scale, e.g. derived from the Elo RD feature) and the scoreline CDFs are
    rebuilt via cdf_builder(shifts) -> (cdf, cdf_et). The chunk-to-chunk
    spread then yields honest credible intervals instead of pure MC noise.
    """
    rng = np.random.default_rng(seed)
    t = len(teams)
    reach_counts = {r: np.zeros(t) for r in ROUNDS}
    third_q = np.zeros(t)
    champ_per_chunk: list[np.ndarray] = []
    fallbacks = 0
    alloc_cache: dict[int, np.ndarray | None] = {}

    done = 0
    while done < n_sims:
        n = min(chunk, n_sims - done)
        if strength_sd is not None and cdf_builder is not None:
            chunk_shifts = rng.normal(0.0, strength_sd)
            cdf, cdf_et = cdf_builder(chunk_shifts)
        pts, gd, gf = play_groups(cdf, fixtures, n, rng)
        winners, runners, thirds, third_key = rank_groups(pts, gd, gf, group_index, rng)
        q_groups = best_thirds(third_key, rng)            # (n, 8) group ids

        # reach R32 = winner, runner, or qualified third
        for arr in (winners, runners):
            for g in range(12):
                reach_counts["r32"] += np.bincount(arr[:, g], minlength=t)
        q_third_teams = np.take_along_axis(thirds, q_groups, axis=1)  # (n, 8)
        for s in range(8):
            cnt = np.bincount(q_third_teams[:, s], minlength=t)
            reach_counts["r32"] += cnt
            third_q += cnt

        # --- third allocation via 495-subset cache (bitmask key) ---
        masks = np.zeros(n, dtype=np.int64)
        for s in range(8):
            masks |= 1 << q_groups[:, s].astype(np.int64)
        slot_group = np.empty((n, 8), dtype=np.int32)  # slot k -> group id
        slot_names = [m["away"] for m in bracket.r32 if m["away"].startswith("T")]
        for mask in np.unique(masks):
            sel = masks == mask
            if mask not in alloc_cache:
                qg = [GROUPS[b] for b in range(12) if mask >> b & 1]
                alloc_cache[mask] = allocate_thirds(qg, bracket.r32)
            alloc = alloc_cache[mask]
            if alloc is None:
                fallbacks += int(sel.sum())
                qg_sorted = sorted(GROUPS[b] for b in range(12) if mask >> b & 1)
                alloc = dict(zip(slot_names, qg_sorted))
            for k, sname in enumerate(slot_names):
                slot_group[sel, k] = GROUPS.index(alloc[sname])

        # slot -> actual third TEAM id for those sims
        third_team_for_slot: dict[str, np.ndarray] = {}
        for k, sname in enumerate(slot_names):
            third_team_for_slot[sname] = thirds[np.arange(n), slot_group[:, k]]

        # --- knockout DAG ---
        match_winner: dict[str, np.ndarray] = {}
        for m in bracket.r32:
            h = _ref_team(m["home"], winners, runners, third_team_for_slot)
            a = _ref_team(m["away"], winners, runners, third_team_for_slot)
            match_winner[m["id"]] = resolve_knockout(cdf, cdf_et, h, a, rng, elo, kappa)
        round_map = {"r16": "r16", "qf": "qf", "sf": "sf", "final": "final"}
        for rnd in ("r16", "qf", "sf", "final"):
            for m in bracket.rounds[rnd]:
                h = match_winner[m["home"][2:]]
                a = match_winner[m["away"][2:]]
                reach_counts[round_map[rnd]] += np.bincount(h, minlength=t)
                reach_counts[round_map[rnd]] += np.bincount(a, minlength=t)
                match_winner[m["id"]] = resolve_knockout(cdf, cdf_et, h, a, rng, elo, kappa)
        champ = match_winner[bracket.rounds["final"][0]["id"]]
        cc = np.bincount(champ, minlength=t)
        reach_counts["champion"] += cc
        champ_per_chunk.append(cc / n)
        done += n

    reach = {r: reach_counts[r] / n_sims for r in ROUNDS}
    spread = np.stack(champ_per_chunk)
    lo = np.percentile(spread, 5, axis=0)
    hi = np.percentile(spread, 95, axis=0)
    champion_ci = {teams[i]: (float(lo[i]), float(hi[i])) for i in range(t)}
    return SimResult(
        teams=teams, n_sims=n_sims, reach=reach, champion_ci=champion_ci,
        third_qualify=third_q / n_sims, allocation_fallbacks=fallbacks,
    )
