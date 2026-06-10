"""Inverse tournament simulation (Leitner-Hornik-Zeileis trick).

Given consensus championship probabilities p_bar for the 48 teams, find
Bradley-Terry abilities a such that simulating the 2026 tournament with
  P(i beats j) = a_i / (a_i + a_j)                         # Eq. (BT)
reproduces p_bar. Fixed-point iteration with damping:
  a_i <- a_i * (p_bar_i / W_hat_i)^alpha                   # alpha ~ 0.5

The simulator here is a vectorized *approximation* of the real bracket
(group round-robin -> top 2 + 8 best thirds -> seeded single elimination).
It is used ONLY to invert outright odds into pairwise abilities; the exact
Annex C bracket lands in the Stage-5 simulator. The approximation error on
extracted abilities is second-order (abilities move smoothly with W_hat).

All sims are vectorized across N tournaments (numpy); 4k sims x 40
iterations runs in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_groups(static_dir: Path) -> dict[str, list[str]]:
    with open(static_dir / "groups_2026.json") as f:
        return json.load(f)["groups"]


def simulate_champion_probs(
    strengths: np.ndarray,
    group_index: np.ndarray,
    n_sims: int = 4000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Championship probability per team under BT play.

    strengths: (48,) positive abilities, group_index: (12, 4) team indices.
    """
    rng = rng or np.random.default_rng(0)
    n_teams = len(strengths)
    pts = np.zeros((n_sims, n_teams))

    # --- group stage: 6 pairings per group, vectorized over sims ---
    for g in range(group_index.shape[0]):
        idx = group_index[g]
        for i in range(4):
            for j in range(i + 1, 4):
                ti, tj = idx[i], idx[j]
                p_i = strengths[ti] / (strengths[ti] + strengths[tj])
                win_i = rng.random(n_sims) < p_i
                pts[:, ti] += np.where(win_i, 3, 0)
                pts[:, tj] += np.where(win_i, 0, 3)

    jitter = rng.random((n_sims, n_teams)) * 1e-3  # random tiebreak (lots)
    score = pts + jitter

    # --- qualification: per group ranks; collect winners/runners/thirds ---
    winners = np.empty((n_sims, 12), int)
    runners = np.empty((n_sims, 12), int)
    thirds = np.empty((n_sims, 12), int)
    third_scores = np.empty((n_sims, 12))
    for g in range(12):
        idx = group_index[g]
        order = np.argsort(-score[:, idx], axis=1)  # (N,4) positions in idx
        winners[:, g] = idx[order[:, 0]]
        runners[:, g] = idx[order[:, 1]]
        thirds[:, g] = idx[order[:, 2]]
        third_scores[:, g] = np.take_along_axis(score[:, idx], order[:, 2:3], axis=1)[:, 0]

    # best 8 thirds per sim
    third_rank = np.argsort(-third_scores, axis=1)[:, :8]  # (N,8) group ids
    best_thirds = np.take_along_axis(thirds, third_rank, axis=1)

    # --- seeded single-elimination over 32 slots ---
    # slot layout: interleave winners and (runners shifted by 6) so same-group
    # teams land in opposite halves; thirds fill the last 8 slots.
    field = np.concatenate(
        [winners, runners[:, list(range(6, 12)) + list(range(6))], best_thirds], axis=1
    )  # (N, 32)

    while field.shape[1] > 1:
        a = field[:, 0::2]
        b = field[:, 1::2]
        p_a = strengths[a] / (strengths[a] + strengths[b])
        win_a = rng.random(p_a.shape) < p_a
        field = np.where(win_a, a, b)

    champ = field[:, 0]
    counts = np.bincount(champ, minlength=n_teams)
    return counts / n_sims


def invert_outrights(
    target: pd.Series,
    groups: dict[str, list[str]],
    alpha: float = 0.5,
    n_iter: int = 40,
    n_sims: int = 4000,
    seed: int = 26,
) -> pd.Series:
    """Find BT abilities reproducing the consensus championship probs.

    target: per-team consensus probabilities (index = canonical team names).
    Returns abilities normalized to mean 1.
    """
    teams = [t for g in "ABCDEFGHIJKL" for t in groups[g]]
    missing = set(teams) - set(target.index)
    if missing:
        raise ValueError(f"target missing teams: {sorted(missing)}")
    p_bar = target.reindex(teams).to_numpy()
    p_bar = np.maximum(p_bar, 1e-5)
    p_bar = p_bar / p_bar.sum()

    group_index = np.array([[teams.index(t) for t in groups[g]] for g in "ABCDEFGHIJKL"])
    rng = np.random.default_rng(seed)
    a = p_bar / p_bar.mean()  # warm start proportional to target

    for _ in range(n_iter):
        w_hat = simulate_champion_probs(a, group_index, n_sims=n_sims, rng=rng)
        w_hat = np.maximum(w_hat, 1.0 / (n_sims * 10))
        a = a * (p_bar / w_hat) ** alpha
        a = a / a.mean()

    return pd.Series(a, index=teams, name="bt_ability")


def bt_match_probs(a_home: float, a_away: float, draw_share: float = 0.24) -> np.ndarray:
    """Map BT abilities to W/D/L by carving a draw share out of the BT split.

    draw_share is the international base draw rate; the ensemble's ordered
    logit recalibrates this head anyway, so a fixed share is acceptable here.
    """
    p_win_2way = a_home / (a_home + a_away)
    # draws are likelier between equals: scale by 1 - |p - 0.5|
    p_draw = draw_share * (1.0 - abs(p_win_2way - 0.5))
    p_w = p_win_2way * (1.0 - p_draw)
    p_l = (1.0 - p_win_2way) * (1.0 - p_draw)
    out = np.array([p_w, p_draw, p_l])
    return out / out.sum()
