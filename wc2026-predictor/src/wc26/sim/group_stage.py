"""Vectorized group-stage simulation + FIFA ranking logic.

Ranking key (FIFA Art. tiebreakers, vectorized approximation):
points -> goal difference -> goals for -> random jitter standing in for the
tail criteria (head-to-head among tied, fair play, drawing of lots). The
jitter tier only decides ties that survive pts/GD/GF, which is also where
the real criteria are near-random for forecasting purposes.
"""

from __future__ import annotations

import numpy as np

N_TEAMS = 48
GOALS_DIM = 11  # 0..10 goals per side (score grids are 11x11)


def sample_scorelines(cdf_row: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-CDF sample of flattened 11x11 scoreline grids.

    cdf_row: (121,) cumulative probs for ONE fixed pairing; u: (N,) uniforms.
    Returns (home_goals, away_goals) arrays of shape (N,).
    """
    s = np.searchsorted(cdf_row, u, side="right")
    s = np.clip(s, 0, GOALS_DIM * GOALS_DIM - 1)
    return s // GOALS_DIM, s % GOALS_DIM


def sample_scorelines_gather(cdf: np.ndarray, h: np.ndarray, a: np.ndarray,
                             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Same, for per-sim pairings. cdf: (T, T, 121); h, a: (N,) team indices."""
    rows = cdf[h, a]                      # (N, 121)
    u = rng.random((len(h), 1))
    s = (u > rows).sum(axis=1)
    s = np.clip(s, 0, GOALS_DIM * GOALS_DIM - 1)
    return s // GOALS_DIM, s % GOALS_DIM


def play_groups(
    cdf: np.ndarray,
    fixtures: list[tuple[int, int]],
    n_sims: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Play all 72 group fixtures. Returns (pts, gd, gf), each (n_sims, 48)."""
    pts = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
    gd = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
    gf = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
    for h, a in fixtures:
        hg, ag = sample_scorelines(cdf[h, a], rng.random(n_sims))
        home_win = hg > ag
        away_win = ag > hg
        draw = hg == ag
        pts[:, h] += 3 * home_win + draw
        pts[:, a] += 3 * away_win + draw
        gd[:, h] += hg - ag
        gd[:, a] += ag - hg
        gf[:, h] += hg
        gf[:, a] += ag
    return pts, gd, gf


def rank_groups(
    pts: np.ndarray, gd: np.ndarray, gf: np.ndarray,
    group_index: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-group ranking. group_index: (12, 4) team ids.

    Returns (winners, runners, thirds) each (N, 12) team ids, plus
    third_key (N, 12): the composite ranking key of each group's third,
    used to rank thirds across groups.
    """
    n = pts.shape[0]
    key = (pts.astype(np.float64) * 1e6 + gd * 1e3 + gf
           + rng.random((n, N_TEAMS)))  # jitter = residual tiebreak tier
    winners = np.empty((n, 12), dtype=np.int32)
    runners = np.empty((n, 12), dtype=np.int32)
    thirds = np.empty((n, 12), dtype=np.int32)
    third_key = np.empty((n, 12))
    for g in range(12):
        idx = group_index[g]
        order = np.argsort(-key[:, idx], axis=1)
        winners[:, g] = idx[order[:, 0]]
        runners[:, g] = idx[order[:, 1]]
        thirds[:, g] = idx[order[:, 2]]
        third_key[:, g] = np.take_along_axis(key[:, idx], order[:, 2:3], axis=1)[:, 0]
    return winners, runners, thirds, third_key


def best_thirds(third_key: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(N, 12) third keys -> (N, 8) qualified GROUP indices, best first."""
    return np.argsort(-third_key, axis=1)[:, :8].astype(np.int32)
