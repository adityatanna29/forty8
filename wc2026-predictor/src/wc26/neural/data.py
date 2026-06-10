"""Sequence dataset for the Forty8Net dynamic-rating model (numpy only;
torch enters in model/train so these builders are testable everywhere).

Each training sample = one match. For each side we encode the team's last
``max_len`` internationals as a token sequence; the model produces a dynamic
team state from raw history instead of consuming a precomputed rating.

Token features (TOKEN_DIM = 14), all leakage-free at the encoded match's
own kickoff:
  0 win  1 draw  2 loss            (result from the team's perspective)
  3 gf/4 (clip 0-2.5)  4 ga/4      (scoreline, scaled)
  5 was_home  6 was_neutral
  7-11 tier one-hot (wc_finals, continental_finals, qualifier, tournament, friendly)
  12 log1p(days_since_that_match)/8   (recency)
  13 opponent pre-match Elo z-score   (ablation: --no-elo zeroes this — the
     anchoring experiments require the encoder to work WITHOUT Elo input)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TOKEN_DIM = 14
TIERS = ("wc_finals", "continental_finals", "qualifier", "tournament", "friendly")
CTX_DIM = 1 + len(TIERS)  # neutral flag + tier one-hot of the match being predicted

_ELO_MEAN, _ELO_SD = 1500.0, 200.0


@dataclass
class SequencePack:
    """Aligned arrays for n samples. hist: (n, 2, max_len, TOKEN_DIM)."""

    hist: np.ndarray
    mask: np.ndarray          # (n, 2, max_len)  True = real token
    ctx: np.ndarray           # (n, CTX_DIM)
    y: np.ndarray             # (n,) 0 home win / 1 draw / 2 home loss
    goals: np.ndarray         # (n, 2) clipped at 8
    match_index: np.ndarray   # (n,) row index into the source matches frame


def _token(row: pd.Series, persp_home: bool, asof: pd.Timestamp,
           opp_elo: float, use_elo: bool) -> np.ndarray:
    tok = np.zeros(TOKEN_DIM, dtype=np.float32)
    gf, ga = (row.home_goals, row.away_goals) if persp_home else (row.away_goals, row.home_goals)
    tok[0 if gf > ga else (1 if gf == ga else 2)] = 1.0
    tok[3] = min(gf, 10) / 4.0
    tok[4] = min(ga, 10) / 4.0
    tok[5] = 1.0 if (persp_home and not row.neutral) else 0.0
    tok[6] = 1.0 if row.neutral else 0.0
    tok[7 + TIERS.index(row.tier)] = 1.0
    tok[12] = np.log1p(max((asof - row.date).days, 0)) / 8.0
    tok[13] = ((opp_elo - _ELO_MEAN) / _ELO_SD) if use_elo else 0.0
    return tok


def build_pack(
    matches: pd.DataFrame,
    elo_history: pd.DataFrame | None,
    sample_idx: np.ndarray,
    max_len: int = 64,
    use_elo: bool = True,
) -> SequencePack:
    """Build sequences for the matches at ``sample_idx`` (chronological df).

    elo_history must be row-aligned with matches (run_elo output) when
    use_elo; pass None with use_elo=False for the Elo-free ablation.
    """
    if use_elo and elo_history is None:
        raise ValueError("use_elo=True requires elo_history")

    # per-team list of (row position, was_home) in chronological order
    team_rows: dict[str, list[tuple[int, bool]]] = {}
    homes = matches["home"].to_numpy(); aways = matches["away"].to_numpy()
    for r in range(len(matches)):
        team_rows.setdefault(homes[r], []).append((r, True))
        team_rows.setdefault(aways[r], []).append((r, False))
    # position of each appearance for O(log) cutoff lookup
    team_pos: dict[str, np.ndarray] = {
        t: np.array([r for r, _ in lst]) for t, lst in team_rows.items()
    }

    n = len(sample_idx)
    hist = np.zeros((n, 2, max_len, TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n, 2, max_len), dtype=bool)
    ctx = np.zeros((n, CTX_DIM), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    goals = np.zeros((n, 2), dtype=np.float32)

    eh = elo_history["elo_home"].to_numpy() if use_elo else None
    ea = elo_history["elo_away"].to_numpy() if use_elo else None

    for k, m in enumerate(sample_idx):
        row = matches.iloc[m]
        ctx[k, 0] = 1.0 if row.neutral else 0.0
        ctx[k, 1 + TIERS.index(row.tier)] = 1.0
        gd = row.home_goals - row.away_goals
        y[k] = 0 if gd > 0 else (1 if gd == 0 else 2)
        goals[k, 0] = min(row.home_goals, 8)
        goals[k, 1] = min(row.away_goals, 8)

        for side, team in enumerate((row.home, row.away)):
            lst = team_rows[team]
            pos = team_pos[team]
            cut = int(np.searchsorted(pos, m))           # appearances strictly before m
            take = lst[max(0, cut - max_len):cut]
            for j, (r, was_home) in enumerate(take):
                prow = matches.iloc[r]
                if use_elo:
                    opp_elo = float(ea[r] if was_home else eh[r])  # type: ignore[index]
                else:
                    opp_elo = _ELO_MEAN
                hist[k, side, j] = _token(prow, was_home, row.date, opp_elo, use_elo)
                mask[k, side, j] = True

    return SequencePack(hist=hist, mask=mask, ctx=ctx, y=y, goals=goals,
                        match_index=np.asarray(sample_idx))


def chronological_splits(matches: pd.DataFrame, train_end: str, val_end: str,
                         min_history: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/val/test row indices; samples need >= min_history prior matches
    for BOTH teams (cold-start rows are useless gradient noise)."""
    counts: dict[str, int] = {}
    ok = np.zeros(len(matches), dtype=bool)
    homes = matches["home"].to_numpy(); aways = matches["away"].to_numpy()
    for r in range(len(matches)):
        ok[r] = counts.get(homes[r], 0) >= min_history and counts.get(aways[r], 0) >= min_history
        counts[homes[r]] = counts.get(homes[r], 0) + 1
        counts[aways[r]] = counts.get(aways[r], 0) + 1
    dates = matches["date"]
    tr = np.where(ok & (dates < pd.Timestamp(train_end)).to_numpy())[0]
    va = np.where(ok & (dates >= pd.Timestamp(train_end)).to_numpy()
                  & (dates < pd.Timestamp(val_end)).to_numpy())[0]
    te = np.where(ok & (dates >= pd.Timestamp(val_end)).to_numpy())[0]
    return tr, va, te
