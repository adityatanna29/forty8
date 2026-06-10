"""Walk-forward tournament backtest (framework v2 §6, v3 §B).

For each target tournament window:
  1. Heads are built strictly from data before the tournament start
     (Elo/Pi histories are pre-match by construction; the Poisson model and
     the ordered logit are refit with as_of = start).
  2. Pool weights + temperature are fit on a VALIDATION window (the
     ``val_years`` before the start), never on the tournament itself.
  3. The tournament's matches are scored: pooled model vs the Elo-only
     ordered-logit baseline. Market baseline slots in when odds CSVs exist.

Output: one report dict per tournament + a combined table; serialized by
the pipeline into calibration_report.json.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wc26.eval.metrics import accuracy, log_loss, reliability_table, rps
from wc26.features import build_features
from wc26.models.ordered_logit import OrderedLogit
from wc26.models.poisson import PoissonModel, PoissonParams, fit_poisson
from wc26.models.pool import apply_temperature, fit_pool_weights, fit_temperature, log_pool
from wc26.ratings.elo import EloParams, run_elo
from wc26.ratings.pi import run_pi


@dataclass
class TournamentWindow:
    name: str               # report label, e.g. "WC 2022"
    tournament_match: str   # substring of the `tournament` column, e.g. "FIFA World Cup"
    start: pd.Timestamp
    end: pd.Timestamp


def _poisson_probs(model: PoissonModel, rows: pd.DataFrame, fallback: np.ndarray) -> np.ndarray:
    """Score-grid W/D/L per row; unseen teams fall back to the ratings head."""
    out = fallback.copy()
    for i, (_, r) in enumerate(rows.iterrows()):
        if r["home_team"] in model.index and r["away_team"] in model.index:
            out[i] = model.outcome_probs(r["home_team"], r["away_team"], neutral=r["home"] == 0.0)
    return out


def backtest_tournament(
    matches: pd.DataFrame,
    window: TournamentWindow,
    elo_params: EloParams | None = None,
    poisson_params: PoissonParams | None = None,
    confed_of: dict[str, str] | None = None,
    val_years: float = 2.0,
) -> dict:
    elo = run_elo(matches, elo_params)
    pi = run_pi(matches)
    feats = build_features(matches, elo, pi)

    is_test = (
        feats["tournament"].str.contains(window.tournament_match, case=False)
        & ~feats["tournament"].str.contains("qualification", case=False)
        & (feats["date"] >= window.start)
        & (feats["date"] <= window.end)
    ).to_numpy()
    if is_test.sum() == 0:
        raise ValueError(f"no matches found for {window.name}")

    val_start = window.start - pd.Timedelta(days=int(val_years * 365.25))
    is_val = ((feats["date"] >= val_start) & (feats["date"] < window.start)).to_numpy()
    is_train = (feats["date"] < val_start).to_numpy()

    y = feats["outcome"].to_numpy()
    dr = feats["elo_diff"].to_numpy()
    home = feats["home"].to_numpy()

    # --- heads ---
    ol = OrderedLogit().fit(dr[is_train], home[is_train], y[is_train])
    ratings_head_val = ol.predict(dr[is_val], home[is_val])
    ratings_head_test = ol.predict(dr[is_test], home[is_test])

    pois = fit_poisson(matches, as_of=window.start, params=poisson_params, confed_of=confed_of)
    pois_val = _poisson_probs(pois, feats[is_val], ratings_head_val)
    pois_test = _poisson_probs(pois, feats[is_test], ratings_head_test)

    # --- pool + temperature on validation only ---
    stack_val = np.stack([ratings_head_val, pois_val])
    w = fit_pool_weights(stack_val, y[is_val])
    t = fit_temperature(log_pool(stack_val, w), y[is_val])

    stack_test = np.stack([ratings_head_test, pois_test])
    pooled_test = apply_temperature(log_pool(stack_test, w), t)

    y_test = y[is_test]
    return {
        "tournament": window.name,
        "n_matches": int(is_test.sum()),
        "pool_weights": {"ratings": float(w[0]), "poisson": float(w[1])},
        "temperature": float(t),
        "model": {
            "log_loss": log_loss(pooled_test, y_test),
            "rps": rps(pooled_test, y_test),
            "accuracy": accuracy(pooled_test, y_test),
        },
        "elo_baseline": {
            "log_loss": log_loss(ratings_head_test, y_test),
            "rps": rps(ratings_head_test, y_test),
            "accuracy": accuracy(ratings_head_test, y_test),
        },
        "uniform_log_loss": float(np.log(3.0)),
        "reliability": reliability_table(pooled_test, y_test),
    }


def run_backtests(matches: pd.DataFrame, windows: list[TournamentWindow], **kw) -> pd.DataFrame:
    reports = [backtest_tournament(matches, w, **kw) for w in windows]
    rows = []
    for r in reports:
        rows.append(
            {"tournament": r["tournament"], "n": r["n_matches"],
             "model_ll": r["model"]["log_loss"], "elo_ll": r["elo_baseline"]["log_loss"],
             "model_rps": r["model"]["rps"], "elo_rps": r["elo_baseline"]["rps"],
             "model_acc": r["model"]["accuracy"], "elo_acc": r["elo_baseline"]["accuracy"]}
        )
    return pd.DataFrame(rows)


WC_WINDOWS = [
    TournamentWindow("WC 2014", "FIFA World Cup", pd.Timestamp("2014-06-12"), pd.Timestamp("2014-07-13")),
    TournamentWindow("WC 2018", "FIFA World Cup", pd.Timestamp("2018-06-14"), pd.Timestamp("2018-07-15")),
    TournamentWindow("WC 2022", "FIFA World Cup", pd.Timestamp("2022-11-20"), pd.Timestamp("2022-12-18")),
    TournamentWindow("Euro 2024", "UEFA Euro", pd.Timestamp("2024-06-14"), pd.Timestamp("2024-07-14")),
]
