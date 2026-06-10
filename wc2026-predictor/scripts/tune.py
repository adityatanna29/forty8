"""Hyperparameter tuning over the 4-tournament backtest (iteration 5).

Grid: Poisson half-life x confederation calibration on/off. Objective: mean
walk-forward log-loss + RPS across WC 2014/18/22 + Euro 2024. Writes the best
half-life back recommendation to stdout (update configs/default.yaml by hand
— config changes should be deliberate, not silent).

Run (after data download):  python scripts/tune.py
Takes a few minutes (each cell refits 4 Poisson models).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc26.data.normalize import load_matches
from wc26.eval.backtest import WC_WINDOWS, run_backtests
from wc26.models.poisson import PoissonParams
from wc26.pipeline import load_confederations
from wc26.config import REPO_ROOT, load_config


def main() -> None:
    cfg = load_config()
    matches = load_matches()
    confed_of = load_confederations(REPO_ROOT / cfg["paths"]["static"])

    results = []
    for half_life in (365, 540, 730, 1095, 1460):
        for use_confed in (True, False):
            table = run_backtests(
                matches, WC_WINDOWS,
                poisson_params=PoissonParams(half_life_days=half_life),
                confed_of=confed_of if use_confed else None,
            )
            row = {
                "half_life": half_life,
                "confed_cal": use_confed,
                "mean_model_ll": table["model_ll"].mean(),
                "mean_model_rps": table["model_rps"].mean(),
                "mean_elo_ll": table["elo_ll"].mean(),
                "wins_vs_elo": int((table["model_ll"] < table["elo_ll"]).sum()),
            }
            results.append(row)
            print(f"hl={half_life:5d} confed={use_confed!s:5s} "
                  f"ll={row['mean_model_ll']:.4f} rps={row['mean_model_rps']:.4f} "
                  f"beats_elo {row['wins_vs_elo']}/4")

    best = min(results, key=lambda r: r["mean_model_ll"])
    print("\nBEST:", best)
    print("-> set poisson.half_life_days in configs/default.yaml accordingly.")


if __name__ == "__main__":
    main()
