"""Penalty-shootout skill effect, fit on real shootouts (closing a TODO).

martj42 ships shootouts.csv (date, home_team, away_team, winner) — unused
until now. We merge it with pre-match Elo (leakage-free by construction) and
fit the linear-in-Elo model the simulator already uses:

  P(home wins shootout) = clip(0.5 + kappa * elo_diff / 400, 0.35, 0.65)

by MLE, with a standard error from the observed Fisher information. The
literature expects kappa ~ 0 (shootouts are near coin flips); we TEST that
instead of assuming it. ``recommended_kappa`` returns 0.0 unless the effect
is significant at |z| > 1.96 — the simulator then uses whatever this says.

Run: python -m wc26.models.shootout   (after data download)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

_EPS = 1e-9


@dataclass
class ShootoutFit:
    kappa: float
    se: float
    n: int
    z: float
    significant: bool

    @property
    def recommended_kappa(self) -> float:
        return self.kappa if self.significant else 0.0


def _prob(dr: np.ndarray, kappa: float) -> np.ndarray:
    return np.clip(0.5 + kappa * dr / 400.0, 0.35, 0.65)


def fit_kappa(elo_diff: np.ndarray, home_won: np.ndarray) -> ShootoutFit:
    """MLE for kappa given pre-shootout Elo diffs and outcomes."""
    dr = np.asarray(elo_diff, float)
    y = np.asarray(home_won, float)

    def nll(kappa: float) -> float:
        p = _prob(dr, kappa)
        return -float(np.sum(y * np.log(p + _EPS) + (1 - y) * np.log(1 - p + _EPS)))

    res = minimize_scalar(nll, bounds=(-1.0, 1.0), method="bounded")
    k = float(res.x)
    # observed information via central second difference
    h = 1e-3
    d2 = (nll(k + h) - 2 * nll(k) + nll(k - h)) / (h * h)
    se = float(1.0 / np.sqrt(max(d2, _EPS)))
    z = k / se if se > 0 else 0.0
    return ShootoutFit(kappa=k, se=se, n=len(y), z=float(z), significant=abs(z) > 1.96)


def fit_from_files(out_path: Path | None = None) -> ShootoutFit:
    """Merge shootouts.csv with pre-match Elo from the canonical history."""
    from wc26.config import REPO_ROOT, load_config
    from wc26.data.normalize import canonicalize, load_alias_map, load_matches
    from wc26.ratings.elo import run_elo

    cfg = load_config()
    raw = REPO_ROOT / cfg["paths"]["raw"]
    so = pd.read_csv(raw / "shootouts.csv")
    aliases = load_alias_map()
    for col in ("home_team", "away_team", "winner"):
        so[col] = so[col].map(lambda t: canonicalize(str(t), aliases))
    so["date"] = pd.to_datetime(so["date"])

    matches = load_matches()
    hist = run_elo(matches).history
    merged = so.merge(
        hist, left_on=["date", "home_team", "away_team"],
        right_on=["date", "home", "away"], how="inner",
    )
    fit = fit_kappa((merged["elo_home"] - merged["elo_away"]).to_numpy(),
                    (merged["winner"] == merged["home_team"]).to_numpy())
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {**asdict(fit), "recommended_kappa": fit.recommended_kappa}, indent=2))
    return fit


if __name__ == "__main__":
    from wc26.config import REPO_ROOT, load_config

    cfg = load_config()
    out = REPO_ROOT / cfg["paths"]["processed"] / "shootout_fit.json"
    f = fit_from_files(out)
    print(f"n={f.n} kappa={f.kappa:.4f} se={f.se:.4f} z={f.z:.2f} "
          f"significant={f.significant} -> simulator uses {f.recommended_kappa:.4f}")
