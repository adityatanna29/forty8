"""Experiment E1 — the synthetic cross-confederation anchoring probe.

The figure that makes the paper: K confederations with KNOWN latent strength
offsets; cross-confederation match rate p swept from 0.5% to 20%. For each
p, each rating system estimates between-confederation strength gaps; we plot
gap RMSE vs p. Hypothesis: shared-parameter sequence models degrade more
gracefully as p -> 0 than Elo; WHR is the honest middle.

Systems compared: Elo (forward filter) · WHR-lite (smoother) ·
Forty8Net (shared encoder, Elo-free tokens — use_elo=False so the probe
cannot leak Elo's own anchoring failure into the model).

Gap estimation per system: average pairwise cross-zone predicted win prob
-> invert through the logistic to a strength difference; compare to truth.

  python -m wc26.neural.anchoring --p-grid 0.005 0.01 0.02 0.05 0.1 0.2 \
      --n-teams-per-conf 12 --n-matches 30000 --seeds 3
Writes runs/anchoring/results.json (plot-ready).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from wc26.config import REPO_ROOT
from wc26.baselines.whr import fit_whr
from wc26.ratings.elo import run_elo

CONF_OFFSETS = (0.0, 0.45, -0.3, 0.15)  # true latent gaps (log-strength)


def make_world(n_conf: int, n_per: int, n_matches: int, p_cross: float,
               seed: int) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    teams, strength = [], []
    for c in range(n_conf):
        for i in range(n_per):
            teams.append(f"C{c}T{i:02d}")
            strength.append(CONF_OFFSETS[c % len(CONF_OFFSETS)] + rng.normal(0, 0.35))
    strength = np.array(strength)
    rows = []
    dates = pd.date_range("2008-01-01", periods=n_matches, freq="3h")
    for m in range(n_matches):
        if rng.random() < p_cross:
            ca, cb = rng.choice(n_conf, 2, replace=False)
        else:
            ca = cb = rng.integers(n_conf)
        i = int(ca) * n_per + int(rng.integers(n_per))
        j = int(cb) * n_per + int(rng.integers(n_per))
        while j == i:
            j = int(cb) * n_per + int(rng.integers(n_per))
        lam_i = np.exp(0.12 + 0.6 * (strength[i] - strength[j]))
        lam_j = np.exp(0.12 + 0.6 * (strength[j] - strength[i]))
        rows.append({"date": dates[m], "home": teams[i], "away": teams[j],
                     "home_goals": int(rng.poisson(lam_i)),
                     "away_goals": int(rng.poisson(lam_j)),
                     "tournament": "synthetic", "neutral": True, "tier": "qualifier"})
    return pd.DataFrame(rows), strength, teams


def conf_gap_rmse(est_strength: np.ndarray, true_strength: np.ndarray,
                  n_conf: int, n_per: int) -> float:
    """RMSE over between-confederation mean-gap estimates (scale-free:
    both vectors are standardized to unit within-confederation spread)."""

    def gaps(v: np.ndarray) -> np.ndarray:
        means = np.array([v[c * n_per:(c + 1) * n_per].mean() for c in range(n_conf)])
        within_sd = np.std(np.concatenate(
            [v[c * n_per:(c + 1) * n_per] - means[c] for c in range(n_conf)])) + 1e-9
        means = means / within_sd
        return np.array([means[a] - means[b]
                         for a in range(n_conf) for b in range(a + 1, n_conf)])

    return float(np.sqrt(np.mean((gaps(est_strength) - gaps(true_strength)) ** 2)))


def elo_strengths(matches: pd.DataFrame, teams: list[str]) -> np.ndarray:
    res = run_elo(matches)
    return np.array([res.ratings.get(t, 1500.0) for t in teams])


def whr_strengths(matches: pd.DataFrame, teams: list[str]) -> np.ndarray:
    model = fit_whr(matches)
    return np.array([model.skill(t) for t in teams])


def neural_strengths(matches: pd.DataFrame, teams: list[str], seed: int) -> np.ndarray:
    """Train a small Elo-free Forty8Net, then read each team's strength as
    the average logit-implied superiority against a fixed reference sample."""
    import torch

    from wc26.neural.data import build_pack, chronological_splits
    from wc26.neural.model import Forty8Net, NetConfig

    torch.manual_seed(seed)
    tr, va, _ = chronological_splits(matches, str(matches["date"].iloc[int(len(matches) * 0.8)].date()),
                                     str(matches["date"].iloc[-1].date()), min_history=3)
    pack_tr = build_pack(matches, None, tr, max_len=32, use_elo=False)
    model = Forty8Net(NetConfig(d_model=48, n_layers=2))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(pack_tr.hist), torch.from_numpy(pack_tr.mask),
        torch.from_numpy(pack_tr.ctx), torch.from_numpy(pack_tr.y),
        torch.from_numpy(pack_tr.goals))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    model.train()
    for _ in range(8):
        for hist, mask, ctx, y, goals in dl:
            opt.zero_grad()
            logits, rates = model(hist, mask, ctx)
            model.loss(logits, rates, y, goals).backward()
            opt.step()

    # probe: every team vs a common reference set at end of history
    ref = teams[:: max(1, len(teams) // 12)]
    rows = [{"date": matches["date"].max() + pd.Timedelta(days=1), "home": t,
             "away": r, "home_goals": 0, "away_goals": 0, "tournament": "probe",
             "neutral": True, "tier": "qualifier"}
            for t in teams for r in ref if r != t]
    aug = pd.concat([matches, pd.DataFrame(rows)], ignore_index=True)
    idx = np.arange(len(matches), len(aug))
    pk = build_pack(aug, None, idx, max_len=32, use_elo=False)
    model.eval()
    with torch.no_grad():
        p = model.predict_proba(torch.from_numpy(pk.hist), torch.from_numpy(pk.mask),
                                torch.from_numpy(pk.ctx)).numpy()
    superiority = np.log((p[:, 0] + 1e-6) / (p[:, 2] + 1e-6))
    out = np.zeros(len(teams)); cnt = np.zeros(len(teams))
    k = 0
    for ti, t in enumerate(teams):
        for r in ref:
            if r == t:
                continue
            out[ti] += superiority[k]; cnt[ti] += 1; k += 1
    return out / np.maximum(cnt, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-grid", type=float, nargs="+",
                    default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--n-conf", type=int, default=4)
    ap.add_argument("--n-teams-per-conf", type=int, default=12)
    ap.add_argument("--n-matches", type=int, default=30000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-neural", action="store_true")
    args = ap.parse_args()

    results = []
    for p_cross in args.p_grid:
        for seed in range(args.seeds):
            m, truth, teams = make_world(args.n_conf, args.n_teams_per_conf,
                                         args.n_matches, p_cross, seed)
            row = {"p_cross": p_cross, "seed": seed,
                   "elo": conf_gap_rmse(elo_strengths(m, teams), truth,
                                        args.n_conf, args.n_teams_per_conf),
                   "whr": conf_gap_rmse(whr_strengths(m, teams), truth,
                                        args.n_conf, args.n_teams_per_conf)}
            if not args.skip_neural:
                row["neural"] = conf_gap_rmse(neural_strengths(m, teams, seed), truth,
                                              args.n_conf, args.n_teams_per_conf)
            results.append(row)
            print(row)

    out = REPO_ROOT / "runs" / "anchoring"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"-> {out / 'results.json'} (plot gap-RMSE vs p_cross per system)")


if __name__ == "__main__":
    main()
