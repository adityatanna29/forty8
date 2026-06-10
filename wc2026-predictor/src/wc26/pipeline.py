"""End-to-end Stage 1-4 pipeline: data -> heads -> pool -> 2026 match artifacts.

Run order (user-executed):
  python -m wc26.data.download                     # fetch raw CSVs (network)
  python -m wc26.pipeline                          # everything below

Steps:
  1. load + normalize matches, validate all 48 WC-2026 teams resolve
  2. Elo (fitted params if processed/elo_params.json exists) + Pi + features
  3. ordered-logit ratings head; time-decay DC-Poisson goal head (confed prior)
  4. pool weights + temperature fit on the last 2 pre-tournament years
  5. optional market head if data/raw/outright_odds.csv exists
     (teams x bookmaker decimal odds; Shin de-vig -> consensus -> inverse sim)
  6. predict all 72 group-stage fixtures; export JSON artifacts + run snapshot

Artifacts (data/processed/):
  match_probs_2026.json   per group fixture: heads, pooled W/D/L, top scorelines
  ratings_2026.json       Elo / Pi / RD / BT-ability per team
  backtest_report.json    walk-forward WC 2014/18/22 + Euro 2024 (--backtest)
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from wc26.config import REPO_ROOT, load_config, snapshot_run
from wc26.data.download import manifest_hash
from wc26.data.normalize import load_matches, validate_wc2026_teams
from wc26.eval.backtest import WC_WINDOWS, run_backtests
from wc26.features import build_features
from wc26.market.consensus import consensus_probs
from wc26.market.inverse_sim import bt_match_probs, invert_outrights
from wc26.models.ordered_logit import OrderedLogit
from wc26.models.poisson import PoissonParams, fit_poisson
from wc26.models.pool import apply_temperature, fit_pool_weights, fit_temperature, log_pool
from wc26.ratings.elo import EloParams, run_elo
from wc26.ratings.pi import run_pi

KICKOFF = pd.Timestamp("2026-06-11")


def load_confederations(static: Path) -> dict[str, str]:
    with open(static / "confederations.json") as f:
        raw = json.load(f)
    return {t: c for c, ts in raw.items() if not c.startswith("_") for t in ts}


def load_elo_params(processed: Path, cfg: dict) -> EloParams:
    fitted = processed / "elo_params.json"
    if fitted.exists():
        d = json.loads(fitted.read_text())["params"]
        return EloParams(**d)
    e = cfg["elo"]
    return EloParams(
        initial_rating=e["initial_rating"],
        k_by_tier={k: float(v) for k, v in e["k_by_tier"].items()},
        home_advantage=e["home_advantage"],
        margin_exponent=e["margin_exponent"],
        rtm_weight=e["rtm_weight"],
    )


def predict_2026(matches: pd.DataFrame, cfg: dict, static: Path, processed: Path,
                 as_of: pd.Timestamp = KICKOFF) -> dict:
    """Pure function over a match table -> artifact dict (testable offline)."""
    with open(static / "groups_2026.json") as f:
        gdata = json.load(f)
    groups: dict[str, list[str]] = gdata["groups"]
    hosts = set(gdata["hosts"])
    confed_of = load_confederations(static)

    elo_params = load_elo_params(processed, cfg)
    elo = run_elo(matches, elo_params)
    pi = run_pi(matches)
    feats = build_features(matches, elo, pi)

    pre = (feats["date"] < as_of).to_numpy()
    val_start = as_of - pd.Timedelta(days=731)
    is_val = pre & (feats["date"] >= val_start).to_numpy()
    is_train = pre & ~is_val

    y = feats["outcome"].to_numpy()
    dr = feats["elo_diff"].to_numpy()
    home = feats["home"].to_numpy()

    ol = OrderedLogit().fit(dr[is_train], home[is_train], y[is_train])
    p_cfg = cfg["poisson"]
    pois = fit_poisson(
        matches, as_of=as_of,
        params=PoissonParams(half_life_days=p_cfg["half_life_days"], l2=p_cfg["l2"],
                             confed_shrink=p_cfg["confed_shrink"], max_goals=p_cfg["max_goals"]),
        confed_of=confed_of,
    )

    # pool weights/temperature from validation window (ratings + poisson heads)
    rows_val = feats[is_val]
    head_r_val = ol.predict(rows_val["elo_diff"].to_numpy(), rows_val["home"].to_numpy())
    head_p_val = head_r_val.copy()
    for i, (_, r) in enumerate(rows_val.iterrows()):
        if r["home_team"] in pois.index and r["away_team"] in pois.index:
            head_p_val[i] = pois.outcome_probs(r["home_team"], r["away_team"], neutral=r["home"] == 0.0)
    stack_val = np.stack([head_r_val, head_p_val])
    w = fit_pool_weights(stack_val, y[is_val])
    temp = fit_temperature(log_pool(stack_val, w), y[is_val])

    # optional market head
    bt = None
    odds_csv = REPO_ROOT / cfg["paths"]["raw"] / "outright_odds.csv"
    if odds_csv.exists():
        odds = pd.read_csv(odds_csv, index_col=0)
        target = consensus_probs(odds, method=cfg["market"]["devig_method"])
        bt = invert_outrights(target, groups, seed=int(cfg["seed"]))

    # ratings snapshot per team
    ratings_out = {}
    for g, teams in groups.items():
        for t in teams:
            ratings_out[t] = {
                "group": g,
                "elo": elo.ratings.get(t),
                "rd": elo.rd.get(t),
                "pi": pi.ratings.get(t),
                "bt_ability": (float(bt[t]) if bt is not None else None),
            }

    # 72 group fixtures (round robin within groups; venue/home: hosts non-neutral)
    fixtures = []
    for g, teams in groups.items():
        for a, b in itertools.combinations(teams, 2):
            neutral = a not in hosts and b not in hosts
            home_t, away_t = (a, b) if (a in hosts or neutral) else (b, a)
            dr_m = elo.ratings[home_t] - elo.ratings[away_t]
            heads = {
                "ratings": ol.predict(np.array([dr_m]), np.array([0.0 if neutral else 1.0]))[0],
                "poisson": pois.outcome_probs(home_t, away_t, neutral=neutral)
                if home_t in pois.index and away_t in pois.index else None,
            }
            if bt is not None:
                heads["market"] = bt_match_probs(float(bt[home_t]), float(bt[away_t]))
            used = [h for h in heads.values() if h is not None]
            if len(used) >= 2:
                # extend fitted 2-head weights with equal share for market head
                ww = np.array(list(w) + [w.mean()] * (len(used) - 2)) if len(used) > 2 else w
                ww = ww / ww.sum()
                pooled = apply_temperature(log_pool(np.stack(used)[:, None, :], ww), temp)[0]
            else:
                pooled = used[0]
            grid = (pois.score_grid(home_t, away_t, neutral=neutral)
                    if home_t in pois.index and away_t in pois.index else None)
            top_scores = []
            if grid is not None:
                flat = [(int(i), int(j), float(grid[i, j])) for i in range(grid.shape[0])
                        for j in range(grid.shape[1])]
                top_scores = sorted(flat, key=lambda r: -r[2])[:5]
            fixtures.append(
                {"group": g, "home": home_t, "away": away_t, "neutral": neutral,
                 "probs": {"win": float(pooled[0]), "draw": float(pooled[1]), "loss": float(pooled[2])},
                 "heads": {k: (list(map(float, v)) if v is not None else None) for k, v in heads.items()},
                 "top_scorelines": top_scores}
            )

    return {
        "as_of": str(as_of.date()),
        "pool": {"weights": list(map(float, w)), "temperature": float(temp)},
        "elo_params": asdict(elo_params),
        "ratings": ratings_out,
        "group_fixtures": fixtures,
    }


def simulate_2026(matches: pd.DataFrame, cfg: dict, static: Path,
                  n_sims: int = 100_000, seed: int | None = None) -> dict:
    """Stage-5 tournament simulation -> advancement/title artifact dict."""
    import itertools as _it

    from wc26.sim.bracket import load_bracket
    from wc26.sim.tournament import ROUNDS, build_score_cdfs, run_tournament

    with open(static / "groups_2026.json") as f:
        gdata = json.load(f)
    groups = gdata["groups"]
    teams = [t for g in "ABCDEFGHIJKL" for t in groups[g]]
    confed_of = load_confederations(static)
    p_cfg = cfg["poisson"]
    pois = fit_poisson(
        matches, as_of=KICKOFF,
        params=PoissonParams(half_life_days=p_cfg["half_life_days"], l2=p_cfg["l2"],
                             confed_shrink=p_cfg["confed_shrink"], max_goals=p_cfg["max_goals"]),
        confed_of=confed_of,
    )
    missing = [t for t in teams if t not in pois.index]
    if missing:
        raise ValueError(f"teams missing from goal model: {missing}")
    cdf, cdf_et = build_score_cdfs(pois, teams, hosts=set(gdata["hosts"]))
    group_index = np.array([[teams.index(t) for t in groups[g]] for g in "ABCDEFGHIJKL"])
    fixtures = [(teams.index(x), teams.index(y))
                for g in "ABCDEFGHIJKL" for x, y in _it.combinations(groups[g], 2)]
    bracket = load_bracket(static)
    elo = run_elo(matches, load_elo_params(REPO_ROOT / cfg["paths"]["processed"], cfg))
    elo_arr = np.array([elo.ratings.get(t, 1500.0) for t in teams])
    res = run_tournament(teams, group_index, fixtures, bracket, cdf, cdf_et,
                         n_sims=n_sims, seed=seed if seed is not None else int(cfg["seed"]),
                         elo=elo_arr, kappa=0.0)
    return {
        "n_sims": res.n_sims,
        "bracket_is_template": bracket.template,
        "allocation_fallbacks": res.allocation_fallbacks,
        "teams": {
            t: {
                "group": next(g for g in groups if t in groups[g]),
                **{r: float(res.reach[r][i]) for r in ROUNDS},
                "third_qualify": float(res.third_qualify[i]),
                "champion_ci90": list(res.champion_ci[t]),
            }
            for i, t in enumerate(teams)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true", help="also run WC/Euro walk-forward backtests")
    ap.add_argument("--simulate", action="store_true", help="run the 100k tournament simulation")
    ap.add_argument("--n-sims", type=int, default=100_000)
    args = ap.parse_args()

    cfg = load_config()
    static = REPO_ROOT / cfg["paths"]["static"]
    processed = REPO_ROOT / cfg["paths"]["processed"]
    processed.mkdir(parents=True, exist_ok=True)

    matches = load_matches()
    missing = validate_wc2026_teams(matches)
    if missing:
        raise SystemExit(f"identity resolution failed for: {missing} — extend team_aliases.json")

    artifact = predict_2026(matches, cfg, static, processed)
    (processed / "match_probs_2026.json").write_text(json.dumps(artifact, indent=2))
    (processed / "ratings_2026.json").write_text(json.dumps(artifact["ratings"], indent=2))
    print(f"wrote {processed / 'match_probs_2026.json'} ({len(artifact['group_fixtures'])} fixtures)")

    if args.simulate:
        sim = simulate_2026(matches, cfg, static, n_sims=args.n_sims)
        (processed / "advance_probs_2026.json").write_text(json.dumps(sim, indent=2))
        # site bundle: copy the artifacts the static site reads
        site_data = REPO_ROOT / "app" / "site" / "data"
        site_data.mkdir(parents=True, exist_ok=True)
        for name in ("match_probs_2026.json", "advance_probs_2026.json"):
            src = processed / name
            if src.exists():
                (site_data / name).write_text(src.read_text())
        print(f"wrote {processed / 'advance_probs_2026.json'} and site bundle")

    if args.backtest:
        confed_of = load_confederations(static)
        table = run_backtests(matches, WC_WINDOWS, confed_of=confed_of)
        table.to_json(processed / "backtest_report.json", orient="records", indent=2)
        print(table.to_string(index=False))

    snapshot_run(cfg, manifest_hash())


if __name__ == "__main__":
    main()
