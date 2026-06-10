"""Stratified evaluation + 2026 head export for a trained Forty8Net.

The stratified report is the paper's core evidence (experiment E2): if the
model's edge over Elo concentrates in cross-confederation and long-gap
("first meeting in >= N years") strata while matching Elo within zones, the
anchoring hypothesis survives. Uniform improvement everywhere would mean
"generic representation gain", a different (weaker) claim.

  python -m wc26.neural.evaluate --ckpt runs/neural/<tag>/best_seed0.pt
  python -m wc26.neural.evaluate --ckpt ... --export-2026   # writes the
      ensemble head: data/processed/neural_head_2026.json (pipeline picks
      it up automatically as a 4th pool head)
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch

from wc26.config import REPO_ROOT, load_config
from wc26.data.normalize import load_matches
from wc26.eval.metrics import log_loss, rps
from wc26.models.ordered_logit import OrderedLogit
from wc26.neural.data import TIERS, build_pack, chronological_splits
from wc26.neural.model import Forty8Net, NetConfig
from wc26.pipeline import KICKOFF, load_confederations
from wc26.ratings.elo import run_elo


def _load(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Forty8Net(NetConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _last_meeting_gap_years(matches: pd.DataFrame) -> np.ndarray:
    last: dict[frozenset, pd.Timestamp] = {}
    gap = np.full(len(matches), np.inf)
    for r in range(len(matches)):
        key = frozenset((matches["home"].iat[r], matches["away"].iat[r]))
        if key in last:
            gap[r] = (matches["date"].iat[r] - last[key]).days / 365.25
        last[key] = matches["date"].iat[r]
    return gap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-end", default="2018-06-01")
    ap.add_argument("--val-end", default="2024-08-01")
    ap.add_argument("--export-2026", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    matches = load_matches()
    elo = run_elo(matches)
    model, ckpt = _load(args.ckpt)
    use_elo = bool(ckpt["use_elo"]); max_len = int(ckpt["max_len"])

    _, va, _ = chronological_splits(matches, args.train_end, args.val_end)
    pack = build_pack(matches, elo.history if use_elo else None, va,
                      max_len=max_len, use_elo=use_elo)
    with torch.no_grad():
        p = model.predict_proba(torch.from_numpy(pack.hist),
                                torch.from_numpy(pack.mask),
                                torch.from_numpy(pack.ctx)).numpy()

    # Elo baseline on identical rows
    dr = (elo.history["elo_home"] - elo.history["elo_away"]).to_numpy()
    home = (~matches["neutral"]).to_numpy(float)
    gd = matches["home_goals"].to_numpy() - matches["away_goals"].to_numpy()
    y_all = np.where(gd > 0, 0, np.where(gd == 0, 1, 2))
    tr_mask = matches["date"] < pd.Timestamp(args.train_end)
    ol = OrderedLogit().fit(dr[tr_mask.to_numpy()], home[tr_mask.to_numpy()],
                            y_all[tr_mask.to_numpy()])
    p_base = ol.predict(dr[va], home[va])
    y = pack.y

    confed_of = load_confederations(REPO_ROOT / cfg["paths"]["static"])
    rows = matches.iloc[va]
    ch = rows["home"].map(confed_of); ca = rows["away"].map(confed_of)
    cross = (ch.notna() & ca.notna() & (ch != ca)).to_numpy()
    within = (ch.notna() & ca.notna() & (ch == ca)).to_numpy()
    gap = _last_meeting_gap_years(matches)[va]
    strata = {
        "all": np.ones(len(va), bool),
        "within_confed": within,
        "cross_confed": cross,
        "tournament_grade": rows["tier"].isin(["wc_finals", "continental_finals"]).to_numpy(),
        "first_meeting_10y": gap >= 10,
    }
    report = {}
    for name, m in strata.items():
        if m.sum() < 30:
            continue
        report[name] = {
            "n": int(m.sum()),
            "model_ll": log_loss(p[m], y[m]), "elo_ll": log_loss(p_base[m], y[m]),
            "model_rps": rps(p[m], y[m]), "elo_rps": rps(p_base[m], y[m]),
            "delta_ll": log_loss(p_base[m], y[m]) - log_loss(p[m], y[m]),
        }
    print(json.dumps(report, indent=2))
    out = REPO_ROOT / cfg["paths"]["processed"] / "neural_stratified_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    if args.export_2026:
        with open(REPO_ROOT / cfg["paths"]["static"] / "groups_2026.json") as f:
            gdata = json.load(f)
        groups, hosts = gdata["groups"], set(gdata["hosts"])
        # synthetic future rows so build_pack encodes full history up to kickoff
        fixture_rows, names = [], []
        for g in "ABCDEFGHIJKL":
            for a, b in itertools.combinations(groups[g], 2):
                neutral = a not in hosts and b not in hosts
                h, w = (a, b) if (a in hosts or neutral) else (b, a)
                fixture_rows.append({"date": KICKOFF, "home": h, "away": w,
                                     "home_goals": 0, "away_goals": 0,
                                     "tournament": "FIFA World Cup",
                                     "neutral": neutral, "tier": "wc_finals"})
                names.append(f"{h}|{w}")
        aug = pd.concat([matches, pd.DataFrame(fixture_rows)], ignore_index=True)
        # elo history rows for the synthetic block: reuse final ratings (pre-match)
        elo_aug = run_elo(aug)
        idx = np.arange(len(matches), len(aug))
        pk = build_pack(aug, elo_aug.history if use_elo else None, idx,
                        max_len=max_len, use_elo=use_elo)
        with torch.no_grad():
            pf = model.predict_proba(torch.from_numpy(pk.hist),
                                     torch.from_numpy(pk.mask),
                                     torch.from_numpy(pk.ctx)).numpy()
        head = {nm: [float(x) for x in pr] for nm, pr in zip(names, pf)}
        path = REPO_ROOT / cfg["paths"]["processed"] / "neural_head_2026.json"
        path.write_text(json.dumps(head, indent=2))
        print(f"wrote {path} ({len(head)} fixtures) — pipeline will pool it in")


if __name__ == "__main__":
    main()
