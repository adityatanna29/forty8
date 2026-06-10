"""Training loop for Forty8Net. Walk-forward, seeded, checkpointed.

Run (after data download; GPU strongly recommended for seed sweeps):

  python -m wc26.neural.train                       # defaults from config
  python -m wc26.neural.train --no-elo              # Elo-free ablation (E3/E4)
  python -m wc26.neural.train --max-len 200 --seeds 5

Outputs to runs/neural/<tag>/: best.pt, config.json, metrics.json
(per-epoch + final val metrics vs the Elo-ordered-logit baseline on the
SAME validation rows — the only comparison that means anything).

Compute guidance: the default model is ~120k params and trains in ~10 min
on one modern GPU. With "a lot of compute", spend it on (1) seed sweeps
(>=5 seeds, report mean +/- sd — single-seed deltas at this scale are noise),
(2) the E1-E4 ablation grid in docs/RESEARCH_PROGRAM.md, (3) max_len 200+,
NOT on a bigger model — capacity hurts in the 50k-sample regime.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from wc26.config import REPO_ROOT, load_config
from wc26.data.normalize import load_matches
from wc26.eval.metrics import log_loss, rps
from wc26.models.ordered_logit import OrderedLogit
from wc26.neural.data import build_pack, chronological_splits
from wc26.neural.model import Forty8Net, NetConfig
from wc26.ratings.elo import run_elo


def _loader(pack, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(pack.hist), torch.from_numpy(pack.mask),
        torch.from_numpy(pack.ctx), torch.from_numpy(pack.y),
        torch.from_numpy(pack.goals),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


@torch.no_grad()
def _eval(model: Forty8Net, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for hist, mask, ctx, y, _ in loader:
        p = model.predict_proba(hist.to(device), mask.to(device), ctx.to(device))
        probs.append(p.cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def train_one(seed: int, args, matches, elo, tr, va, out_dir: Path) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_elo = not args.no_elo
    pack_tr = build_pack(matches, elo.history if use_elo else None, tr,
                         max_len=args.max_len, use_elo=use_elo)
    pack_va = build_pack(matches, elo.history if use_elo else None, va,
                         max_len=args.max_len, use_elo=use_elo)
    dl_tr = _loader(pack_tr, args.batch_size, True)
    dl_va = _loader(pack_va, args.batch_size, False)

    model = Forty8Net(NetConfig(d_model=args.d_model, n_layers=args.n_layers)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_ll, best_state, patience = np.inf, None, 0
    history = []
    for epoch in range(args.epochs):
        model.train(); t0 = time.time()
        for hist, mask, ctx, y, goals in dl_tr:
            opt.zero_grad()
            logits, rates = model(hist.to(device), mask.to(device), ctx.to(device))
            loss = model.loss(logits, rates, y.to(device), goals.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        p_va, y_va = _eval(model, dl_va, device)
        ll = log_loss(p_va, y_va)
        history.append({"epoch": epoch, "val_ll": ll, "val_rps": rps(p_va, y_va),
                        "secs": round(time.time() - t0, 1)})
        if ll < best_ll - 1e-4:
            best_ll, best_state, patience = ll, model.state_dict(), 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": asdict(model.cfg),
                "max_len": args.max_len, "use_elo": use_elo, "seed": seed},
               out_dir / f"best_seed{seed}.pt")
    p_va, y_va = _eval(model, dl_va, device)

    # Elo ordered-logit baseline on the SAME validation rows
    dr = (elo.history["elo_home"] - elo.history["elo_away"]).to_numpy()
    home = (~matches["neutral"]).to_numpy(float)
    gd = matches["home_goals"].to_numpy() - matches["away_goals"].to_numpy()
    y_all = np.where(gd > 0, 0, np.where(gd == 0, 1, 2))
    ol = OrderedLogit().fit(dr[tr], home[tr], y_all[tr])
    p_base = ol.predict(dr[va], home[va])

    return {"seed": seed, "epochs_run": len(history), "history": history,
            "val_ll": float(log_loss(p_va, y_va)), "val_rps": float(rps(p_va, y_va)),
            "elo_baseline_ll": float(log_loss(p_base, y_all[va])),
            "elo_baseline_rps": float(rps(p_base, y_all[va]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2018-06-01")
    ap.add_argument("--val-end", default="2024-08-01")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-elo", action="store_true",
                    help="Elo-free ablation: encoder sees raw history only")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    load_config()  # seeds global RNGs + validates config exists
    matches = load_matches()
    elo = run_elo(matches)
    tr, va, te = chronological_splits(matches, args.train_end, args.val_end)
    print(f"train {len(tr)}  val {len(va)}  test(held-out, untouched) {len(te)}")

    tag = args.tag or time.strftime("%Y%m%dT%H%M%S") + ("_noelo" if args.no_elo else "")
    out_dir = REPO_ROOT / "runs" / "neural" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    results = [train_one(s, args, matches, elo, tr, va, out_dir)
               for s in range(args.seeds)]
    lls = [r["val_ll"] for r in results]
    summary = {
        "val_ll_mean": float(np.mean(lls)), "val_ll_sd": float(np.std(lls)),
        "val_rps_mean": float(np.mean([r["val_rps"] for r in results])),
        "elo_baseline_ll": results[0]["elo_baseline_ll"],
        "elo_baseline_rps": results[0]["elo_baseline_rps"],
        "runs": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    print(f"checkpoints + metrics -> {out_dir}")


if __name__ == "__main__":
    main()
