"""Forty8Net — shared-parameter sequence encoder for dynamic team ratings.

Architecture (deliberately small: ~50k international matches is the
Grinsztajn danger zone, so capacity is the enemy):

  token (14) --Linear--> d --TransformerEncoder(L layers, H heads)-->
  masked mean-pool + last-token  -> team state h in R^d
  match head: MLP([h_home, h_away, h_home - h_away, ctx]) ->
      3 W/D/L logits  +  softplus rates (lam_home, lam_away)

Multi-task loss = CE(W/D/L, label-smoothed) + alpha * PoissonNLL(goals).
The SAME encoder weights serve all ~250 national teams — the shared-
parameter property whose anchoring behavior is the research question
(docs/RESEARCH_PROGRAM.md, experiment E4 splits it per confederation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from wc26.neural.data import CTX_DIM, TOKEN_DIM


@dataclass
class NetConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 2
    dropout: float = 0.15
    head_hidden: int = 96
    label_smoothing: float = 0.05
    poisson_alpha: float = 0.35


class HistoryEncoder(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.proj = nn.Linear(TOKEN_DIM, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, hist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hist: (B, T, TOKEN_DIM), mask: (B, T) True = real
        x = self.proj(hist)                                   # (B, T, d)
        z = self.encoder(x, src_key_padding_mask=~mask)       # (B, T, d)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (z * mask.unsqueeze(-1)).sum(dim=1) / denom    # (B, d)
        # last real token (recency emphasis)
        idx = (mask.float().cumsum(dim=1).argmax(dim=1))      # (B,)
        last = z[torch.arange(z.size(0)), idx]
        return self.norm(mean + last)


class Forty8Net(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        self.encoder = HistoryEncoder(self.cfg)
        d = self.cfg.d_model
        self.head = nn.Sequential(
            nn.Linear(3 * d + CTX_DIM, self.cfg.head_hidden),
            nn.GELU(), nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden, 3 + 2),
        )

    def forward(self, hist: torch.Tensor, mask: torch.Tensor,
                ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hist: (B, 2, T, F) -> encode both sides with the SAME weights
        b = hist.size(0)
        flat = hist.view(b * 2, hist.size(2), hist.size(3))
        fmask = mask.view(b * 2, mask.size(2))
        h = self.encoder(flat, fmask).view(b, 2, -1)          # (B, 2, d)
        hh, ha = h[:, 0], h[:, 1]
        out = self.head(torch.cat([hh, ha, hh - ha, ctx], dim=-1))
        logits = out[:, :3]                                    # W/D/L
        rates = F.softplus(out[:, 3:]) + 1e-4                  # (lam_h, lam_a)
        return logits, rates

    def loss(self, logits: torch.Tensor, rates: torch.Tensor,
             y: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, y, label_smoothing=self.cfg.label_smoothing)
        pois = (rates - goals * torch.log(rates)).mean()       # Poisson NLL (no const)
        return ce + self.cfg.poisson_alpha * pois

    @torch.no_grad()
    def predict_proba(self, hist: torch.Tensor, mask: torch.Tensor,
                      ctx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(hist, mask, ctx)
        return F.softmax(logits, dim=-1)
