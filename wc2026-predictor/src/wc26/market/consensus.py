"""Bookmaker consensus on the logit scale (Leitner, Hornik & Zeileis 2010).

Input: outright (tournament winner) decimal odds from J bookmakers for the
48 teams, as a DataFrame teams x bookmakers (NaN = not quoted).
Steps: de-vig per book -> logit -> average -> inverse-logit -> renormalize.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wc26.market.devig import devig

_EPS = 1e-9


def consensus_probs(outright_odds: pd.DataFrame, method: str = "shin") -> pd.Series:
    """Per-team consensus championship probability (sums to 1)."""
    logits = []
    for col in outright_odds.columns:
        o = outright_odds[col].dropna()
        if len(o) < 2:
            continue
        p = pd.Series(devig(o.to_numpy(), method), index=o.index)
        logits.append(np.log(p + _EPS) - np.log(1 - p + _EPS))
    if not logits:
        raise ValueError("no usable bookmaker columns")
    avg = pd.concat(logits, axis=1).mean(axis=1)
    p = 1.0 / (1.0 + np.exp(-avg))
    p = p.reindex(outright_odds.index).fillna(p.min() * 0.5)
    return p / p.sum()
