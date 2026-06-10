import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def toy_matches() -> pd.DataFrame:
    """Tiny deterministic match table: A strong, B mid, C weak."""
    rows = []
    import itertools
    dates = pd.date_range("2020-01-01", periods=60, freq="7D")
    fixtures = list(itertools.permutations(["A", "B", "C"], 2))
    score = {("A", "B"): (2, 0), ("B", "A"): (0, 2), ("A", "C"): (3, 0),
             ("C", "A"): (0, 3), ("B", "C"): (1, 0), ("C", "B"): (0, 1)}
    for i, d in enumerate(dates):
        h, a = fixtures[i % len(fixtures)]
        hg, ag = score[(h, a)]
        rows.append({"date": d, "home": h, "away": a, "home_goals": hg,
                     "away_goals": ag, "tournament": "Friendly",
                     "neutral": True, "tier": "friendly"})
    return pd.DataFrame(rows)
