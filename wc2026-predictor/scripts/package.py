"""Zip the repo for handoff (excludes raw data, runs, caches).

Run: python scripts/package.py [-o wc2026-predictor.zip]
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".git", "runs", "node_modules"}
EXCLUDE_PREFIXES = ("data/raw", "data/processed", "app/site/data")


def _excluded(rel: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in rel.parts):
        return True
    s = str(rel)
    return any(s == p or s.startswith(p + "/") for p in EXCLUDE_PREFIXES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="wc2026-predictor.zip")
    args = ap.parse_args()
    out = Path(args.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT)
            if _excluded(rel):
                continue
            if rel.suffix in {".pyc", ".zip"}:
                continue
            z.write(p, Path("wc2026-predictor") / rel)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
