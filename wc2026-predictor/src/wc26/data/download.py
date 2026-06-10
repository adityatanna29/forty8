"""Pinned-URL downloader with SHA256 manifest (framework §A.4, tier T1).

All sources are free and license-clean (martj42: CC0, openfootball: public
domain). Each download records its SHA256 into data/raw/manifest.json so every
ForecastSnapshot can be traced back to exact input bytes.

Run: ``python -m wc26.data.download``
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import requests

from wc26.config import REPO_ROOT, load_config

MARTJ42 = "https://raw.githubusercontent.com/martj42/international_results/master"

SOURCES: dict[str, str] = {
    "results.csv": f"{MARTJ42}/results.csv",
    "shootouts.csv": f"{MARTJ42}/shootouts.csv",
    "goalscorers.csv": f"{MARTJ42}/goalscorers.csv",
    "former_names.csv": f"{MARTJ42}/former_names.csv",
    "worldcup_2026.json": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json",
}


@dataclass(frozen=True)
class DownloadResult:
    name: str
    path: Path
    sha256: str
    bytes: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all(raw_dir: Path | None = None, timeout: int = 60) -> list[DownloadResult]:
    """Fetch every pinned source; write manifest.json; return results.

    Existing files are re-downloaded (sources update after each matchday);
    the manifest captures what this run actually saw.
    """
    cfg = load_config()
    raw = raw_dir or (REPO_ROOT / cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []
    for name, url in SOURCES.items():
        dest = raw / name
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        results.append(DownloadResult(name, dest, _sha256(dest), len(resp.content)))
    manifest = {r.name: {"sha256": r.sha256, "bytes": r.bytes, "url": SOURCES[r.name]} for r in results}
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return results


def manifest_hash(raw_dir: Path | None = None) -> str | None:
    """Single hash over the whole manifest (provenance key for runs)."""
    cfg = load_config()
    p = (raw_dir or (REPO_ROOT / cfg["paths"]["raw"])) / "manifest.json"
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


if __name__ == "__main__":
    for r in download_all():
        print(f"{r.name:24s} {r.bytes:>10,d} B  {r.sha256[:12]}")
