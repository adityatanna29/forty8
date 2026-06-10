# Forty8 (wc2026-predictor)

Brand: **Forty8** — logo/favicon/OG share card in `app/site/assets/`
(SVG sources + rendered PNGs). Public front: branded index + methodology page
(`about.html`), disclaimers, no cookies/trackers. License: MIT.
Public-readiness audit: `docs/PUBLIC_READINESS_REVIEW.md`.

Complete FIFA World Cup 2026 prediction system: data pipeline, fitted-constant
Elo + Pi ratings, time-decayed Dixon-Coles goal model, market layer (Shin
de-vig, bookmaker consensus, inverse tournament simulation), player/squad
features, log-opinion-pool ensemble with temperature calibration, walk-forward
backtests, a vectorized 100k-run tournament simulator (third-place allocation,
extra time, shootouts, credible intervals), and a $0 Cloudflare hosting stack
with a live-score Worker and GitHub Actions automation.

Everything is CPU-only: the full 100k-simulation run takes ~2 seconds.

---

## How to run — in order

### 0. Install (once)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 1. Verify the codebase

```bash
python -m pytest tests/ -q          # 45 tests, ~4 s, no network needed
```

Tests run against a synthetic generator with known ground truth — models must
recover orderings/parameters, beat uniform out-of-sample, and satisfy every
tournament-conservation invariant.

### 2. Download data (network)

```bash
python -m wc26.data.download
```

Fetches martj42 international results (CC0) + openfootball 2026 schedule into
`data/raw/` with a SHA256 manifest. Re-run after every matchday — the source
updates and the manifest records exactly what you saw.

### 3. (Once, recommended) Fit Elo constants

```bash
python - <<'EOF'
from pathlib import Path
from wc26.data.normalize import load_matches
from wc26.ratings.fit import fit_elo_params
params, ll = fit_elo_params(load_matches(), out_path=Path("data/processed/elo_params.json"))
print(params, ll)
EOF
```

Takes minutes; replaces the folk constants (K=60/50/…, HA=100) with values
optimized on out-of-sample W/D/L log-loss. The pipeline picks the JSON up
automatically on subsequent runs.

### 4. (Optional) Drop in market + squad data

- `data/raw/outright_odds.csv` — rows = the 48 canonical team names, columns =
  bookmakers, cells = decimal outright odds. Activates the market head
  (Shin de-vig → logit consensus → inverse tournament simulation).
- Squad tables for the player layer (`wc26.players.features` docstring lists
  the columns; sources: Transfermarkt dumps / EA FC 26 Kaggle CSVs).

### 5. Run the pipeline

```bash
python -m wc26.pipeline --simulate --backtest
```

Produces in `data/processed/`:

| Artifact | Contents |
|---|---|
| `match_probs_2026.json` | all 72 group fixtures: per-head + pooled calibrated W/D/L, top scorelines |
| `advance_probs_2026.json` | per team: P(R32/R16/QF/SF/final/champion) + 90% CI, third-qualify prob |
| `ratings_2026.json` | Elo, RD, Pi, BT-ability per team |
| `backtest_report.json` | WC 2014/18/22 + Euro 2024 walk-forward vs Elo-only baseline |
| `runs/<timestamp>/run.json` | config hash + data-manifest hash (provenance) |

It also copies the site bundle into `app/site/data/`.

### 6. CRITICAL before publishing: verify the bracket

`data/static/bracket_2026.json` ships as a **structurally-valid TEMPLATE**
(8 winner-vs-third, 4 winner-vs-runner, 4 runner-vs-runner; all invariants
test-enforced; all 495 third-place subsets allocate cleanly). Replace the R32
pairings and third pools with the official FIFA schedule + Annex C tables
(FIFA regulations / Wikipedia "2026 FIFA World Cup knockout stage"), then
re-run `pytest tests/test_stage5_sim.py`. The site shows a warning banner
while `_template` is true.

### 7. Preview the site locally

```bash
cd app/site && python -m http.server 8000   # open http://localhost:8000
```

Static HTML over the JSON artifacts — no build step, no framework.

### 8. Deploy ($0 hosting — Cloudflare)

Why Cloudflare: the only free host with **unlimited static bandwidth** — a
viral matchday (~150 GB) exceeds the monthly caps of Vercel/Netlify/GitHub
Pages in one day, and free always-on alternatives either sleep (Render) or no
longer exist (Fly/Railway). The site is pure static + JSON, so there is no
backend to scale; the CDN absorbs any spike.

```bash
# one-time
npx wrangler login
npx wrangler pages project create wc26
npx wrangler pages deploy app/site --project-name wc26
```

Automation: add repo secrets `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`;
`.github/workflows/update.yml` then refreshes data, re-runs the pipeline +
simulation, commits timestamped forecasts to `history/<date>/`
(append-only pre-registration), and redeploys — nightly at 05:17 UTC and
on-demand via *Run workflow*.

### 9. (Optional) Live scores Worker

```bash
cd app/worker
npx wrangler kv namespace create LIVE     # paste id into wrangler.toml
npx wrangler secret put FOOTBALL_DATA_TOKEN   # licensed fallback (free key)
npx wrangler deploy
```

One central poller (cron, every 2 min) fetches the scoreboard, stores
normalized `live.json` in KV; route `your-domain/data/live.json` to the
Worker so the site's same-origin fetch hits it. A million visitors poll your
CDN; upstream sees ~1 request/min. ESPN's keyless feed is undocumented —
that's why the licensed football-data.org fallback is wired in behind a flag.

---

## Repository layout

```
configs/default.yaml          all hyperparameters; runs snapshot their resolved copy
data/static/                  groups (Dec-2025 draw), aliases, confederations, bracket
src/wc26/
  config.py                   config + seeding + run provenance
  data/{download,normalize}   pinned downloads w/ manifest; identity resolution
  ratings/{elo,pi,fit}        Elo (param., RTM, RD), Pi ratings, constant fitting
  models/{poisson,ordered_logit,hybrid,pool}
                              DC-Poisson (analytic grads, confed shrink), W/D/L
                              head, regularized LightGBM, log-pool + temperature
  market/{devig,consensus,inverse_sim}
                              Shin/power/proportional, LHZ consensus, BT inversion
  players/features.py         squad features incl. availability multiplier
  features.py                 leakage-free per-match assembly
  eval/{metrics,backtest}     log-loss/RPS/Brier/reliability; walk-forward harness
  sim/{bracket,group_stage,knockout,tournament}
                              bracket DAG + validation, FIFA ranking, ET/shootout,
                              chunked vectorized Monte Carlo
  pipeline.py                 end-to-end orchestrator (--simulate --backtest)
app/site/                     static site (index, about, assets/, _headers, data/)
app/worker/                   live-poller Worker + wrangler.toml
.github/workflows/update.yml  nightly refresh + deploy + forecast pre-registration
scripts/package.py            handoff zip
tests/                        45 tests incl. ground-truth integration + e2e
```

## Method-to-code map

| Method | Source | Code |
|---|---|---|
| Elo expectancy/update, tiered K, margin G(d) | eloratings.net rules | `ratings/elo.py` |
| Constants fit by predictive likelihood | Hvattum & Arntzen 2010 | `ratings/fit.py` |
| Pi ratings (goal units, log-dampened) | Constantinou & Fenton 2013 | `ratings/pi.py` |
| Weighted-likelihood Poisson + time decay | Ley et al. 2019 | `models/poisson.py` |
| Low-score correction tau (Eq. 4.3) | Dixon & Coles 1997 | `models/poisson.py` |
| Ordered logit W/D/L on Elo diff | Robberechts & Davis 2018 | `models/ordered_logit.py` |
| Shin de-vig | Shin 1993 | `market/devig.py` |
| Consensus + inverse tournament simulation | Leitner, Hornik & Zeileis 2010 | `market/{consensus,inverse_sim}.py` |
| Log-opinion pool / temperature scaling | Ranjan & Gneiting 2010 / Guo et al. 2017 | `models/pool.py` |
| Hybrid GBM on covariates | Groll et al. 2019 | `models/hybrid.py` |
| 48-team format, thirds, tiebreakers | FIFA 2026 regulations | `sim/*` |

## Honest limitations

- **Bracket is a template** until you complete step 6 — the single remaining
  manual verification, by design (transcribing Annex C from memory would risk
  silently wrong advancement odds).
- Shootouts default to 50/50 (`kappa=0`); fit kappa on the ~49 historical WC
  shootouts before trusting a skill effect.
- The hybrid head is fit/test-ready but enters the 2026 path only once squad
  CSVs are supplied; the shipped pool is ratings + Poisson (+ market if odds
  provided).
- Tail criteria of FIFA group tiebreakers (head-to-head subtable, fair play)
  are approximated by seeded randomness after points/GD/GF.
- Expectations: ~55% 3-way accuracy, RPS ≈ 0.19–0.20, parity with the market
  is the realistic ceiling — distrust anything claiming more.

## Packaging

```bash
python scripts/package.py -o wc2026-predictor.zip
```
