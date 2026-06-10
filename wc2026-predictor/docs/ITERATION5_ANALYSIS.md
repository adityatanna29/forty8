# Iteration 5 — Analysis of the first real run, fixes shipped, and what's next

## 1. What your results actually say

### Backtest (the honest scoreboard)

| Tournament | Model LL | Elo LL | Model RPS | Elo RPS | Verdict |
|---|---|---|---|---|---|
| WC 2014 | 0.949 | **0.903** | 0.199 | **0.182** | Elo wins clearly |
| WC 2018 | **0.947** | 0.978 | **0.197** | 0.208 | Model wins |
| WC 2022 | 1.084 | **1.061** | 0.221 | 0.227 | ~tie, both weak |
| Euro 2024 | **0.999** | 1.030 | **0.190** | 0.196 | Model wins |
| **Mean** | 0.995 | 0.993 | 0.202 | 0.203 | **Dead heat** |

Context: uniform = 1.099; published market-grade numbers sit ~0.95–1.05 LL and
~0.19–0.20 RPS depending on the cup (2022 was chaos for everyone — even
bookmakers were ~1.04). So the machinery works and nothing is leaking, but
**the ensemble adds no edge over plain Elo yet.** That is the expected result
for a results-only model: the two signals the literature ranks highest —
bookmaker consensus and squad market values — were not in your data path
(no `outright_odds.csv`, no squad CSVs), so the pool was blending two heads
fed by the *same* match-history information. Blending correlated heads ≈ the
better head, minus noise.

WC 2014 is the diagnostic gem: Elo-only crushed it (0.903) because 2014 was
chalky, and the pool *diluted* a great head with a worse-that-year Poisson
head using weights learned on qualifiers/friendlies. That's a
distribution-shift problem, now fixed (see §2).

### Title odds (the bias fingerprint)

| Team | Your model | Market/peer models | Gap |
|---|---|---|---|
| Argentina | **15.8% (#1)** | ~8–10% | +6–8 pts inflated |
| Colombia | 4.1% | ~1.5–2% | inflated |
| Ecuador | 3.6% | ~1% | inflated |
| Spain | 11.5% | 14.5–16.7% (#1) | deflated |
| France | **5.7%** | 12–13% | **−7 pts deflated** |
| England | 6.7% | 11–12% | deflated |
| USA (host) | 0.2%, R32 66% | ~1–2%, R32 ~80% | deflated |

One pattern explains all of it: **cross-confederation anchoring failure.**
CONMEBOL's internal qualifiers are competitive and frequent → intra-zone wins
pump Elo/abilities; UEFA's non-tournament games (friendlies, rotated Nations
League squads) deflate theirs. Results-only models can't see the scale
between zones because inter-zone matches are sparse and time-decayed.
Secondary finding: Germany reaching the final at 12.9% vs France 10.1%
despite France's higher rating = **template-bracket asymmetry**, not signal.
And the ±0.15% CIs are Monte-Carlo noise only — not real uncertainty.

## 2. Fixes shipped in this iteration (in the new zip)

1. **`ratings/confed.py` — cross-confederation calibration.** With the goal
   model frozen, per-confederation strength shifts (UEFA = 0 reference) are
   fit by time-decayed Poisson MLE on *inter-confederation matches only*,
   then applied to attack+defence and (in Elo points) to the ratings head.
   Test-proven to recover an injected cross-zone bias the base model cannot
   identify. Expect CONMEBOL down, UEFA relatively up on your next run.
2. **Tier-weighted training.** The ordered logit now weights matches by
   importance (WC 4×, continental 3×, qualifier 2×, friendly 1×) in both the
   backtest and the 2026 path.
3. **Tournament-conditions validation.** Pool weights + temperature are now
   fit on competitive (non-friendly) validation matches only — the WC 2014
   dilution failure mode.
4. **Honest credible intervals.** The simulator now redraws per-team
   strength noise per chunk (RD-derived, goal-log scale) and rebuilds the
   scoreline CDFs, so `champion_ci90` reflects parameter uncertainty, not
   just simulation error. Expect intervals ~5–10× wider — that's correct.
5. **`scripts/tune.py`** — half-life × confed-calibration grid over the
   4-tournament backtest; prints the best setting to copy into the config.

## 3. What's next, ranked by expected impact (the precision roadmap)

1. **Bookmaker outright odds → market head (biggest single fix).** One CSV
   (`data/raw/outright_odds.csv`: 48 team rows × bookmaker columns, decimal
   odds from oddschecker/Kalshi/Polymarket) activates Shin de-vig →
   consensus → inverse simulation. This is the cheapest correction for every
   bias in §1 — markets already price squad strength, injuries, and zone
   gaps. The Zeileis-line research puts this as the #1-importance signal.
2. **Squad market values → player layer + hybrid head.** Transfermarkt
   totals per squad (dcaribou Kaggle dump). #2-importance signal; also
   powers the availability multiplier for injury re-pricing.
3. **Official bracket (README step 6).** The template is now distorting
   semifinal-path odds measurably. 30 minutes of transcription + tests.
4. **Run `fit_elo_params` + `scripts/tune.py`** if you haven't — folk
   constants and a guessed half-life are leaving a few LL points on the
   table.
5. (Then) host-advantage refinement for the 3 co-hosts, ET goal-rate fit on
   historical ET periods, shootout kappa on the 49 historical WC shootouts.

Realistic ceiling check, so expectations stay honest: with 1+2 done, target
mean LL ≈ 0.96–0.98 and beating Elo on 3–4 of 4 tournaments. Nobody
consistently beats the market; matching it with full transparency is the win.

## 4. Viewing the UI from Google Colab

The site is static — serve `app/site/` and tunnel the port. Two ways:

**A. Colab-native (simplest, notebook cell):**
```python
# cell 1 — serve the site
import subprocess
subprocess.Popen(["python3", "-m", "http.server", "8000"],
                 cwd="/content/forty8/wc2026-predictor/app/site")

# cell 2 — open a proxied browser tab (Pro+ supports this fine)
from google.colab import output
output.serve_kernel_port_as_window(8000)   # or serve_kernel_port_as_iframe(8000, height=900)
```
The link it prints is authenticated to your Google session.

**B. Public URL from the terminal (shareable, no account needed):**
```bash
cd /content/forty8/wc2026-predictor/app/site && python3 -m http.server 8000 &
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
cloudflared tunnel --url http://localhost:8000
```
It prints a `https://<random>.trycloudflare.com` URL — open it anywhere.
(Quick tunnels die when the Colab VM recycles; for anything persistent,
deploy to Cloudflare Pages per README step 8 — that's the real hosting.)

Note: `simulate_2026` now rebuilds CDFs per chunk for the uncertainty draws,
so the 100k-run sim takes ~10–20 s instead of ~2 s. Still CPU-only.
