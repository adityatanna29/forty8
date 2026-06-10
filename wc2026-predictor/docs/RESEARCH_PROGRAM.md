# Forty8 Research Program — what to train, what to claim, how to not fool yourself

## The research question (one sentence)

**Do shared-parameter sequence encoders implicitly solve the cross-
confederation anchoring problem in international football rating — and if
so, through which mechanism?**

## Why this is defensible novelty (and what is NOT claimed)

Genuinely new: (1) the question itself — JORS 2026 just established that
sparse-network comparability is a live problem but explored only Elo
variants; nobody has tested amortized/shared-encoder raters on it; (2) the
domain — every neural football paper is club football with player/event
data; international football (sparse, 150-year horizon) is untouched; (3)
benchmarking a neural rater against WHR/TTT smoothers — never done in
football, and WHR is the *honest* comparator because smoothing already
propagates cross-zone information backward in time.

Explicitly NOT claimed: beating bookmakers (no public model ever has,
consistently); transformer-over-history as an architecture novelty
(HIGFormer et al. are adjacent); multi-task scoreline heads as novelty.
If the headline RPS only matches Elo, the paper survives on the diagnosis;
if it only matches WHR, that is itself the finding ("amortization buys
nothing beyond smoothing"). Every outcome is reportable. That is the design.

## What was shipped (run/train these)

| Component | Command | Purpose |
|---|---|---|
| Shootout skill fit | `python -m wc26.models.shootout` | uses your previously-unused shootouts.csv; simulator auto-consumes the result |
| WHR-lite baseline | `wc26.baselines.whr.fit_whr` | the smoothing comparator (Davidson draws, Wiener drift, analytic gradients) |
| Forty8Net | `python -m wc26.neural.train` | shared-parameter transformer over match histories; multi-task W/D/L + Poisson heads |
| Stratified eval | `python -m wc26.neural.evaluate --ckpt ...` | within/cross-confederation, tournament, first-meeting-in-10y strata vs Elo |
| 2026 head export | `... --export-2026` | model becomes a 4th log-pool head automatically |
| Anchoring probe | `python -m wc26.neural.anchoring` | experiment E1 below — the headline figure |

## The experiment grid (publishability-critical; this is where your compute goes)

**E1 — Synthetic anchoring probe** (`wc26.neural.anchoring`). K
confederations with known latent gaps; cross-match rate p swept 0.5%→20%;
plot between-confederation gap-RMSE vs p for Elo / WHR / Forty8Net
(Elo-free tokens). The hypothesis lives or dies on the p→0 behavior.
≥3 seeds per cell; ~1 GPU-hour total.

**E2 — Stratified real-data evaluation** (`wc26.neural.evaluate`). The claim
succeeds iff the model's edge over Elo concentrates in cross-confederation /
first-meeting strata while matching Elo within zones. Uniform improvement =
generic representation gain = different, weaker claim. Report log-loss AND
RPS (the Wheatcroft caveat), plus reliability diagrams.

**E3 — Sequence-vs-tabular ablation (the Grinsztajn test).** Same
information, three encoders: Forty8Net sequences · the same token features
mean-pooled into LightGBM (`wc26.models.hybrid`) · GRU swap. If transformer ≈
pooled-GBM, the sequence structure adds nothing — report it honestly; the
anchoring diagnosis (E1/E2/E4) is the contribution either way. Also ablate
`--max-len` ∈ {10, 50, 64, 200}: if long history doesn't help, the
1872-present pitch is decoration.

**E4 — Shared-parameter ablation (the mechanism test).** Train one shared
encoder vs per-confederation encoders (split the training set by the home
team's confederation; same total capacity). If sharing is the anchoring
mechanism, the shared model wins specifically on cross-zone strata. This
isolates *why*, not just *whether* — the difference between a results paper
and a science paper.

**E5 — Pre-registered WC2026 evaluation.** Freeze the ensemble (with and
without the neural head), archive full probability tables + model hash to
OSF/arXiv BEFORE June 11 kickoff, evaluate post-tournament with per-match
log-loss/RPS and whole-tournament TRPS (Ekstrøm et al.) against: Elo-only,
WHR-only, de-vigged market odds, and the published Zeileis et al. WC2026
forecast (a live academic benchmark — a gift).

## Compute allocation (you have a lot; spend it where variance lives)

The default Forty8Net is ~120k parameters — deliberately. 50k matches is the
Grinsztajn danger zone; capacity is the enemy. Spend compute on: **seed
sweeps** (≥5 seeds everywhere; single-seed deltas at this scale are noise —
report mean ± sd), the **E1-E4 grid** (~40 configurations × 5 seeds ≈ a
weekend on one GPU), and **max_len 200+** (sequence building is the
bottleneck, not the GPU). Do NOT scale d_model past ~128 without first
showing the validation curve demands it.

## Additional public data still on the table (ranked)

1. Bookmaker outright odds snapshots (manual CSV; activates the market head
   — biggest non-research accuracy lever).
2. Transfermarkt squad values (dcaribou Kaggle dumps) → player layer + E5
   ensemble variant.
3. FIFA ranking history 1992+ (Kaggle cashncarry mirror) → covariate +
   "FIFA-rank-implied" baseline.
4. StatsBomb open event data (WC 2018/2022, EUROs) → xG-based token features
   for the most recent history window (post-2018 only; document the
   missingness honestly).
5. Club results for shared players (football-data.co.uk) → club-to-
   international transfer learning. Untouched in the literature; ambitious
   follow-up paper, not this one.

## Manuscript skeleton (target: MLSA/StatsBomb conf, JQAS, or MLJ soccer track)

1. Intro: anchoring problem (JORS 2026 frames it; we generalize beyond Elo).
2. Related work: §1/§6/§7 of docs/LITERATURE_SURVEY.md.
3. Methods: Forty8Net + WHR-lite + the calibrated log-pool production system.
4. Experiments: E1 (headline figure), E2, E3, E4.
5. Pre-registered WC2026 case study: E5.
6. Honest limitations: data scale, no player data pre-1990s, market gap.

The repo IS the portfolio: pre-registered forecasts, hash-chained provenance,
45+ tests, a production site — the combination of engineering rigor and a
crisp falsifiable question is what makes this PhD-application material, not
the headline number.
