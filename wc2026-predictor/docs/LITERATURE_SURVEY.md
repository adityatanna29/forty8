# Forty8 — Literature Survey (consolidated, June 2026)

Consolidates six research passes. Citations verified against live sources at
survey time; the two marked ⚠ could not be fully verified and must be checked
before any manuscript cites them.

## 1. Rating systems

**Elo family.** eloratings.net rules (importance-tiered K, goal-margin
multiplier, +100 home advantage) are the folk standard; Hvattum & Arntzen
(IJF 2010) established the right protocol — Elo covariates into an ordered
logit, constants chosen by out-of-sample likelihood — and that odds beat Elo
on log-loss. Tuned constants land far from folk values (opisthokonta: club
K≈18.5, HA≈68). Margin-of-victory variants systematically beat plain Elo
(G-Elo, arXiv:2010.11187). FiveThirtyEight's SPI added between-season
regression to a team's trailing mean and offense/defense splits.

**Pi-ratings.** Constantinou & Fenton (JQAS 2013): goal-unit ratings with
log-dampened errors; beat Elo on EPL and was profitable vs 2007-12 market
odds; penaltyblog's 2025 replication confirms the RPS edge over Elo.

**Uncertainty-carrying raters.** Glicko-2 with football modifications
(arXiv:2310.11459); TrueSkill adds little for 1v1 team sports. The modern
unifying statement is the state-space view of skill rating
(arXiv:2308.02414): Elo/Glicko/TrueSkill are filtering approximations of one
latent Gaussian model; smoothing variants exist and matter (see §6).

**Whole-history smoothing.** TrueSkill Through Time (Dangauthier et al.,
NIPS 2007) and Whole-History Rating (Coulom 2008) do full smoothing — late
information revises early ratings. Coulom reports WHR > Elo/Glicko/TrueSkill
on prediction. Football applications: essentially none. Implemented here as
`wc26/baselines/whr.py` (Davidson draws, Wiener drift, analytic-gradient MAP).

## 2. Goal models

Maher (1982) → Dixon & Coles (1997; low-score tau correction Eq. 4.3 +
exponential time decay) → bivariate Poisson (Karlis & Ntzoufras 2003; in
practice ≈ DC) → dynamic state-space strengths (Rue & Salvesen 2000; Koopman
& Lit 2015). For NATIONAL teams the decisive comparison is Ley, Van de Wiele
& Van Eetvelde (Statistical Modelling 2019; arXiv:1705.09575): ten models
under weighted likelihood with half-life decay — (bivariate) Poisson with
decay wins on RPS. That model is our goal head.

## 3. World Cup forecasting systems

Academic line: Groll/Ley/Schauberger/Van Eetvelde hybrid random forests —
WC2018 (arXiv:1806.03208), WWC2019 (arXiv:1906.01131), EURO2020
(arXiv:2106.05799), EURO2024 (arXiv:2410.09068; key ablation: top-8 of 21
features beat the full set, with bookmaker-consensus ability and market value
the top importances), and a live WC2026 forecast (Zeileis et al., June 2026:
Spain 14.5%). Bookmaker-consensus methodology: Leitner, Hornik & Zeileis
(IJF 2010) — de-vig, logit-average, invert through tournament simulation.
Ordered logit on Elo diff beat bivariate Poisson and bookmakers at WC level
(Robberechts & Davis, MLSA 2018). Commercial: Opta supercomputer (odds +
power rankings), Nate Silver's PELE (Transfermarkt-dated rosters), Goldman
Sachs (Elo + ad-hoc "mentality"). Benchmarks: ~53-58% three-way accuracy,
RPS ≈ 0.19-0.20, market parity is the realistic ceiling (arXiv:2012.04380).

## 4. Markets, calibration, pooling

De-vigging: Shin (1993) ≈ power > proportional (Clarke et al. 2017); Python
reference mberk/shin. Odds-history blending: Egidi, Pauli & Torelli
(arXiv:1802.08848); Skellam-inversion follow-up (arXiv:2605.16066 ⚠ 2026
preprint, verify). Calibration: temperature scaling (Guo et al. 2017),
beta/Dirichlet calibration (Kull et al., NeurIPS 2019); model selection on
calibration flips betting ROI sign (Walsh & Joshi, arXiv:2303.06021).
Pooling: linear pools of calibrated forecasts are provably miscalibrated
(Ranjan & Gneiting, JRSS-B 2010); log-opinion pools are externally Bayesian.
No football paper uses a log pool — a modest methodological niche we occupy.
Metrics: RPS standard but contested (Wheatcroft, arXiv:1908.08980 — report
log-loss too); whole-tournament forecasts need TRPS (Ekstrøm, Van Eetvelde,
Ley & Brefeld, arXiv:1912.07364).

## 5. Player-level evidence

Plus-minus ratings: Sæbø & Hvattum → Pantuso & Hvattum (2021, age curve +
teammate shrinkage); squad averages transfer to national teams (used in every
recent Groll-line forecast). Lineup-average player ratings + Elo
significantly beat Elo alone (Arntzen & Hvattum, Statistical Modelling 2021).
Squad Transfermarkt value out-predicts FIFA rank and Elo (Peeters,
wisdom-of-crowds). EA-game positional ratings ≈ 55-56% accuracy standalone.
Direct player-vs-player dyad features: zero supporting literature at
international level; 0-3 observations per dyad — excluded from modeling by
design. VAEP/xT (Decroos et al.; socceraction) computable from StatsBomb open
WC 2018/2022 events but stale for 2026 — UI content only.

## 6. Neural & learned approaches (the research frontier)

Sequence/graph models exist for CLUB football with rich data: HIGFormer
(arXiv:2507.10626; reported KDD'25 ⚠ venue unconfirmed) — player-team
heterogeneous graph transformer; axial transformer for in-game forecasting
(arXiv:2511.18730; 62k matches); LSTM literature consistently ~52% and below
bookmakers (Danisik 2018); deep entries lost to odds consensus in the 2023
Soccer Prediction Challenge (arXiv:2309.14807, Machine Learning 2024);
GElo random-walk graph embeddings + Elo (arXiv:2304.08257, esports); GNNs for
sports outcomes (Xenopoulos & Silva, arXiv:2207.14124, ~52% in soccer).
**No published work trains a shared-parameter sequence encoder over
national-team histories**, and no neural football paper benchmarks against
WHR/TTT smoothers.

The skeptical prior: tree ensembles beat deep nets on ~10k-sample tabular
data (Grinsztajn et al., NeurIPS 2022); both public soccer prediction
challenges (2017: 216k matches, won by XGBoost-over-pi-ratings, Hubáček et
al., Machine Learning 2019; 2023: odds consensus on top) confirm it in this
exact domain. Any neural claim must survive the sequence-vs-tabular ablation.

## 7. The sparse cross-confederation problem (our anchor claim)

"The issue of sparse networks in sports competitions: can Elo ratings
efficiently compare football teams that never play a match?" (Journal of the
Operational Research Society, online Jan 2026) — tests 8 Elo variants on
fragmented networks; odds-input and league-consistent Elo help; **stays
entirely inside the Elo family**. Practitioner evidence of regional bias in
eloratings.net exists (football-rankings.info; non-academic). Only ~26% of
competitive internationals are cross-confederation outside World Cups.
Cross-league strength transfer: plus-minus with league-strength terms
(arXiv:1706.04943); club→international transfer learning: nothing published.

This is the gap Forty8's research track targets: the problem is freshly
established (JORS 2026), the solution space outside Elo is untouched, and the
honest baselines (WHR/TTT) have never been run on football.

## 8. Practice standards worth copying

arXiv-timestamped pre-tournament forecasts (Groll-line, every tournament
since 2014); OSF archiving of challenge protocols (osf.io/ftuva);
hash-chained append-only forecast logs (this repo's forecast history) exceed
both.
