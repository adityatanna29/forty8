# Forty8 — Public-Readiness Review (final pass)

A deliberate re-examination of the whole project from angles the engineering
iterations did not center: legal exposure, end-user experience, operations
under failure, and what a public audience will actually do with the numbers.
Each item: status ✅ handled in this run · ⚠ mitigated, residual risk · ❌ open,
needs a decision or future work.

---

## 1. Legal & intellectual property

| Item | Status | Notes |
|---|---|---|
| FIFA trademarks | ✅ | Brand is "Forty8" (no FIFA marks in name/logo); footer carries a non-affiliation disclaimer; tournament name used nominatively only. Do not use the official emblem, mascot, or trophy imagery — ever. |
| Gambling exposure | ✅/⚠ | Prominent "not betting advice" in footer, about page, and OG card. Residual: some jurisdictions regulate *publishing* odds-like content with affiliate links — we have no ads/affiliates; keep it that way or get advice first. |
| Data licensing | ✅ | Shipped pipeline uses CC0 + public-domain sources only. Odds snapshots and Transfermarkt-derived squad data are user-supplied inputs, never redistributed in the repo or site artifacts. |
| Live feed ToS | ⚠ | ESPN's endpoint is undocumented; polled centrally at ~1 req/min, never from browsers, licensed fallback wired in. If the project monetizes or scales seriously, switch the flag to the licensed source permanently. |
| Code license | ✅ | MIT `LICENSE` file added (the site footer claimed MIT; now it's true). |

## 2. End-user experience

| Item | Status | Notes |
|---|---|---|
| First-visit comprehension | ✅ | Tagline, "as of" date, sim count in the header; probabilities always paired with context (CI, advancement bars). |
| Probability literacy | ⚠ | A 14% favorite means 86% something-else; the about page says this, but a "how to read this" micro-explainer near the title table would help casual users. Open: one-paragraph tooltip. |
| Mobile | ✅ | Single-column collapse, tap targets ≥40px on rows, no horizontal scroll at 360px. |
| Accessibility | ✅/⚠ | Semantic landmarks, alt text, keyboard-expandable team rows (tabindex + Enter), aria-expanded, alert role on the template warning, color never the sole encoding (numbers everywhere). Open: a full screen-reader pass on the round-bars visualization (title-attr only today). |
| Empty/failure states | ✅ | Missing artifacts, absent live.json, and template-bracket states all render explicit messages instead of blank sections. |
| Trust surface | ✅ | Methodology page in plain language, honest-limitations card, pre-registration explained, open JSON linked in the nav. |
| What's deliberately absent | — | Accounts, comments, push notifications, prediction game: all post-MVP by design; none block launch. |

## 3. Operations & reliability

| Item | Status | Notes |
|---|---|---|
| Pipeline failure mid-tournament | ✅ | Site always serves the last committed artifacts (static); a failed Action changes nothing user-visible. Stale data is the failure mode, signaled by the "as of" date. |
| Actions cron jitter | ✅ | Off-peak minute (`:17`), `workflow_dispatch` manual override documented. |
| Upstream data lag (martj42 volunteer updates) | ⚠ | If results lag, ratings lag. Mitigation: manual results entry is possible (CSV append) — documented as the emergency path in the workflow file comments. |
| Worker/KV quota | ✅ | Cron at 2-min intervals ≈ 21.6k invocations/mo, well within free tier; static JSON fetches don't count toward Worker limits. |
| Single maintainer risk | ❌ | Bus factor 1. Everything is reproducible from the repo, which is the honest mitigation. |
| Monitoring | ❌ | No uptime/staleness alerting. Cheap fix later: a scheduled Action that fails loudly when `as_of` falls >2 days behind during the tournament. |

## 4. Model & numbers (what users will encounter)

| Item | Status | Notes |
|---|---|---|
| Bracket template | ⚠ | The one launch blocker: verify R32 pairings + third pools against the official schedule (README step 6). The site warns until `_template` is false — by design impossible to miss. |
| Calibration drift in-tournament | ⚠ | Pool/temperature refit at matchweek gates; between gates, probabilities can be mildly miscalibrated after shocks. Acceptable and disclosed. |
| Small-sample backtests | ✅ disclosed | Four tournaments; stated plainly on the about page. |
| Eliminated-team queries | ✅ | Post-elimination, advancement probs go to ~0 naturally in re-sims; no special casing needed. |
| Adversarial reading | ⚠ | Someone will screenshot a wrong favorite and dunk. The pre-registered archive is the defense; consider a "model vs market vs reality" scoreboard page post-group-stage. |

## 5. Security & privacy

| Item | Status | Notes |
|---|---|---|
| Attack surface | ✅ | No backend, no forms, no user input, no cookies, no third-party scripts. XSS surface limited to our own JSON (team names are from our own static files). |
| Privacy/GDPR | ✅ | No personal data collected; no consent banner needed; stated in the footer. If analytics are ever added, prefer a cookieless option. |
| Supply chain | ✅ | Site has zero JS dependencies; Python deps are mainstream and pinned by floor version; Actions use tagged versions. |
| Secrets | ✅ | Only Cloudflare token + optional football-data key, both as platform secrets, never in repo. |

## 6. Growth & SEO (deferred consciously)

Per-entity URLs (`/team/spain`), JSON-LD structured data, per-team OG cards via
a Worker, and the embeddable odds widget are all specced in framework v3 §C.2
and deliberately post-MVP: they multiply shareability but don't change
correctness. The base OG card + open JSON API ship now.

## Verdict

Launchable once README step 6 (bracket verification) is done and real data has
flowed through `download → pipeline → simulate` on the maintainer's machine.
The two ❌ items (bus factor, staleness alerting) are accepted risks for a
free, open project — both have documented paths when they start to hurt.
