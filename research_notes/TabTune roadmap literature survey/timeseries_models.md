# Time-Series Foundation Models: Candidates for TabTune (2024 – Nov 2026)

> **Method caveat (read first):** Direct page fetches were blocked in this session. huggingface.co, arxiv.org, mistral.ai, theforecastingcompany.com, beri.net and cfotech.com.au all failed DNS because they are outside the session's allowed network, and GitHub API access was not enabled. Every citation below therefore points to the URL a **WebSearch result summary** drew on. The underlying page was **not opened** by me. Treat numbers as "reported by source, unverified against primary page". Anything from my background knowledge is labelled **[unverified – background]**. To verify, allow huggingface.co and arxiv.org in network settings and re-check the GIFT-Eval / fev-bench HF Spaces and model cards.
>
> Already in TabTune public (excluded as "new"): Chronos, Chronos-Bolt, Chronos-2, TimesFM 3, Toto 1, Toto 2, TiRex, TiRex-2, Time-MoE, TabPFN-TS-style forecasters.

## Q1. Which TSFMs lead GIFT-Eval / fev-bench (late 2026) and are missing from TabTune?

### Takeaway
Three releases from Sept–Oct 2026 are missing from TabTune, and each one claims #1 on fev-bench and/or GIFT-Eval: **Timestral** (Mistral, 15 Oct 2026, Modified-MIT, open weights), **t0 family** (The Forecasting Company, open weights, Apache-2.0 code) and **TimeH2O-v1** (H2O.ai, non-commercial). The claims come from different snapshots and comparison sets, so they are not directly comparable. Other strong models missing from TabTune are **IBM Granite PatchTST-FM** (2nd on GIFT-Eval in the t0 paper's table), **Moirai 2.0**, **Sundial** and the IBM TTM/FlowState line. **TimesFM 2.5** should also be added as a checkpoint because it is Apache-2.0, while TimesFM 3 weights are non-commercial.

### Cited Findings

**Leaderboard state (conflicting vendor claims)**
- fev-bench launch paper (Sep 2025): Chronos-2 had the highest average win rate and skill score, followed by TiRex and TimesFM-2.5 — [fev-bench arXiv 2509.26468](https://arxiv.org/pdf/2509.26468)
- TimeH2O-v1 (H2O.ai, late Oct 2026): H2O.ai claims #1 of 32 models on fev-bench by win rate (91.9%), as of 23 Oct 2026 on the default view. This is a vendor press release — [cfotech](https://cfotech.com.au/story/h2o-ai-launches-timeh2o-v1-forecasting-model-tops-fev-bench); [Yahoo Finance](https://finance.yahoo.com/technology/ai/articles/h2o-ais-timeh2o-beats-mistral-152900943.html)
- t0-gamma (The Forecasting Company): reported 49.42% skill score and 91.6% win rate on fev-bench, #1 of 30 entries, computed with `fev.pairwise_comparison` on per-task results as of 13 Oct 2026. Also claimed top on GIFT-Eval and TIME — [TFC blog](https://www.theforecastingcompany.com/blog/t0-gamma/)
- Timestral-209M (Mistral): claimed #1 on GIFT-Eval, TIME and fev-bench, and #1 on all six metrics among compared models, from a single checkpoint. The comparison was run by Mistral — [Mistral news](https://mistral.ai/news/timestral/); [arXiv 2610.37585](https://arxiv.org/html/2610.37585v1)
- One headline says Timestral is "beating TimesFM-3" — [aiweekly](https://aiweekly.co/alerts/mistral-opens-timestral-209m-time-series-model-beating-timesfm-3)
- GIFT-Eval (per t0 paper): t0-beta enters 3rd by CRPS and 3rd by MASE (CRPS 0.4738, MASE 0.6865), behind only TimesFM-3.0 and Granite-PatchTST-FM — [t0 arXiv 2609.24559](https://arxiv.org/pdf/2609.24559)
- GIFT-Eval (per Toto 2.0 paper): the three largest Toto 2.0 sizes lead the foundation models. 2.5B-FT and the "FnF" ensemble take the top two spots outright, and the 2.5B leads on CRPS rank 3.43, MASE rank 3.54 and CRPS 0.532 — [Toto 2.0 arXiv 2605.20119](https://arxiv.org/html/2605.20119v2); [Datadog blog](https://www.datadoghq.com/blog/ai/toto-2/)
- An aggregator lists 151 models on GIFT-Eval with "STRIDE w/ Synapse" first by average MASE rank (19.37). The snapshot is undated — [TSFM.ai](https://tsfm.ai/benchmarks/gift-eval)
- Ranking-method caveat: Papers with Code ranks by aggregate MASE/CRPS, while the GIFT-Eval Space defaults to mean per-configuration rank. The Space also has model-type filters (foundation / fine-tuned / ensemble / agentic) — [benchopt issue #59](https://github.com/benchopt/benchmark_tsfm/issues/59); [tfc-t0 issue #17](https://github.com/theforecastingcompany/tfc-t0/issues/17)

**Candidate catalogue (models NOT in TabTune)**

| Model | Org | Release | Architecture (1 line) | Tasks | Multivariate / covariates | Max context | Weight licence | Leaderboard standing | Fine-tuning |
|---|---|---|---|---|---|---|---|---|---|
| **Timestral** (5M/28M/123M/209M) | Mistral AI | 15 Oct 2026 | Patch-based probabilistic forecaster with 21-quantile output (details in paper) | Forecasting | Uni- and multivariate targets; past-only and known-future covariates, numeric or categorical (target-encoded); NaN for missing | 32,768 steps (1,024 patches); horizon 1,024 per pass | **Modified MIT**; a third party mentions a monthly-revenue clause (unconfirmed) | Self-reported #1 on GIFT-Eval, fev-bench and TIME | Not stated in results |
| Sources (Timestral) | [Mistral docs](https://docs.mistral.ai/models/timestral-1-0-209m.md), [HF Timestral-5M](https://huggingface.co/mistralai/Timestral-5M), [braindetox](https://braindetox.kr/en/posts/timestral_mistral_time_series_model.html), [startuphub](https://www.startuphub.ai/ai-news/ai-research/2026/mistral-says-209m-timestral-leads-on-three-benchmarks) | | | | | | | | |
| **t0-alpha / t0-beta / t0-gamma** | The Forecasting Company | t0 paper ~Sep 2026; PyPI last release 30 Jul 2026 | t0-alpha 102M and t0-beta 256M (t0-gamma size not found); "forecasting with context" | Forecasting | Past and known-future covariates (future covariates conditioned on, not forecast). Covariates add +6.3 fev-bench skill points on known-future tasks and +2.7 on past-only tasks | Not found | Open weights, downloadable without auth. **Code Apache-2.0**; weight licence unconfirmed | t0-gamma: #1 fev-bench (vendor, 13 Oct 2026). t0-beta: #3 on GIFT-Eval | ONNX export exists for t0-alpha only |
| Sources (t0) | [PyPI tfc-t0](https://pypi.org/project/tfc-t0/), [t0 arXiv](https://arxiv.org/pdf/2609.24559), [PyPI org](https://pypi.org/org/tfc/) | | | | | | | | |
| **TimeH2O-v1** | H2O.ai | late Oct 2026 | 108M (aggregator figure) | Forecasting, zero-shot | Univariate and with covariates (aggregator) | Not found | **h2o-model-non-commercial-licence-v1.0** (HF listing); h2ofm code Apache-2.0 | Vendor-claimed #1 on fev-bench by win rate | Not found |
| Sources (TimeH2O) | [HF discussions page](https://huggingface.co/h2oai/timeh2o-v1/discussions), [llmreference](https://www.llmreference.com/model/timeh2o-v1) | | | | | | | | |
| **Moirai 2.0** (R-small ≈11.4M) | Salesforce AI Research | Aug 2025 weights; arXiv 2511.11698 (Nov 2025) | Decoder-only; quantile loss and multi-token prediction; trained on 36M series | Forecasting | Not stated in results. Moirai 1.x supported any-variate input **[unverified – background]** | Not found | **Unconfirmed.** One Japanese overview says CC BY-NC 4.0 | Salesforce paper places it 5th on MASE and 6th on CRPS on GIFT-Eval at the time (secondary source). Blog claims 16% better MASE, 13% better CRPS and 44% faster than Moirai-1-Large | uni2ts library (fine-tuning scripts) |
| Sources (Moirai 2.0) | [arXiv 2511.11698](https://arxiv.org/html/2511.11698v1), [Salesforce blog](https://www.salesforce.com/blog/moirai-2-0/), [uni2ts GitHub](https://github.com/SalesforceAIResearch/uni2ts), [ai-souken](https://www.ai-souken.com/article/what-is-time-series-foundation-model), [beri.net](https://www.beri.net/article/timesfm-vs-chronos-vs-moirai-time-series-foundation-model-licence-bigquery-sagemaker-cost-2026) | | | | | | | | |
| **Moirai-MoE** | Salesforce | Oct 2024 | Sparse mixture-of-experts Moirai | Forecasting | Not stated in results | Not found | Unconfirmed | Superseded by Moirai 2.0 | uni2ts |
| Sources (Moirai-MoE) | [arXiv 2410.10469](https://arxiv.org/pdf/2410.10469), [Salesforce blog](https://www.salesforce.com/blog/time-series-morai-moe/) | | | | | | | | |
| **Granite PatchTST-FM (r1, r2)** | IBM Research | r1 2026; r2 reported later | PatchTST-style FM | Forecasting | Not found | Not found | Not confirmed. A secondary article calls r2 "SOTA open source" | 2nd on GIFT-Eval behind TimesFM-3.0 in the t0 paper's table | granite-tsfm toolkit **[unverified – background]** |
| Sources (PatchTST-FM) | [t0 arXiv](https://arxiv.org/pdf/2609.24559), [IBM research blog](https://research.ibm.com/blog/time-series-ai-enterprise), [ecosistemastartup](https://ecosistemastartup.com/ibm-granite-patchtst-fm-r2-sota-open-source-en-forecasting/) | | | | | | | | |
| **TTM (Tiny Time Mixers) r3** | IBM Granite | r3 in 2026 | MLP-mixer, ~few-M params, CPU inference | Forecasting (with channel-mixing / exogenous support **[unverified – background]**) | Exogenous support **[unverified – background]** | ProbTS lists TTM context 512 / horizon 96 (older revisions) | Apache-2.0 **[unverified – background]** | Not in results | Yes, fast fine-tune via tsfm_public **[unverified – background]** |
| Sources (TTM) | [IBM research blog](https://research.ibm.com/blog/time-series-ai-enterprise), [IBM Granite TS docs](https://www.ibm.com/granite/docs/models/time-series), [ProbTS](https://arxiv.org/pdf/2310.07446) | | | | | | | | |
| **FlowState r1 / r1.1** | IBM Research | arXiv 2508.05287 (Aug 2025); r1.1 2026 | S5 state-space encoder; sampling-rate-equivariant (time-scale adjustable) | Forecasting | Not found | Not found | **Conflicting:** HF YAML says apache-2.0, but the newer `ibm-research/flowstate` checkpoint is labelled "Research checkpoint (non-commercial use)" | Paper compares against TimesFM-2.5, TiRex, Chronos-2-Synth, Kairos, Sundial and Toto on GIFT-Eval (numbers not retrieved) | tsfm_public |
| Sources (FlowState) | [arXiv 2508.05287](https://arxiv.org/pdf/2508.05287), [HF commit](https://huggingface.co/ibm-research/flowstate/commit/f63b6711b6f3a1f44cec27b2afbe7abbd11cf4b1), [IBM docs](https://www.ibm.com/granite/docs/models/time-series) | | | | | | | | |
| **Sundial** | THUML, Tsinghua | arXiv 2502.00816 (Feb 2025); ICML 2025 oral | Decoder Transformer with TimeFlow loss (flow-matching next-patch distribution, no discrete tokens); pretrained on TimeBench (~1T points) | Generative probabilistic forecasting | Univariate **[unverified – background]** | Not found | **Apache-2.0** (gitee mirror of README) | Authors claim zero-shot SOTA on GIFT-Eval and FEV at release | Not found |
| Sources (Sundial) | [arXiv](https://arxiv.org/html/2502.00816v3), [gitee mirror](https://gitee.com/chenlinos/Sundial), [TS-Arena](https://dag-upb-ts-arena.hf.space/models/family/sundial) | | | | | | | | |
| **Kairos** | academic (arXiv 2509.25826) | Sep 2025 | Dynamic patching tokenizer and mixture-of-size encoder; >300B training points | Forecasting | Not found | Not found | Not found | Competitive zero-shot on GIFT-Eval vs YingLong, Moirai, Chronos, VisionTS, Toto-Base and Sundial-Base while 5.6× smaller | Not found |
| Sources (Kairos) | [arXiv 2509.25826](https://arxiv.org/html/2509.25826v2), [alphaxiv](https://alphaxiv.org/abs/2509.25826) | | | | | | | | |
| **TimesFM 2.5** (new checkpoint of an existing family) | Google | 2025; XReg covariates added Oct 2025 | ~200M decoder-only; univariate core with XReg covariate regression | Forecasting | Covariates via XReg; univariate core | 16k (reported) | **Apache-2.0.** TimesFM 3.0 weights use the timesfm-non-commercial-license-v1.0 (no production use, no mirroring) | Top-3 on fev-bench at launch | Yes (official fine-tune recipes **[unverified – background]**) |
| Sources (TimesFM) | [gitee mirror of repo README](https://gitee.com/jerrycell/timesfm), [mindstudio](https://www.mindstudio.ai/blog/timesfm-licensing-commercial-use), [aiweekly](https://aiweekly.co/alerts/google-timesfm-3-tops-forecast-benchmarks-non-commercial), [explainx](https://explainx.ai/blog/google-timesfm-3-multivariate-time-series-foundation-model-2026) | | | | | | | | |
| **TabPFN-TS-3 (official)** | Prior Labs / Freiburg | v1.1.0 README dated "12-05-2026" (likely 12 May 2026) | TabPFN backbone plus temporal featurization; 11M params for the v2-based model | Forecasting (covariate-informed is its strong suit) | Covariates yes | Not found | Not confirmed in results (TabPFN-v2 weights used a modified licence **[unverified – background]**) | Original TabPFN-TS reached top rank on GIFT-Eval at the time | Ships a fine-tuned TS-3 checkpoint; requires tabpfn>=8.0.0 |
| Sources (TabPFN-TS) | [GitHub tabpfn-time-series](https://github.com/PriorLabs/tabpfn-time-series), [arXiv 2501.02945](https://arxiv.org/html/2501.02945v4) | | | | | | | | |
| **Toto 2.0 2.5B-FT** (new checkpoint of an existing family) | Datadog | 2026 (blog ~May 2026) | Toto 2.0 family, 4M–2.5B params, 5 open-weight models | Forecasting (observability focus) | Toto supports exogenous covariates ([blog](https://www.datadoghq.com/blog/ai/toto-exogenous-covariates.md)) | Not found | **Apache-2.0** | GIFT-Eval leaders among FMs (self-reported). BOOM: 2.5B MASE ≈0.581, CRPS ≈0.349 (third-party import 10 Jul 2026) | FT variant published |
| Sources (Toto 2.0) | [Datadog blog](https://www.datadoghq.com/blog/ai/toto-2/), [HF Datadog papers](https://huggingface.co/Datadog/papers), [benchmarklist BOOM](https://benchmarklist.com/benchmarks/boom/) | | | | | | | | |
| **Lag-Llama** | Morgan Stanley / ServiceNow / Mila et al. | 2023/24 | LLaMA-style decoder on lag features | Univariate probabilistic forecasting | Univariate | Not found | Apache-2.0 **[unverified – background]** | Weak in independent load-forecasting study ("no clear advantage over GP") | Yes |
| Sources (Lag-Llama) | [NeurIPS](https://neurips.cc/virtual/2023/76664), [arXiv 2412.12834](https://arxiv.org/pdf/2412.12834), [survey](https://arxiv.org/pdf/2504.04011) | | | | | | | | |
| **Timer / Timer-XL** | THUML | 2024 | Decoder-only generative transformer; Timer-XL pretrained on ~1B points | Forecasting | Timer-XL multivariate **[unverified – background]** | Not found | Not found | Superseded by Sundial | Not found |
| Sources (Timer-XL) | [survey arXiv 2504.04011](https://arxiv.org/pdf/2504.04011) | | | | | | | | |

- YingLong appears only as a baseline in Kairos' GIFT-Eval figure. I found no detail on its own release — [Kairos arXiv](https://arxiv.org/html/2509.25826v2)
- Toto-Open-Base-1.0 ranked #1 on the original BOOM leaderboard, ahead of Moirai 1.1, TimesFM 2.0 and Chronos-Bolt — [BOOM HF space about.py](https://huggingface.co/spaces/Datadog/BOOM/blob/main/src/about.py)
- BOOM covers 350M observations across 2,807 series and follows GIFT-Eval methodology — [Datadog blog](https://www.datadoghq.com/blog/ai/toto-boom-unleashed/)

### Inferences
- **Proposed top 6 to add, ranked by value to TabTune:**
  1. **Timestral (Mistral).** Newest model, claimed #1 on all three leaderboards, small sizes (5M–209M), 32k context, multivariate with past and future covariates including categorical ones, near-permissive Modified-MIT. It fits TabTune's covariate pipeline directly. Check the revenue clause before shipping it in the internal version.
  2. **t0 family (TFC).** Strongest independent-looking fev-bench numbers, covariate-aware, open weights, Apache code, pip-installable (`tfc-t0`) and ONNX for t0-alpha. Confirm the weight licence.
  3. **IBM Granite TS suite (TTM-r3, FlowState-r1.1, PatchTST-FM, TSPulse).** One integration (`tsfm_public`) unlocks forecasting, classification, anomaly detection and imputation. The models are CPU-friendly, mostly Apache. PatchTST-FM is reported #2 on GIFT-Eval. This is the best "breadth" pick and the best fit for the internal version, which has no TS support yet.
  4. **TimesFM 2.5 checkpoint.** Cheap to add because the TimesFM loader already exists. It is the **only commercially usable TimesFM**, since 3.0 is non-commercial. That matters for the internal version.
  5. **Moirai 2.0 (uni2ts).** Standard baseline, tiny (11M), widely cited, and has a fine-tuning library. Lower priority if the CC BY-NC licence is confirmed.
  6. **Sundial (THUML).** Apache-2.0, generative flow-matching sampler that gives real sample paths rather than only quantiles, ICML oral. It diversifies the ensemble and leaderboard.
  - Honourable mentions: the **official TabPFN-TS-3 checkpoint**, which is an upgrade of TabTune's existing TabPFN-TS-style path and synergistic with its tabular core; the **Toto 2.0 2.5B-FT** checkpoint; and Kairos as a research option. **TimeH2O-v1** is low priority because of its non-commercial licence, despite its leaderboard claim.
- Licence summary: permissive, or reported permissive, are Toto 2.0 (Apache), TimesFM ≤2.5 (Apache), Sundial (Apache), Granite TTM/TSPulse (Apache, background), t0 code (Apache) and Timestral (Modified MIT). Non-commercial are TimesFM 3 (already in TabTune, so flag it), TimeH2O-v1, possibly Moirai 2.0, and the FlowState research checkpoint.
- **Action for TabTune public:** add licence metadata and warnings per checkpoint. TimesFM 3 is already integrated but cannot be used in production.

### Gaps
- Live GIFT-Eval and fev-bench leaderboards could not be opened (huggingface.co blocked), so there is no independent current ranking.
- Not researched or not found: **Mamba4Cast**, YingLong details, t0-gamma's parameter count and weight licence, Timestral architecture and fine-tuning API, PatchTST-FM licence, Moirai 2.0 licence text, TTM-r3 context length.
- The ranking of TimeH2O vs t0-gamma vs Timestral on one common snapshot is unknown.

## Q2. What time-series classification / representation FMs exist?

### Takeaway
The main options are **Mantis / Mantis-V2** (ICML 2026, synthetic-data contrastive pretraining), **MOMENT** (multi-task, 40/125/385M) and **IBM TSPulse** (classification, anomaly detection, imputation and similarity search, CPU-friendly). An independent 2026 benchmark nonetheless finds MiniRocket, MultiRocket and Hydra still beat pretrained FMs on UCR/UEA. A ROCKET-family baseline plus an FM embedding path is therefore the right module design.

### Cited Findings
- Mantis: transformer pretrained only on synthetic data with self-supervised contrastive learning. Authors report new SOTA across four dataset collections. Test-time tricks are intermediate-layer representations, self-ensembling and cross-model embedding fusion. Published ICML 2026 (PMLR v306) — [PMLR](https://proceedings.mlr.press/v306/feofanov26a.html); [ICML poster](https://icml.cc/virtual/2026/poster/62438)
- Independent benchmark of 13 methods on 33 UCR/UEA datasets with a frozen protocol (mean pooling and logistic probe): MiniRocket, MultiRocket and Hydra rank above all pretrained models. MOMENT is the best pretrained model at ~2 ranks behind MiniRocket. Mantis-V2 has mean rank 9.733, behind TiRex and catch22. Results depend on the probe protocol — [aimultiple](https://aimultiple.com/time-series-classification)
- MOMENT: model family of 40/125/385M params pretrained on 1.13B points — [survey 2504.04011](https://arxiv.org/pdf/2504.04011); [MOMENT arXiv 2402.03885](https://arxiv.org/pdf/2402.03885)
- TSPulse (IBM, r1): dual-space (time and frequency) masked reconstruction. Recommended for anomaly detection, classification and other non-forecasting tasks. HF id `ibm-granite/granite-timeseries-tspulse-r1`, loaded via tsfm_public with separate reconstruction and classification classes — [IBM research blog](https://research.ibm.com/blog/time-series-ai-enterprise); [IBM tutorial](https://www.ibm.com/think/tutorials/time-series-analysis-granite-time-tspulse)
- IBM says the Granite TS family (FlowState, TTM, TSPulse) covers forecasting, classification, anomaly detection, imputation and similarity search with GPU-free inference — [IBM Granite TS docs](https://www.ibm.com/granite/docs/models/time-series)

### Inferences
- A TS classification module could expose (a) ROCKET-family baselines, (b) frozen FM embeddings from Mantis, MOMENT, TSPulse or TiRex fed to TabTune's existing tabular learners, and (c) light fine-tuning. Option (b) reuses TabTune's tabular core, much like the SeriesFeaturizer bridge does.
- TiRex appearing in the classification ranking suggests TabTune's existing TiRex integration could expose embeddings cheaply.

### Gaps
- No results on NuTime. MOMENT and Mantis licences were not verified (MOMENT is MIT **[unverified – background]**).

## Q3. Anomaly detection, imputation, irregular series, hierarchical and calibration modules

### Takeaway
FM-based anomaly detection exists (TSPulse, TimeRCD, Forecast2Anomaly), but an ICLR 2026 study finds FMs barely beat one-liner baselines on TSB-AD. Any anomaly module must ship strong simple baselines and VUS-PR evaluation. Irregular-series FMs are emerging (FlexTSF, a 2026 hybrid-attention decoder). I found no foundation-model-native hierarchical reconciliation, so classical probabilistic reconciliation is the gap to fill.

### Cited Findings
- TSB-AD (NeurIPS 2024 D&B): ~1,020–1,070 curated series from 33–40 domains. Recommends VUS-PR and warns about data contamination when evaluating FMs — [NeurIPS](https://neurips.cc/virtual/2024/poster/97690); [PyPI TSB-AD](https://pypi.org/project/tsb-ad)
- IBM reports TSPulse beat top statistical models by 24% and larger FMs by ≥33% on TSB-AD (vendor claim) — [IBM tutorial](https://www.ibm.com/think/tutorials/time-series-analysis-granite-time-tspulse)
- "When Foundation Models are One-Liners" (ICLR 2026): anomaly detection with MOMENT, Chronos, TimesFM, Time-MoE and TSPulse does not differ significantly from simple one-liner baselines — [ICLR](https://iclr.cc/virtual/2026/poster/10010437); [mlanthology](https://mlanthology.org/iclr/2026/zhu2026iclr-foundation/)
- TimeRCD: zero-shot anomaly detection FM using synthetic data and relative context discrepancy pretraining — [arXiv 2509.21190](https://arxiv.org/pdf/2509.21190)
- Forecast2Anomaly (F2A): adapts multivariate forecasting FMs for anomaly *prediction*, evaluated on TSB-AD-M — [arXiv 2511.03149](https://arxiv.org/html/2511.03149v1)
- Imputation: "time-indexed FMs" (TabPFN-TS, MoTM) studied for imputation — [arXiv 2511.05980](https://arxiv.org/html/2511.05980v1)
- Irregular series: FlexTSF natively supports regular and irregular series — [arXiv 2410.23160](https://arxiv.org/abs/2410.23160). A 2026 decoder-only hybrid-attention FM jointly models values and continuous timestamps — [arXiv 2609.22836](https://arxiv.org/pdf/2609.22836)
- Hierarchical: probabilistic reconciliation via conditioning / Bayes' rule, implemented in R `bayesRecon`. Results were classical, not FM-specific — [bayesRecon](https://packages.oit.ncsu.edu/cran/web/packages/bayesRecon/readme/README.html); [arXiv 2210.02286](https://arxiv.org/abs/2210.02286v2); [Hierarchies Everywhere arXiv 2209.15583](https://arxiv.org/pdf/2209.15583)
- Conformal prediction on FM forecasts studied in 2025 — [arXiv 2507.08858](https://arxiv.org/pdf/2507.08858)

### Inferences
- Module gaps for TabTune, in priority order:
  1. **Anomaly detection module.** Use TSPulse or forecast-residual scoring with any integrated TSFM, plus one-liner and IsolationForest baselines and VUS-PR metrics (TSB-AD loader).
  2. **Classification and embedding module** (see Q2).
  3. **Hierarchical reconciliation.** Add bottom-up, MinT and probabilistic reconciliation by conditioning on top of any TSFM's quantiles or samples. This is model-agnostic and cheap.
  4. **Calibration beyond conformal.** Quantile recalibration, CRPS-based ensemble weighting and adaptive conformal for drift. This is an inference only; I found no specific FM paper here.
  5. **Imputation API** using masked-reconstruction FMs (TSPulse, MOMENT) or time-indexed models.
  6. **Irregular / event series.** Research-stage; low priority.

### Gaps
- No live TSB-AD leaderboard found (README says "Coming Soon"). No FM-specific hierarchical or calibration papers surfaced in one search round.

## Q4. State of multimodal (TS + text), LLM-based and agentic forecasting

### Takeaway
Multimodal TS+text is active but immature. The practical pattern in 2026 is **a numeric TSFM prior plus an LLM reviser or router** (POSTTIME, TimeCopilot, TimeRouter), not end-to-end LLM forecasters. Covariate-aware numeric FMs (t0, Timestral) also absorb much of the "context" value. TabTune would benefit most from a thin agentic / routing layer over its existing models.

### Cited Findings
- ChatTime (AAAI 2025): treats TS as a "foreign language". Zero-shot and context-guided forecasting on 8 datasets. 1.7B base on HF — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33384); [HF](https://huggingface.co/ChengsenWang/ChatTime-1-7B-Base)
- ChatTS: ~14B (Qwen2.5-14B) aligned via synthetic data for TS understanding and reasoning, not forecasting. Chronicle (May 2026): 324M model jointly pretrained on text and TS from scratch — [arXiv 2605.20268](https://huggingface.co/papers/2605.20268.md)
- POSTTIME (May 2026): post-trains an LLM (Gemma-3-4B) as a context-guided reviser over a TSFM prior (TimesFM-2.5). Beats standalone TSFMs and LLM-only baselines on TimesX — [alphaXiv 2605.29401](https://www.alphaxiv.org/abs/2605.29401.md)
- TimesX (ICML 2026): context-enriched benchmark. Simple ensembles using rich textual context can beat strong baselines — [alphaXiv 2607.06973](https://www.alphaxiv.org/abs/2607.06973)
- Other names in this literature: TTS, Aurora, Time-MQA, Time-R1 (not individually verified) — [same alphaXiv](https://www.alphaxiv.org/abs/2605.29401.md)
- Conversational TSFM ensemble (arXiv 2512.16022): an LLM-orchestrated ensemble that reports beating leading TSFMs on GIFT-Eval CRPS and MASE — [arXiv 2512.16022](https://arxiv.org/html/2512.16022)
- TimeCopilot (arXiv 2509.00616): open-source agentic framework where an LLM coordinates TSFMs and classical models (feature analysis, model selection, CV, forecast). Reports SOTA probabilistic results on GIFT-Eval at low cost — [arXiv](https://arxiv.org/abs/2509.00616v3)
- TimeRouter (arXiv 2606.11625): routes among existing TSFMs and reports SOTA MASE on the GIFT-Eval leaderboard — [arXiv](https://arxiv.org/pdf/2606.11625)
- GIFT-Eval now has an "agentic" model-type filter — [benchopt issue #59](https://github.com/benchopt/benchmark_tsfm/issues/59)
- TimeSeriesGym (NeurIPS 2025): benchmark for ML-engineering agents on TS tasks. Frontier LLMs plus AIDE produce valid but often unsound solutions on hard tasks — [arXiv 2505.13291](https://arxiv.org/pdf/2505.13291)
- Instruction-conditioned in-context TS FM (arXiv 2603.22586) — [arXiv](https://arxiv.org/pdf/2603.22586)

### Inferences
- TabTune already has a Leaderboard, Ensemble and Benchmark, so a **learned router / meta-selector** (TimeRouter-style, using series features from SeriesFeaturizer) is a natural, high-value module. An optional **LLM agent wrapper** (TimeCopilot-style) is a lower-effort add-on.
- Text-context forecasting: support a "text covariate" slot (embedded text feeding covariate-aware FMs), or a POSTTIME-style LLM post-hoc reviser. Treat this as experimental.

### Gaps
- No licences or weights checked for Chronicle, POSTTIME or TimeRouter. No head-to-head multimodal leaderboard exists (TimesX is the closest).
