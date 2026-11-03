# Benchmark and Library Ecosystem for Tabular and Time-Series FMs (2024 to Nov 2026): Gaps and Integration Opportunities for TabTune

Method note: written 2026-11-03. huggingface.co, arxiv.org, auto.gluon.ai and some vendor or news sites could not be reached from this session (DNS blocked), so the live HF leaderboards (TabArena, GIFT-Eval, fev-bench) could not be opened. github.com pages were opened directly. Each citation is tagged:
- **[opened]**: I fetched and read the page.
- **[search]**: the claim comes from a web-search result summary. The URL was returned but the page itself was not opened, so the claim needs checking before it goes in a final report.

## Q1. Benchmarks TabTune should integrate as one-line runners: what each measures, how to run it, licence, current leaders

### Takeaway
The tabular benchmark landscape has settled on **TabArena** as the living reference. It has a successor, **BeyondArena**, which adds temporal and grouped, non-IID tasks. **TALENT**, **TabReD**, **MultiTab** and the OpenML suites (CC18, CTR23) fill specific niches, and **RelBench/RelArena** covers relational data. For time series, **GIFT-Eval** and **fev-bench** are the standard forecasting boards, **BOOM** covers observability and **TSB-AD** covers anomaly detection. Almost all are Apache-2.0 and pip- or uv-installable, which makes them good candidates for thin wrappers. By late 2026 the top of every board is held by in-context-learning FMs: TabPFN-3/3.5, LimiX-2 and TabICLv2 on tabular, and Chronos-2, TimesFM-3 and Toto-2 on time series. Many of the "#1" claims are vendor-reported.

### Cited Findings

**TabArena / BeyondArena (tabular, living)**
- TabArena is a living benchmark hosted under the AutoGluon org. One codebase drives two benchmarks. **TabArena-v0.1** has 51 curated IID datasets with 9–30 splits each, 27+ methods (10+ tabular FMs) and more than 50M trained models. **BeyondArena** covers IID, temporal and grouped tasks across **142 datasets** and "will supersede TabArena-v0.1 in the future." — [opened: GitHub autogluon/tabarena](https://github.com/autogluon/tabarena)
- Protocol: each method is "represented at its best" through cross-validated ensembles, method-contributed HPO search spaces, early stopping, refit, parallel bagging and memory estimation. Official validation is 8-fold × 1. **TabArena-Lite** uses the first split of each dataset, with the default configuration plus about 25 random HPO configurations. Results include test and validation error, train time and inference time. Leaderboards report Elo, win-rate, average rank and improvability. — [opened](https://github.com/autogluon/tabarena)
- How to run: install with `uv pip install --prerelease=allow -e "./packages/tabarena[benchmark]"` (Python 3.11–3.13), then run `examples/benchmarking/run_quickstart_tabarena_model.py`. To add a method, evaluate it on TabArena-Lite and open a PR. Maintainers re-run it on the full task set. Licence: **Apache-2.0**. Papers: TabArena-v0.1 (arXiv 2506.16791, NeurIPS 2025 D&B) and BeyondArena (arXiv 2606.30410). — [opened](https://github.com/autogluon/tabarena)
- Leaderboard status in 2026: a June 2026 third-party article reports TabPFN-3 (default) at **1673 Elo on board v0.1.4** against XGBoost at 1375. — [search: ecorpit.com](https://ecorpit.com/tabular-foundation-models-tabpfn-vs-xgboost-enterprise-2026/). A September 2026 article says **LimiX-2** (400M params, Stability AI and Tsinghua LimiX team) reports **1935 Elo** and tops the live board ahead of TabPFN-3.5. — [search: intelligentliving.co](https://www.intelligentliving.co/limix-2-tops-tabarena-tabular-ai/). **Unverified**: the Elo scales differ across board versions, the figures are vendor-reported, and the live board at tabarena.ai or HF could not be opened.
- A BeyondArena paper titled "Beyond IID: How General Are Tabular Foundation Models, Really?" (arXiv 2606.30410) exists. — [search](https://arxiv.org/pdf/2606.30410). I could not open the abstract, so its findings are a gap.

**TALENT (tabular toolkit + benchmark)**
- 35+ deep methods, 10+ classical methods, 300 datasets (120 binary, 80 multiclass, 100 regression) plus a 22-dataset large benchmark. — [search: GitHub LAMDA-Tabular/TALENT](https://github.com/LAMDA-Tabular/TALENT); JMLR paper "Talent: A Tabular Analytics and Learning Toolbox" — [search](http://jmlr.org/papers/volume26/25-0512/25-0512.pdf)
- FM tech reports such as TabICLv2 and TabPFN-3 use a **274-dataset TALENT subset**, excluding 26 datasets used during development. Metrics are accuracy for classification and RMSE for regression. — [search: TabICLv2](https://arxiv.org/pdf/2602.11139); [search: TabPFN-3 report](https://arxiv.org/pdf/2605.13986)

**TabReD (industry-grade, temporal shift)**
- 8 industry datasets: 5 regression (Sberbank Housing, Cooking Time, Delivery ETA, Maps Routing, Weather) and 3 classification (Homesite Insurance, Ecom Offers, Homecredit Default). They feature temporal distribution shift and rich feature-engineered feature sets. Splits are `default`, `random-{0,1,2}` and `sliding-window-{0,1,2}`. Download with `uvx tabred download [names]`, which needs Kaggle auth. Apache-2.0. ICLR 2025 Spotlight (arXiv 2406.19380). — [opened: GitHub yandex-research/tabred](https://github.com/yandex-research/tabred)

**MultiTab, OpenML-CC18/CTR23, TabZilla**
- MultiTab (2025): 196 public datasets categorised by data characteristics (sample size, label imbalance, feature interaction), 13 models. The goal is regime-specific analysis rather than aggregate scores. Data, code and logs are on HF (LGAI-DILab/Multitab). — [search: arXiv 2505.14312](https://arxiv.org/abs/2505.14312)
- OpenML-CTR23: 35 curated regression problems on OpenML. — [search](https://arxiv.org/pdf/2305.02997v2)
- TabZilla: 176 datasets including OpenML-CC18, plus a released 36-dataset "hard" subset (GitHub naszilla/tabzilla). Its finding: HPO matters more than the choice between NNs and GBDTs. — [search: mcml.ai](https://mcml.ai/publications/fhf+23/)

**RelBench / RelArena (relational, multi-table)**
- KumoRFM-2 is evaluated on 41 predictive tasks across RelBench v1, RelBench v2, SALT and 4DBInfer. — [search: arXiv 2604.12596](https://arxiv.org/html/2604.12596v1)
- Leaderboard validity is disputed. A RelBench author states that a reproducible KumoRFM-v2 had not been submitted to RelBench or RelArena, so its numbers "aren't officially validated." — [search: Kapso issue #121](https://github.com/Leeroo-AI/kapso/issues/121). A later RelBench submission issue lists a "KumoRelational" entry. — [search: relbench issue #421](https://github.com/stanford-star/relbench/issues/421)
- A newer paper, "RelArena-α, TabPFN-Rel and RPI," pushes open, reproducible relational learning, which suggests TabPFN-style ICL models are being extended to relational data. — [search: arXiv 2608.16319](https://arxiv.org/pdf/2608.16319)
- A third-party SOTA table puts KumoRFM-2 second on RelBench v2 binary classification, behind a supervised "Rel-Agent." — [search: sota2.com](https://www.sota2.com/research/sota/binary-classification-on-relbench-v2) (low-authority aggregator)

**GIFT-Eval (TS forecasting)**
- Chronos-2 (Oct 2025) reports beating the previous leaders, TiRex and TimesFM-2.5, on win rate and skill score under WQL and MASE. — [search: arXiv 2510.15821](https://arxiv.org/pdf/2510.15821)
- A paper that froze its model selection on 9 Apr 2026 reports Chronos-2 ranked #1 and TTM-R3-PT #5. Both were pretrained on a subset of GIFT-Eval training data, so leakage or contamination caveats apply. — [search: arXiv 2607.23146](https://arxiv.org/pdf/2607.23146)
- Claim that Google's TimesFM 3.0 (Aug 2026) is #1 on fev-bench and TIME and #1 among FMs on GIFT-Eval. — [search: Medium](https://medium.com/@visrow/timesfm-vs-chronos-vs-moirai-vs-timegpt-time-series-forecasting-models-compared-2026-live-demo-3fb1184a68ec). **Unverified**: weak source and no primary Google source found.
- Moirai 2.0 report — [search: arXiv 2511.11698](https://arxiv.org/pdf/2511.11698)

**fev / fev-bench (TS forecasting library + benchmark)**
- `pip install fev`. Build a `fev.Task` from an HF dataset, loop over `task.iter_windows()`, call `task.evaluation_summary(...)`, then aggregate with `fev.leaderboard(...)`. It supports past and known-future covariates and point and probabilistic metrics (MASE is the default). Version 0.6.0 appears in the examples. Apache-2.0. Described as research code, "not production-ready." The leaderboard lives at HF Space autogluon/fev-bench, which I could not open. — [opened: GitHub autogluon/fev](https://github.com/autogluon/fev)

**BOOM (observability TS)**
- Datadog's benchmark: about 350M observations across 2,807 real-world observability series. It follows the GIFT-Eval methodology. Released May 2025 under a permissive licence. — [search: Datadog blog](https://www.datadoghq.com/blog/ai/toto-boom-unleashed/); [search: Telecom Reseller](https://telecomreseller.com/2025/05/22/datadog/)
- Datadog reports Toto 2.0 takes the top spots on BOOM, GIFT-Eval and TIME. The HF cards list a BOOM CRPS of 0.349 for the 2.5B and 1B models. — [search: Toto 2.0 blog](https://www.datadoghq.com/blog/ai/toto-2/); [search: HF Toto-2.0-2.5B](https://huggingface.co/Datadog/Toto-2.0-2.5B). These are vendor numbers.

**TSB-AD (TS anomaly detection)**
- 1,070 series from 40 datasets, split into TSB-AD-U (univariate) and TSB-AD-M (multivariate). VUS-PR is recommended as the most reliable metric. 40 algorithms in three groups: statistical, NN and FM (OFA, Lag-Llama, Chronos, TimesFM, MOMENT). Finding: simple and statistical methods often beat NNs, NNs are strong on multivariate data, and FMs are strong on point anomalies. Install with `pip install TSB-AD` and run `python -m TSB_AD.main --AD_Name IForest`. Apache-2.0. An online leaderboard accepts submissions. — [opened: GitHub TheDatumOrg/TSB-AD](https://github.com/TheDatumOrg/TSB-AD)

### Inferences
- **Priority one-line runners for TabTune**, ordered by value over effort: (1) a TabArena-Lite and full TabArena adapter that exposes TabTune models as TabArena "methods", which also opens a route to official leaderboard submissions; (2) BeyondArena temporal and grouped splits, since non-IID is where FM claims are weakest; (3) TabReD, which needs only a uvx download and has realistic shift; (4) TALENT-274 for comparability with FM tech reports; (5) fev-bench and GIFT-Eval for TS (fev's Task API makes a wrapper trivial); (6) TSB-AD, which matters only if TabTune adds TS anomaly detection; (7) BOOM for observability users; (8) OpenML CC18/CTR23 via openml-python as a cheap sanity suite. RelBench should wait until TabTune supports multi-table data.
- TabTune's existing TabularLeaderboard would be more credible if it reproduced TabArena's protocol (8-fold CV, Elo, improvability) instead of an in-house one. Leaderboard numbers would then be directly comparable to tabarena.ai.
- Data contamination is a real problem. TALENT-274 excludes development datasets, and Chronos-2 and TTM were trained on GIFT-Eval training data. Benchmark runners should carry per-model "seen-dataset" exclusion lists.

### Gaps
- The current live top-10 of TabArena, GIFT-Eval and fev-bench could not be verified because HF and tabarena.ai were unreachable. The LimiX-2 #1 and TimesFM-3 #1 claims remain unverified.
- BeyondArena's findings, MultiTab's top models and TabReD's model findings were not read; the papers were inaccessible.
- The current TSB-AD leaderboard leader is not given on the GitHub page.
- Licences of MultiTab, OpenML-CTR23 and TabZilla were not checked. OpenML datasets carry per-dataset licences.

## Q2. Competing and adjacent libraries: what they offer that TabTune lacks

### Takeaway
**AutoGluon** is the closest competitor. Its 1.6 `extreme` preset is a commercially usable portfolio of tabular FMs (Nori, TabICLv2, TabDPT-Turbo, TabPrep-LGBM), it has a separate `noncommercial` preset with TabPFN-3, and it integrates Chronos-2 (with LoRA or full fine-tuning) and Toto-2 for TS. **Prior Labs' tabpfn-extensions** sets the bar for FM add-ons: SHAP interpretability, many-class handling, unsupervised generation and outlier detection, embeddings, image columns, TabEBM augmentation, CRT p-values and BO surrogates. **PyTorch Frame** leads on multimodal and relational columns. **Darts** and **sktime** have adopted TSFM zoos with conformal prediction and anomaly detection. **Nixtla** offers hosted multivariate TimeGPT-2.1. Several libraries now ship MCP or agent integrations.

### Cited Findings

**AutoGluon (tabular + TimeSeries)**
- Release dates: 1.5.0 on 19 Dec (2025, inferred), 1.6.0 on 05 Aug 2026, 1.6.3 on 18 Sep 2026. — [opened: GitHub releases](https://github.com/autogluon/autogluon/releases)
- 1.5.0 added RealTabPFN-2.5 (non-commercial, must be specified manually, HF terms gate), TabDPT and experimental TabPrep-LightGBM. Its `extreme` preset is a fixed FM portfolio and needs a GPU with ideally 20 GB+ VRAM. 1.5.0 also added **Chronos-2** for TS (zero-shot, LoRA or full fine-tune). — [opened](https://github.com/autogluon/autogluon/releases)
- 1.6.0 added **Nori, TabICLv2, TabPFN-2.6, TabPFN-3 and TabDPT-Turbo**. The new `extreme` preset (Nori + TabICLv2 + TabDPT-Turbo + TabPrep-LGBM) is "free for commercial use." A new **`noncommercial` preset** adds TabPFN-3, whose commercial use needs a Prior Labs licence. 1.6.0 also added **Toto-2** for TS, new forecasting metrics, and a `validation_structure` fit argument for grouped or temporal data. It removed "lite" mode, cut import time by about 25%, supports Python 3.10–3.13, and cannot load models from older versions. — [opened](https://github.com/autogluon/autogluon/releases)
- The 1.5 docs describe the extreme preset as "meta-learned on TabArena" (TabPFNv2, TabICL, Mitra, TabDPT, TabM). The reported win rate over 1.4 Extreme on the 51 TabArena datasets conflicts across pages (70% vs 85%). — [search: AutoGluon 1.5 whats_new](https://auto.gluon.ai/stable/whats_new/v1.5.0.html) (site not reachable)

**Prior Labs tabpfn + tabpfn-extensions**
- tabpfn-extensions modules: **interpretability** (SHAP, feature selection), **many_class** (more classes than the checkpoint supports), **unsupervised** (data generation, outlier detection), **embedding** (dense sample embeddings), **image** (image columns to frozen DINOv3 features), **tabebm** (energy-based augmentation), **pval_crt** (CRT feature-relevance p-values), **bayesian_optimization** (TabPFN as BO surrogate with exact EI). Examples also include survival and predictive_distribution. Embedding and BO work only with the local `tabpfn` package; most others also work with `tabpfn-client`. Apache-2.0. Earlier modules such as post-hoc ensembles, HPO, rf_pfn and classifier_as_regressor were **not listed** on the current README, so they may have been removed or moved. — [opened: GitHub PriorLabs/tabpfn-extensions](https://github.com/PriorLabs/tabpfn-extensions)
- TabPFN-3 (12 May 2026): native support for up to **160 classes**. The docs table lists 1M rows × 2,000 columns, while the changelog says 1M × 200 features (conflict). Variants: base (local), **Plus** (native text, API/VPC only) and **Thinking** (extra test-time compute, API/VPC, 200k rows). Licence is "TABPFN-3.0 License v1.0" (research and internal evaluation), and the models page lists it as non-commercial. Code moved to Apache-2.0 at package v9.0.0. **TabPFN-3.5** shipped 15 Sep 2026 and is now the default `ModelVersion.V3_5`. — [search: docs.priorlabs.ai changelog](https://docs.priorlabs.ai/changelog/tabpfn-3); [search: GitHub TabPFN releases](https://github.com/PriorLabs/TabPFN/releases); [search: TabPFN-3.5 report](https://arxiv.org/pdf/2609.17895)
- Serving features: `fit_mode="fit_with_cache"` computes and reuses an attention KV cache, at about 7 GB GPU per estimator at 1M rows (56 GB for the default 8 estimators). The managed API supports caching only for TabPFN-3+ and not in Thinking mode. A **TabPFN-as-MLP/Tree** output engine is recommended for large data and CPU inference. There is a cookbook for wrapping TabPFN as an **MLflow PythonModel** and registering it in Unity Catalog / Mosaic AI Model Serving. — [search: docs.priorlabs.ai kv-cache](https://docs.priorlabs.ai/capabilities/kv-cache); [search: best practices](https://docs.priorlabs.ai/best-practices); [search: cookbook](https://docs.priorlabs.ai/cookbook)

**PyTorch Frame**
- Column types: numerical, categorical, multicategorical, timestamp, embedding, text (embedded or tokenized) and image (embedded). Text-encoder integrations: OpenAI, Cohere, HF and Voyage, with LoRA fine-tuning of the text encoder. Models: Trompt, FT-Transformer, ResNet, TabNet, ExcelFormer, TabTransformer, plus GBDTs with Optuna HPO. With PyG it enables relational-DB deep learning (RelBench). MIT. — [opened: GitHub pyg-team/pytorch-frame](https://github.com/pyg-team/pytorch-frame)

**TALENT toolkit**: a single API across 35+ deep and 10+ classical methods on 300 datasets. — [search](https://github.com/LAMDA-Tabular/TALENT)

**Darts**
- Integrated zero-shot TSFMs: Chronos2Model, TimesFM2p5Model, TimesFM3Model, TiRexModel, TiRex2Model, PatchTSTFMModel, T0Model. There is no Moirai. Features: **ConformalNaiveModel / ConformalQRModel** for calibrated intervals on pretrained global models, backtesting, the **`darts.ad`** anomaly-detection module (scorers, detectors, aggregators, PyOD wrapper) and SHAP explainability. ONNX and distributed inference are not mentioned. Apache-2.0. — [opened: GitHub unit8co/darts](https://github.com/unit8co/darts)

**sktime**
- Reported sktime 1.0 with TSFMs under one interface (TimesFM 2, Sundial, Timer-S1, Kronos, FalconTST, WindFM, Moirai 2.0), plus "agentic pipelines, MCP integration, and a new pretraining API." — [search: LinkedIn post](https://it.linkedin.com/in/sxandie). **Unverified**: social-media source only.

**Nixtla**
- The TimeGPT-2 family (mini, 2 and pro; private preview Oct 2025) claims up to 60% accuracy improvement over the previous generation. **TimeGPT-2.1** is the first multivariate TimeGPT, with better exogenous handling and guaranteed non-negative outputs, and is now the default in the client (>= 0.7.4). The current docs also list timegpt-2-lab. The `nixtla` SDK was at 0.9.0 on 21 Sep 2026. GA status is unclear. — [search: Nixtla TimeGPT-2.1 blog](https://www.nixtla.io/blog/timegpt-2-1-announcement); [search: quickstart](https://www.nixtla.io/docs/forecasting/timegpt_quickstart); [search: PyPI nixtla](https://pypi.org/project/nixtla/)

**uni2ts (Salesforce)**
- Moirai-2.0-R-small was released Aug 2025. The fine-tuning module was improved and runs through a Hydra-configured CLI after data conversion. — [search: PyPI mirror of uni2ts README](https://pypi.org/project/tsagentkit-uni2ts/) (mirror, not the official repo)

**Agent / LLM integrations**
- Prior Labs' official TabPFN MCP server (beta) exposes fit-and-predict and predict-from-fitted tools over streamable HTTP and is usable from ChatGPT, Claude and n8n. — [search: Show HN](https://news.ycombinator.com/item?id=46901522)
- Vector Institute's **tabctx-mcp** exposes TabPFN and TabICL to coding agents over stdio for fit, predict, what-if and explain. — [search: GitHub PR](https://github.com/VectorInstitute/tabctx/pull/84)
- TimeCopilot is an agentic forecasting framework in which an LLM orchestrates TSFMs. — [search: arXiv 2509.00616](https://arxiv.org/pdf/2509.00616)

### Inferences
- **Features in competitors that TabTune (public) appears to lack**, assuming TabTune's public feature list as given:
  - an embeddings API (tabpfn-extensions);
  - native many-class wrapping beyond the checkpoint limit;
  - unsupervised density, synthetic generation and outlier detection with tabular FMs;
  - image and text column featurisers (tabpfn-ext image, PyTorch Frame text encoders, TabPFN-3 Plus);
  - FM-portfolio AutoML presets with licence-filtered portfolios (AutoGluon's `extreme` vs `noncommercial`), which maps directly onto TabTune's licence-aware registry;
  - group- and time-aware validation (AutoGluon's `validation_structure`);
  - BO surrogate and HPO with FMs;
  - CRT p-values and feature selection;
  - TS anomaly detection, TS conformal intervals for pretrained models (Darts) and backtesting;
  - MCP / agent tool servers;
  - MLflow pyfunc wrappers and KV-cache-aware serving.
- TabTune's ensembling, distillation, causal and conformal features are differentiators. AutoGluon distils and ensembles too, but **causal inference** and **licence-aware registry** features are rarer. The internal TabAttribute competes with tabpfn-ext interpretability.
- TS zoos: Darts already exposes TimesFM-3 and TiRex-2. AutoGluon exposes Chronos-2 fine-tuning and Toto-2. TabTune's 9 TSFMs should be checked against TimesFM-3, TiRex-2, Toto-2, Moirai-2.0, TimeGPT-2.1 (API) and Sundial.

### Gaps
- **carte-ai, pytabkit (RealMLP/TabM), GluonTS, statsforecast and neuralforecast**: no feature-level sources were retrieved in this session. Their current features are unverified.
- Whether sktime 1.0 really shipped MCP and a pretraining API (social-media source only).
- TabArena code as a library (beyond benchmarking): the details of its model wrappers were not reviewed.
- Mitra's status in AutoGluon 1.6. It seems to have been replaced in `extreme` by Nori and TabICLv2, but this is not confirmed.

## Q3. Deployment and MLOps trends for tabular and TS FMs

### Takeaway
Serving ICL tabular FMs is moving towards **KV-cache "fit once, predict many"**, **distillation into MLP or tree engines for CPU serving**, and **MLflow / Databricks model-serving wrappers**. I found no evidence of mainstream ONNX or TensorRT export for TabPFN-class models. Hosted APIs with VPC tiers (TabPFN-3 Plus and Thinking, TimeGPT-2) gate the strongest variants.

### Cited Findings
- KV cache: `fit_with_cache`, about 7 GB per estimator at 1M rows. The cache is invalidated when data or config changes, and the API supports it only for TabPFN-3+. — [search: Prior Labs KV cache](https://docs.priorlabs.ai/capabilities/kv-cache)
- The TabPFN-as-MLP/Tree engine is recommended for large data and CPU inference. — [search: best practices](https://docs.priorlabs.ai/best-practices)
- MLflow PythonModel → Unity Catalog → Mosaic AI Model Serving cookbook. — [search: cookbook](https://docs.priorlabs.ai/cookbook)
- TabPFN-3 Plus and Thinking are only available via the hosted API, enterprise deployments or VPC. — [search: Prior Labs models](https://docs.priorlabs.ai/models)
- AutoGluon 1.6 cannot load models trained with older versions, which shows how fragile versioned FM artefacts are. — [opened](https://github.com/autogluon/autogluon/releases)
- "Assessing the Operational Viability of Foundation Models for Time Series Forecasting" (2026) studies TSFM deployment. — [search: arXiv 2605.24381](https://arxiv.org/pdf/2605.24381) (content not read)

### Inferences
- TabTune's existing distillation (and the internal PAD/BCF) is the natural basis for a "deploy" path: distil to an MLP or GBDT, then export via ONNX (GBDTs and MLPs export easily even though the transformer does not). A `tabtune export --onnx` plus MLflow pyfunc flavour would fill a clear market gap.
- Licence-aware registry + model cards: auto-generating model cards that include licence, training-data contamination lists and benchmark scores would be distinctive.
- Monitoring and drift for ICL models is underserved. Because the context set *is* the model, context-refresh policies and drift detection on the context are new MLOps primitives.

### Gaps
- No sources found on ONNX or TensorRT export of TabPFN, TabICL or LimiX, on distributed multi-GPU inference for tabular FMs, or on drift monitoring specific to tabular FMs. These should be treated as open opportunities, not confirmed absences.
- TSFM serving (e.g., Chronos-2 on SageMaker or Bedrock) was not researched.

## Q4. Emerging directions: agents, FM-based AutoML, relational and enterprise data, multimodal tables

### Takeaway
Four trends have converged. LLM agents call tabular and TS FMs as tools via MCP. AutoML has become "FM portfolios meta-learned on TabArena." Relational FMs (KumoRFM-2, TabPFN-Rel, RelArena) aim at enterprise multi-table data, though their benchmark claims are contested. Tables are becoming multimodal through native text (TabPFN-3 Plus) and image featurisers (DINOv3 in tabpfn-ext).

### Cited Findings
- MCP servers for TabPFN (official) and for TabPFN plus TabICL (tabctx-mcp). — [search: HN](https://news.ycombinator.com/item?id=46901522); [search: tabctx PR](https://github.com/VectorInstitute/tabctx/pull/84)
- TimeCopilot: LLM agent plus TSFM ensemble. — [search](https://arxiv.org/pdf/2509.00616)
- The Kapso agent claims #1 on RelBench, which a RelBench author disputes. A supervised "Rel-Agent" heads the RelBench v2 binary-classification table on an aggregator. — [search: Kapso issue](https://github.com/Leeroo-AI/kapso/issues/121); [search: sota2](https://www.sota2.com/research/sota/binary-classification-on-relbench-v2)
- AutoGluon's extreme preset is an FM portfolio meta-learned on TabArena. — [search: AG 1.5 docs](https://auto.gluon.ai/stable/whats_new/v1.5.0.html); [opened: releases](https://github.com/autogluon/autogluon/releases)
- KumoRFM-2 (41 relational tasks), TabPFN-Rel and RelArena-α, and parameter-free encoders for RDB FMs. — [search: KumoRFM-2](https://arxiv.org/html/2604.12596v1); [search: RelArena-α](https://arxiv.org/pdf/2608.16319); [search: arXiv 2607.05476](https://arxiv.org/pdf/2607.05476)
- Tabular FMs repurposed as graph FMs. — [search: arXiv 2508.20906](https://arxiv.org/pdf/2508.20906)
- Multimodal: TabPFN-3 Plus (native text), the tabpfn-ext image module (DINOv3) and PyTorch Frame text and image columns. — [search](https://docs.priorlabs.ai/changelog/tabpfn-3); [opened](https://github.com/PriorLabs/tabpfn-extensions); [opened](https://github.com/pyg-team/pytorch-frame)
- New tabular FMs to track: LimiX-2, TabICLv2, Mitra-v2, Nori, TabDPT-Turbo, TabPFN-2.6/3/3.5. — [search: TabICLv2](https://arxiv.org/pdf/2602.11139); [search: Mitra-v2](https://arxiv.org/pdf/2609.04540); [opened: AG releases](https://github.com/autogluon/autogluon/releases)

### Inferences
- An MCP server over TabTune's whole registry (fit, predict, explain, calibrate, leaderboard), with licence filters, would generalise the single-vendor TabPFN MCP. It would also let TabTune ride the coding-agent wave.
- A "TabTune AutoML" preset could take the TabArena meta-learned portfolio, filter it by the licence registry, and combine it with TabTune's ensembling. This would match AutoGluon's `extreme` / `noncommercial` split.
- A relational adapter (RelBench loader, plus flattening or feature propagation into TabPFN-Rel style) is a medium-term opportunity. RelArena-α would be the benchmark of record.

### Gaps
- KumoRFM's licence and open-weight status were not confirmed.
- Whether LimiX-2, TabICLv2, Mitra-v2 and Nori weights are open, and under which licences, needs checking for the registry.

## Top ~10 concrete feature gaps for TabTune (synthesis; prioritised)

### Takeaway
These are the highest-leverage additions, ranked by competitive pressure (what AutoGluon, tabpfn-ext and Darts ship) and fit with TabTune's existing strengths.

### Cited Findings
1. **Official benchmark runners**: `tabtune bench tabarena|beyondarena|tabred|talent274|openml-cc18|ctr23` and `tabtune bench fev|gift-eval|boom|tsb-ad`. TabTune models should also be exposed as TabArena methods so results can be submitted by PR. — [opened: tabarena](https://github.com/autogluon/tabarena), [opened: tabred](https://github.com/yandex-research/tabred), [opened: fev](https://github.com/autogluon/fev), [opened: TSB-AD](https://github.com/TheDatumOrg/TSB-AD)
2. **Licence-filtered FM AutoML presets**: `commercial` and `noncommercial` portfolios meta-learned on TabArena, as AutoGluon 1.6 does. — [opened](https://github.com/autogluon/autogluon/releases)
3. **Embeddings API** across all tabular FMs, for downstream retrieval, clustering and feature reuse. — [opened: tabpfn-ext](https://github.com/PriorLabs/tabpfn-extensions)
4. **Many-class and large-data wrappers**: an ECOC-style many-class wrapper for models below 160 classes, plus context subsampling, retrieval and chunking for more than 1M rows. — [opened: tabpfn-ext](https://github.com/PriorLabs/tabpfn-extensions); [search: TabPFN-3 limits](https://docs.priorlabs.ai/changelog/tabpfn-3)
5. **Multimodal columns**: text-encoder and image-encoder featurisers (DINOv3, HF, OpenAI), as in PyTorch Frame and tabpfn-ext image. — [opened: pytorch-frame](https://github.com/pyg-team/pytorch-frame)
6. **Unsupervised and generative module**: FM-based outlier detection, synthetic data and augmentation (TabEBM), and density estimation. — [opened: tabpfn-ext](https://github.com/PriorLabs/tabpfn-extensions)
7. **Non-IID validation**: group- and time-aware CV in fit and leaderboard (AutoGluon's `validation_structure`, BeyondArena, TabReD sliding windows). — [opened: AG](https://github.com/autogluon/autogluon/releases); [opened: tabred](https://github.com/yandex-research/tabred)
8. **TS breadth**: conformal intervals for pretrained TSFMs, backtesting, a TS anomaly-detection module, and newest-model coverage (TimesFM-3, TiRex-2, Toto-2, Chronos-2 LoRA, Moirai-2, TimeGPT-2.1 via API). — [opened: darts](https://github.com/unit8co/darts); [opened: AG](https://github.com/autogluon/autogluon/releases)
9. **Deployment path**: KV-cache fit modes, an MLflow pyfunc flavour, distil-to-MLP/GBDT plus ONNX export, and auto model cards from the licence registry. — [search: Prior Labs KV cache](https://docs.priorlabs.ai/capabilities/kv-cache); [search: cookbook](https://docs.priorlabs.ai/cookbook)
10. **Agent interface**: an MCP server exposing the registry (fit, predict, explain, calibrate, compare) to LLM agents. — [search: TabPFN MCP](https://news.ycombinator.com/item?id=46901522); [search: tabctx-mcp](https://github.com/VectorInstitute/tabctx/pull/84)
11. (Stretch) **Relational and multi-table support** via RelBench loaders and a TabPFN-Rel style adapter. — [search](https://arxiv.org/pdf/2608.16319)
12. (Stretch) **FM-as-surrogate HPO / BO and CRT feature-significance tests**. — [opened: tabpfn-ext](https://github.com/PriorLabs/tabpfn-extensions)

### Inferences
- Items 1, 2, 7 and 9 build directly on TabTune's existing leaderboard, licence registry and distillation, so they are the cheapest way to differentiate. Items 3–6 close parity gaps with tabpfn-extensions, but generalised across ~21 models rather than one vendor, which is TabTune's core value proposition.
- The internal TabAttribute and PAD/BCF should be positioned against tabpfn-ext interpretability and the TabPFN-as-MLP/Tree engine respectively.

### Gaps
- I did not inspect TabTune's own code here, so some "gaps" may already be partly covered. They should be checked against the TabTune repo.
- Items marked [search] above need their pages opened before final publication, especially the TabPFN-3 row and column limits, licence terms and leaderboard leaders.
