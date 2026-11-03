# Trustworthy-ML and Advanced Analytics on Tabular Foundation Models (2024 – Nov 2026): Gap Survey for TabTune

**Method note (read first).** I ran 15 tool calls (13 web searches, 4 fetch attempts). **arxiv.org would not resolve in this cloud session** (`getaddrinfo ENOTFOUND`), so no arXiv page was opened directly; it has to be allowed in the session's network settings before the arXiv claims can be checked. The only primary page I fully opened was the tabpfn-extensions GitHub README. Every other claim below comes from **search-result snippets or summaries of the cited URL**. Those are marked [snippet]. Claims marked [unverified] come from my background knowledge and have no source opened this session; they sit in Gaps sections. Read reported numbers as author self-reports.

TabTune baseline assumed (from brief). Public and internal both have conformal/recalibration, shift-aware splits, a causal suite (DML and meta-learners with FMs as nuisance models), and a registry. Only internal has TabAttribute (SHAP/LIME, support-set attribution, IG/EG, kNN-Shapley, TracIn, Datamodels, explanation evaluation). Public has no explainability.

---

## Q1. Interpretability beyond SHAP/LIME (Shapley interactions, remove-and-recontextualize, data valuation, GAM-style FMs, counterfactuals)

### Takeaway
The clearest mature gap is **Shapley interactions via shapiq**, including its TabPFN-native "remove-and-recontextualize" game. Open code exists and Prior Labs officially supports it. **GAMformer** and in-context kernel-regression FMs add glass-box TFMs. Public TabTune should get a thin shapiq-based explainability module. Internal TabAttribute should add interaction indices and the recontextualization game as an alternative to imputation.

### Cited Findings
- shapiq (Muschalik et al., NeurIPS 2024 Datasets & Benchmarks) is a library for Shapley interactions (k-SII and others). It frames a coalition's worth as the model's prediction with only those features, with absent features neutralised by marginal or conditional imputation ("remove-and-re-evaluate") — [shapiq NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/eb3a9313405e2d4175a5a3cfcd49999b-Paper-Datasets_and_Benchmarks_Track.pdf) [snippet]; [shapiq docs](https://shapiq.readthedocs.io/en/latest/index.html) [snippet]
- shapiq ships a TabPFN explainer that uses **remove-and-recontextualize**: TabPFN is re-fit (re-contexted) on each feature coalition instead of imputing. The docs credit Rundel et al. (2024) — [Prior Labs interpretability docs](https://docs.priorlabs.ai/capabilities/interpretability) [snippet]; [tabpfn-extensions interpretability README](https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/interpretability/README.md) [snippet]
- Prior Labs exposes two adapters. `get_tabpfn_imputation_explainer` is the recommended default. `get_tabpfn_explainer` is the remove-and-recontextualize option, and it is much slower because the KV cache cannot be reused across coalitions. `index="k-SII", max_order=2` gives pairwise interactions — [Prior Labs interpretability docs](https://docs.priorlabs.ai/capabilities/interpretability) [snippet]
- shapiq v1.3.0 (June 2025) fixed a bug that left TabPFN fitted on a feature subset after explanation. This is a practical integration hazard for TabTune adapters — [Socket.dev shapiq changelog](https://socket.dev/pypi/package/shapiq) [snippet]
- The tabpfn-extensions repo (Apache-2.0) contains `interpretability` (SHAP values plus feature selection), `pval_crt` (statistical feature-relevance p-values via the conditional randomization test), `embedding`, `unsupervised`, `tabebm`, `many_class`, `image` and `bayesian_optimization` — [tabpfn-extensions GitHub](https://github.com/PriorLabs/tabpfn-extensions) **(opened)**
- **GAMformer** (Mueller, Siems, Nori, Salinas, Zela, Caruana, Hutter; NeurIPS 2024 workshop, arXiv 2410.04560) estimates GAM shape functions in context in one forward pass. It is trained only on synthetic data, is reported on par with leading GAMs on classification, and gives interpretable shape functions. Source code is linked from the NeurIPS page — [NeurIPS virtual page](https://neurips.cc/virtual/2024/99159) [snippet]; [alphaxiv 2410.04560](https://www.alphaxiv.org/abs/2410.04560) [snippet]
- "Interpretable Tabular Foundation Models via In-Context Kernel Regression" (arXiv 2602.02162, Feb 2026) is another glass-box TFM direction — [arXiv 2602.02162](https://arxiv.org/html/2602.02162v1) [snippet, title only]
- `pval_crt` gives **hypothesis-test feature relevance** (p-values). This is distinct from attribution and absent from both TabTune versions — [tabpfn-extensions GitHub](https://github.com/PriorLabs/tabpfn-extensions) **(opened)**

### Inferences
- **Public TabTune:** add `tabtune.explain` as a thin wrapper over shapiq. It should cover the imputation game, the recontextualize game and k-SII interactions, plus CRT p-values. This closes the "no explainability" gap with code that already exists and has permissive licences (shapiq is MIT per its repo [unverified]; tabpfn-extensions is Apache-2.0).
- **Internal TabAttribute:** it already covers SHAP and support-set attribution. The new value is (a) **Shapley interaction indices** (k-SII, FSII, STII), (b) the **recontextualization value function** as an option in its faithfulness benchmark (imputation vs. re-context disagreement is a research-grade evaluation axis), and (c) model-agnostic adapters for TabICL, TabDPT and others beyond TabPFN.
- **GAM-style FMs** fit the registry as a "glass-box" capability flag rather than as an explainer.

### Gaps
- Rundel et al. 2024 ("Interpretable Machine Learning for TabPFN", arXiv 2403.10923) [unverified id]: I could not open it to confirm its exact contents (ICE/PD, Shapley via re-context, LOO data valuation).
- **In-context data valuation** specific to TFMs: my search found no dedicated 2024–2026 paper. Internal TabAttribute's LOO, kNN-Shapley and Datamodels are probably ahead of the published literature here.
- **Counterfactual explanations for TFMs**: no source found. Treat this as an open niche; generic DiCE-style generation over an FM predictor is feasible but not FM-specific.
- **Concept/attention-based explanations** of TFMs: no source found in this budget.

---

## Q2. Uncertainty and calibration (FM-native distributions, conformal for ICL, shift, abstention, OOD)

### Takeaway
TabTune already has conformal prediction and recalibration. The remaining gaps are (a) exposing **FM-native predictive distributions** (TabPFN regression outputs a full binned/Riemann distribution), (b) **selective prediction / abstention** and (c) **OOD detection** using FM likelihoods or embeddings. I found little dedicated 2024–2026 literature on (b) and (c) for TFMs. That makes them low-risk engineering modules more than paper reproductions.

### Cited Findings
- **Drift-Resilient TabPFN** (Schnurr, Helli, Hollmann, Müller, Hutter; NeurIPS 2024) pre-trains on evolving SCMs to anticipate temporal drift. It reports better OOD accuracy (0.744 vs 0.688 for TabPFN) and better calibration. The code is on GitHub — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/b2e2774c8e76afe191b5bf518f5cb727-Paper-Conference.pdf) [snippet]; [GitHub automl/Drift-Resilient_TabPFN](https://github.com/automl/Drift-Resilient_TabPFN) [snippet]
- Gains are larger on synthetic benchmarks than on real data, which the authors attribute to the stronger shifts in synthetic data — [alphaxiv 2411.10634](https://www.alphaxiv.org/abs/2411.10634) [snippet]
- "Realistic Evaluation of TabPFN v2 in Open Environments" (arXiv 2505.16226) evaluates TabPFN v2 under open-environment shifts — [arXiv 2505.16226](https://arxiv.org/pdf/2505.16226) [snippet, title only]
- "Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction" (arXiv 2510.02476) uses TFM uncertainty for model selection and reports good TabPFN performance on OOD targets with few-shot context — [arXiv 2510.02476](https://arxiv.org/pdf/2510.02476) [snippet]
- The TabPFN Nature paper (Hollmann et al., 2025) describes density estimation for numerical features. That supports likelihood-based OOD scoring — [Nature s41586-024-08328-6](https://www.nature.com/articles/s41586-024-08328-6) [snippet]

### Inferences
- **Both versions:** a `predictive_distribution` API exposing quantiles, CRPS, PIT histograms and NLL for FM regressors. Pair it with the existing conformal module, e.g. conformalised quantiles from FM-native quantiles (CQR-style).
- **Both versions:** a **selective prediction** module with risk–coverage curves, AURC and abstention thresholds. Confidence scores can come from FM entropy, ensemble disagreement or conformal set size. This is cheap and builds on the existing calibration code.
- **Both versions:** **OOD detection** that scores inputs by FM density/likelihood (TabPFN unsupervised), embedding kNN distance (`embedding` module) or FoMo-0D (see Q4).
- Drift-Resilient TabPFN belongs in the registry as a shift-aware model, evaluated with TabTune's existing shift-aware splits.

### Gaps
- No source was found for **conformal prediction specialised to ICL / TFMs** (e.g. exchangeability when the calibration set is also the context). This is worth a targeted search; the issue matters for TabTune's existing conformal module.
- No 2024–2026 paper was found on **selective prediction or abstention with TFMs** specifically.

---

## Q3. Causal foundation models (Do-PFN, CausalPFN, CausalFM, others; causal discovery)

### Takeaway
At least three amortized causal-effect FMs exist, plus newer 2026 work: Do-PFN, CausalPFN (NeurIPS 2025) and CausalFM, then a partial-graph extension and a joint structure-and-outcome model. TabTune uses FMs only as *nuisance models* inside DML and meta-learners. Adding **end-to-end causal PFNs as estimators**, benchmarked head-to-head against the DML stack, is a high-value differentiator. Cross-paper benchmark results conflict.

### Cited Findings
- **Do-PFN** (Robertson et al., 2025, arXiv 2506.06039) is pre-trained on SCM data to predict interventional outcomes and CATE from observational data. The authors claim an optimal approximation of the conditional interventional distribution with respect to the prior — [arXiv 2506.06039](https://arxiv.org/html/2506.06039) [snippet]
- **CausalPFN** (Balazadeh et al., NeurIPS 2025) amortises back-door effect estimation under strong ignorability and is reported to often beat specialised estimators — [NeurIPS 2025 paper](https://papers.nips.cc/paper_files/paper/2025/file/e3d3db07c1bfb63e1d0b998996de1d12-Paper-Conference.pdf) [snippet]
- **CausalFM** (Ma et al., 2025/26; "Foundation Models for Causal Inference via Prior-Data Fitted Networks") pre-trains on millions of SCMs and covers back-door, **IV and front-door** settings — [ResearchGate](https://www.researchgate.net/publication/392629673_Foundation_Models_for_Causal_Inference_via_Prior-Data_Fitted_Networks) [snippet]
- **"Use What You Know: Causal Foundation Models with Partial Graphs"** (arXiv 2602.14972, 2026) conditions causal FMs on partial causal-graph knowledge — [arXiv 2602.14972](https://arxiv.org/html/2602.14972v2) [snippet, title only]
- **"A Causal Foundation Model for Structure and Outcome Prediction"** (arXiv 2606.26467, 2026) jointly predicts causal structure (discovery) and outcomes — [arXiv 2606.26467](https://arxiv.org/pdf/2606.26467) [snippet, title only]
- A Sept 2026 paper titled **"Causal Foundation Models"** (arXiv 2609.03003) appears to compare these models — [arXiv 2609.03003](https://arxiv.org/html/2609.03003v1) [snippet; fetch blocked]
- **Benchmark conflict.** One source reports CausalPFN best overall on RealCause. Another reports that on Lalonde-CPS, CausalPFN recovers the ATE closely (relative error 0.17) while Do-PFN and CausalFM do not. Results vary by benchmark — [arXiv 2609.03003](https://arxiv.org/html/2609.03003v1) [snippet]; [NeurIPS 2025 CausalPFN](https://papers.nips.cc/paper_files/paper/2025/file/e3d3db07c1bfb63e1d0b998996de1d12-Paper-Conference.pdf) [snippet]

### Inferences
- **Both versions:** a `tabtune.causal.amortized` sub-module that wraps Do-PFN, CausalPFN and CausalFM as estimators behind the existing causal API. Benchmark them on the same RealCause, IHDP, ACIC and Lalonde data as the DML and meta-learner stack. A neutral benchmark harness is the strongest contribution, because the published comparisons conflict.
- Registry flags are needed for identification assumptions: back-door only (CausalPFN) vs. IV/front-door (CausalFM) vs. graph-conditioned (partial-graph FM).
- Causal discovery with FMs (2606.26467) is early and fits the internal version first.

### Gaps
- I could not confirm code repositories or licences for Do-PFN, CausalPFN or CausalFM. I believe all three released code [unverified].
- CInA (Arik & Pfister?, "causal inference via attention", 2023/2024) [unverified] did not come up in search. Its status is unknown.
- The exact authors and contents of arXiv 2609.03003 are unverified because the fetch was blocked.

---

## Q4. Generative tabular: synthesis, imputation, density, anomaly/outlier detection

### Takeaway
Mature, openly coded options exist for all four tasks: TabPFN `unsupervised` (density, synthesis, outliers), TabPFGen/TabEBM (energy-based augmentation), TabImpute (zero-shot imputation) and FoMo-0D (zero-shot outlier detection, TMLR 2025). TabTune has none of these. A "generative and unsupervised" module family is a large, well-supported addition. Independent replication is mixed, especially for imputation.

### Cited Findings
- The TabPFN Nature paper describes **density estimation** for numerical features and **sample synthesis** by sampling features autoregressively — [Nature](https://www.nature.com/articles/s41586-024-08328-6) [snippet]
- tabpfn-extensions `unsupervised` generates synthetic data and detects outliers, and `tabebm` provides TabPFN-based energy-based augmentation (Apache-2.0) — [tabpfn-extensions GitHub](https://github.com/PriorLabs/tabpfn-extensions) **(opened)**
- **TabPFGen** (arXiv 2406.05216, 2024) recasts a pre-trained TabPFN as an energy-based model with class-conditional energy and SGLD sampling, with no extra training. It is evaluated on augmentation, class balancing and imputation, and beats mean imputation on RMSE (self-reported) — [arXiv 2406.05216](https://arxiv.org/abs/2406.05216) [snippet]; [author PDF](http://www.cs.utoronto.ca/~guangweiyu/pdfs/tabpfgen.pdf) [snippet]
- **TabImpute** (Feitelberg, Saha, Choi, Ahmad, Agarwal, Dwivedi; EurIPS 2025 workshop, arXiv 2510.02625) is a zero-shot imputation transformer built on TabPFN. It reports about 100x speedup over the earlier tabpfn-extensions iterative imputation, a missingness-aware synthetic prior and the **MissBench** benchmark (42 OpenML datasets). TabImpute+ ensembles it with TabPFN — [arXiv 2510.02625](https://arxiv.org/html/2510.02625v4) [snippet]; [OpenReview PDF](https://openreview.net/pdf?id=pqgAeeCCT8) [snippet]; [chatpaper summary](https://chatpaper.com/paper/195960) [snippet]
- **Counter-evidence:** a 2026 benchmark-selection paper found TabImpute underperformed a Gaussian conditional imputation baseline on most of its datasets, with near-zero results on a sparse one — [arXiv 2605.02209](https://arxiv.org/pdf/2605.02209) [snippet]
- **FoMo-0D** (Shen, Wen, Akoglu; TMLR 2025, arXiv 2409.05672) is a PFN for **zero-shot tabular outlier detection** that takes unlabeled data as context. It was tested on 57 datasets against 26 baselines, with results not significantly different from the second-best method, at about 7.7 ms per sample. Code and checkpoints are on GitHub — [arXiv 2409.05672](https://arxiv.org/abs/2409.05672) [snippet]; [GitHub FoMo-0D](https://github.com/A-Chicharito-S/FoMo-0D) [snippet]
- Follow-up: "From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection" (arXiv 2602.03018, 2026) — [arXiv 2602.03018](https://arxiv.org/abs/2602.03018) [snippet, title only]

### Inferences
- **Both versions:** `tabtune.unsupervised` with (a) outlier detection (FoMo-0D plus TabPFN unsupervised scores), (b) imputation (TabImpute plus the iterative TabPFN baseline, benchmarked on MissBench), (c) synthesis and augmentation (TabPFN unsupervised, TabEBM/TabPFGen) and (d) density/likelihood scoring. The density scores also feed the OOD detection in Q2.
- Synthetic data needs **fidelity, utility and privacy metrics** (e.g. TSTR, DCR, MIA). That links to Q6; Synth-MIA can serve as the audit harness.
- Imputation results conflict, so ship it with a benchmark against simple baselines rather than as a default.

### Gaps
- TabDiff (ICLR 2025 diffusion generator) and LLM-based generators (GReaT and others) did not appear in my results. They are not FM-native and may belong to another researcher's scope.
- "OutlierPFN" as a named method: no source found.

---

## Q5. Fairness and robustness

### Takeaway
Tabular-ICL fairness is a small but active 2025–2026 literature. Its main levers are **demonstration (context) selection** (open code: Fair-TabICL) and **fairness-aware pre-training** (FairTFM, 2026 preprint). Both are natural TabTune modules because ICL makes context curation the intervention point. I found little on adversarial robustness.

### Cited Findings
- **Kenfack et al., "Towards Fair In-Context Learning with Tabular Foundation Models"** (arXiv 2505.09503, 2025) is presented as the first study of fairness in tabular ICL. It evaluates TabPFN v2, TabICL and TabDPT with three pre-processing methods: correlation removal, group-balanced sample selection and **uncertainty-based demonstration selection**. The last consistently improves group fairness. Code is at github.com/patrikken/Fair-TabICL — [arXiv 2505.09503](https://arxiv.org/abs/2505.09503) [snippet]
- **"Causal Pre-training Under the Fairness Lens: An Empirical Study of TabPFN"** (arXiv 2601.17912, 2026) evaluates TabPFN and fine-tuned variants for fairness, including under covariate shift — [arXiv 2601.17912](https://www.arxiv.org/pdf/2601.17912) [snippet]
- **"Training Fair Tabular Foundation Models" (FairTFM)** (arXiv 2608.14211, 2026) moves fairness into pre-training, so that marking a sensitive column in the prompt yields fairer predictions in one forward pass. It is a preprint and its peer-review status is unconfirmed — [arXiv 2608.14211](https://arxiv.org/pdf/2608.14211) [snippet]; [Pith review](https://pith.science/paper/2608.14211) [snippet]
- A 2025/26 critical review of TFMs reports group disparities that the prior does not remove — [MDPI MAKE 8(8):244](https://www.mdpi.com/2504-4990/8/8/244) [snippet]
- "When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach" (arXiv 2605.19662) covers robustness to strategic manipulation — [arXiv 2605.19662](https://arxiv.org/html/2605.19662v1) [snippet, title only]
- Drift adaptation via Drift-Resilient TabPFN — see Q2.

### Inferences
- **Both versions:** `tabtune.fairness` with group metrics (DP, EO, EOdds, calibration-by-group), context-selection mitigations (group-balanced and uncertainty-based, following Fair-TabICL), correlation-removal pre-processing and threshold post-processing. Make it shift-aware by reusing TabTune's existing shift splits.
- The internal version could explain fairness gaps by applying support-set attribution to group disparity (which context rows drive the disparity). This combination does not appear in the literature I found.

### Gaps
- **Adversarial robustness** (evasion attacks on TFMs): no dedicated 2024–2026 paper surfaced.
- Fair-TabICL's licence is unverified.

---

## Q6. Privacy and governance (MIA, DP, unlearning, leakage, audit)

### Takeaway
2026 preprints show that **TFMs leak context membership**. TabPFN is reported as more vulnerable, and attention-based attacks exist. DP defenses for TFMs are appearing (TabPATE, PrivTab; DP-TabICL earlier for LLM-ICL). **Unlearning for TFMs reduces to dropping context rows** (exact for pure ICL), but I found no paper formalising it. A privacy audit module (MIA harness plus DP-ICL plus certified context deletion) is novel and timely.

### Cited Findings
- **TabPATE** (arXiv 2606.31474, 2026) adapts membership inference to tabular ICL and finds that prediction behaviour reveals context membership. TabPFN is more vulnerable in the authors' experiment. It proposes a PATE-style DP defense built on TabPFN that needs no public data, reporting competitive utility at moderate ε and near-random attack success — [arXiv 2606.31474](https://arxiv.org/abs/2606.31474) [snippet]
- **AMIA, "Privacy Vulnerabilities of Attention Layers in Tabular Foundation Models and Protection of High-Risk Queries"** (arXiv 2606.26021, 2026) is an attention-based MIA tested on TabPFN, TabICL and TabDPT across 6 datasets. It finds TFMs as susceptible to standard MIAs as traditional models — [alphaxiv 2606.26021](https://www.alphaxiv.org/abs/2606.26021) [snippet]
- **PrivTab, "Efficient Provably Private Classification with a Tabular Foundation Model"** (arXiv 2610.10068, Oct 2026) uses ICL to turn sensitive rows into compact, provably private summaries. It notes that standard TFMs give no formal guarantee against context-record MIA — [arXiv 2610.10068](https://arxiv.org/html/2610.10068) [snippet]
- **"Probing Memorization of Tabular In-Context Learning"** (arXiv 2606.31208, 2026) — [arXiv 2606.31208](https://arxiv.org/pdf/2606.31208) [snippet, title only; fetch blocked]
- **DP-TabICL** (arXiv 2403.05681, 2024) privatises serialized demonstrations with local and global DP. It targets LLM-ICL, not TabPFN — [arXiv 2403.05681](https://arxiv.org/pdf/2403.05681) [snippet]
- **Synth-MIA** (arXiv 2509.18014) is a testbed for auditing privacy leakage in tabular data synthesis — [arXiv 2509.18014](https://arxiv.org/pdf/2509.18014) [snippet, title only]
- Unlearning: the closest precedent is LLM "In-Context Unlearning" (arXiv 2310.07579). A tabular federated-unlearning benchmark defines instance unlearning as removing rows. Neither covers TFMs — [arXiv 2310.07579](https://arxiv.org/pdf/2310.07579) [snippet]; [arXiv 2504.00921](https://arxiv.org/pdf/2504.00921) [snippet]
- A 2024 study warns that unlearning evaluations without example-specific (LiRA-style) thresholds overestimate unlearning — via [arXiv 2504.00921](https://arxiv.org/pdf/2504.00921) [snippet]

### Inferences
- **Both versions:** `tabtune.privacy` with (a) an MIA audit harness (LiRA/loss-threshold plus attention-based attacks) for context rows, (b) DP-ICL wrappers (TabPATE-style subsample-and-aggregate), (c) **certified context deletion**. For pure ICL, deleting a row and dropping cached KV gives exact unlearning. Fine-tuned TFMs (which TabTune supports) do not; there, unlearning needs re-fine-tuning or approximate methods, and the registry should track this distinction.
- **Governance:** extend the registry's licence/capability checks into auto-generated **model cards / audit reports** (calibration, fairness, MIA risk, shift performance, training-data leakage checks such as train/test row overlap and pretraining-prior overlap with benchmark datasets).
- **Internal:** support-set attribution scores (LOO, TracIn) double as **MIA risk indicators**. High-influence context rows are the most exposed, which fits AMIA's "high-risk query" framing.

### Gaps
- No paper found that formalises unlearning for TFMs. This is a plausible inference only.
- No paper found on DP fine-tuning (DP-SGD) of TFMs specifically.
- No paper found on leakage or contamination checks between TFM pre-training priors and benchmarks (relevant for real-data-pretrained TFMs like TabDPT).

---

## Ranked Top-10 Module Recommendations (synthesis)

Columns: rank; module; key papers; mechanism; maturity; open code; TabTune fit.

1. **Shapley interactions + recontextualization explainer** — shapiq (NeurIPS'24), Rundel et al. 2024. Exact or approximate k-SII over a re-context or imputation value function. *Mature*; code: shapiq, tabpfn-extensions. **Public** (first explainability) **and Internal** (interactions + value-function comparison in the faithfulness benchmark).
2. **Zero-shot outlier / anomaly detection** — FoMo-0D (TMLR'25), "From Zero to Hero" (2026), TabPFN unsupervised. A PFN takes unlabeled data as context and outputs outlier scores. *Mature*; code: GitHub FoMo-0D, tabpfn-extensions. **Both.**
3. **Amortized causal-effect FMs + neutral benchmark** — Do-PFN, CausalPFN (NeurIPS'25), CausalFM, partial-graph CFM (2026). SCM-prior-trained transformers output CATE/ATE in one pass. *Emerging (2025–26), conflicting benchmarks*; code believed available [unverified]. **Both** (extends the existing causal suite).
4. **Privacy audit + DP-ICL + certified context unlearning** — TabPATE, AMIA, PrivTab (2026), DP-TabICL. MIA harness, PATE-style DP aggregation and row-drop unlearning. *Emerging*; code unverified. **Both**, with an attribution-based risk scorer in **Internal**.
5. **Fairness audit + context-selection mitigation** — Kenfack et al. 2025 (Fair-TabICL), FairTFM 2026, TabPFN fairness study 2026. Group metrics plus uncertainty- or group-balanced demonstration selection. *Emerging, code available (Fair-TabICL)*. **Both.**
6. **Zero-shot imputation** — TabImpute (EurIPS-W'25) + MissBench, TabPFN iterative imputer. A missingness-prior-trained PFN fills cells in one pass. *Moderate; mixed independent results*. **Both** (ship with the benchmark).
7. **Synthetic data / augmentation with FM generators + fidelity/privacy metrics** — TabPFN unsupervised, TabPFGen/TabEBM, Synth-MIA. Autoregressive feature sampling or SGLD on an energy view of the classifier. *Moderate*; code: tabpfn-extensions. **Both.**
8. **Predictive-distribution and selective-prediction toolkit** — TabPFN Nature 2025 (native distributions), Drift-Resilient TabPFN (calibration under drift). Quantiles, CRPS, PIT, risk–coverage and abstention on top of the existing conformal module. *Engineering-mature; thin FM-specific literature*. **Both.**
9. **OOD / shift detection with FM density and embeddings** — TabPFN density, `embedding` extension, Drift-Resilient TabPFN, open-environment TabPFN v2 eval (2505.16226). Likelihood or embedding-distance scores plus drift-resilient models in the registry. *Moderate*. **Both.**
10. **Glass-box TFMs + statistical feature-relevance tests** — GAMformer (2024), in-context kernel regression TFM (2026), `pval_crt`. ICL-estimated shape functions; CRT p-values. *Early to moderate*; code: GAMformer link, tabpfn-extensions. **Public** (CRT, GAMformer as a registry model); **Internal** (compare against TabAttribute attributions).

Honourable mentions (no adequate source found, so marked speculative): counterfactual explanations for TFMs; TFM-specific conformal under context/calibration reuse; auto-generated model cards / audit reports; causal discovery FM (2606.26467); adversarial robustness evaluation.
