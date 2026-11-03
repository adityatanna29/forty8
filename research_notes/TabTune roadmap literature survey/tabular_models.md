# Tabular foundation models not yet in TabTune (2024 to Nov 2026): candidate catalogue and ranking

> **How these notes were made (read this first).** Every finding comes from WebSearch result extracts (17 calls, done 2026-11-03). WebFetch could not reach **arxiv.org, tabarena.ai or thetabularguy.substack.com** (DNS blocked in this cloud session), so no page was opened in full. Each URL below is one the search tool returned, along with the content it pulled from that URL. Numbers that vendors report about themselves are labelled as such. Where sources disagree, both versions are given. Anything I could not check is in the Gaps sections. Entries marked "(background knowledge, unverified)" come from my own knowledge and have no source from this session.

## Q1. Which single-table tabular FMs from 2024 to 2026 lead current leaderboards (TabArena, TALENT) and are missing from TabTune?

### Takeaway
The September 2026 wave of models has mostly been absorbed already. Of the 7 models TALENT added in its "Sep 2026" update, TabTune has 6. The one it lacks is **NVIDIA Kumo Tabular**: open weights under a commercial-use licence (OpenMDW-1.1), reported best on the TabArena accuracy-efficiency Pareto front, and a TabICL/TabPFN-style ICL design. That makes it the clearest new single-table addition. Beyond it, the strongest options are upgrades to models TabTune already has (**LimiX-2**, **TabDPT v1.2-Turbo/v1.3**) and **TabSTAR** for tables with text. The Oct 2026 research models (TAFFY, TabSwift) are too new to have confirmed weights.

### Cited Findings

**Leaderboard state (the snapshots conflict, so treat as provisional)**
- TabArena is a living benchmark of 51 curated datasets with 500 to 250,000 training rows. It ranks models by Elo, using ROC-AUC for binary, log-loss for multiclass and RMSE for regression. — [TabArena paper](https://arxiv.org/pdf/2506.16791)
- A Sep 2026 blog snapshot put Google TabFM first on TabArena at about 1800 Elo, with LG EXAONE Tabular (20.8M params) 18 Elo behind. An Oct 2026 post had **LimiX-2 leading at 1971**, with TabPFN-3.5, Causilo, TabFM and Mitra-v2 statistically close and the top-5 order "not settled". Prior Labs claims TabPFN-3.5 is first on both TabArena and BeyondArena (vendor claim). These are secondary blog snapshots from different dates. — [thetabularguy substack](https://thetabularguy.substack.com/p/how-to-pick-a-tabular-foundation); [Qlik blog](https://www.qlik.com/blog/results-for-tabular-machine-learning); [metavert overview](https://metavert.io/tabular-foundation-models)
- One 2026 review finds TFMs beat tuned XGBoost, CatBoost and LightGBM except on temporal, grouped and large-scale data. The companion **BeyondArena** benchmark tests temporal and group shift and larger tables. — [Qlik blog](https://www.qlik.com/blog/results-for-tabular-machine-learning); [Beyond IID paper, arXiv 2606.30410](https://arxiv.org/pdf/2606.30410)
- A Sep 2026 secondary source says Mitra-v2 (77M params) is "not yet on the official board". — [thetabularguy substack](https://thetabularguy.substack.com/p/how-to-pick-a-tabular-foundation)
- TALENT's README (Oct 2026) lists the Sep 2026 TFMs it added: **TabPFN-3.5, LimiX-2, NVIDIA Kumo Tabular, Mitra-v2, Causilo, Xiaomi-TabLDM, EXAONE-Tabular**. PR #115 also gives every TFM one consistent interface and fixes wrappers that "silently ran a different model or configuration than intended". The search could not confirm the PR is merged. — [TALENT PR #115](https://github.com/LAMDA-Tabular/TALENT/pull/115); [TALENT repo](https://github.com/LAMDA-Tabular/TALENT)

**Candidate: NVIDIA Kumo Tabular (new, not in TabTune)**
- Org: NVIDIA, reportedly built on technology from Kumo AI, which NVIDIA acquired in June 2026. Released 2026-09-29 per the announcements; the SDK table says 2026-09-28. — [HF blog (NVIDIA)](https://huggingface.co/blog/nvidia/kumo-tabular); [unite.ai](https://www.unite.ai/nvidia-releases-open-kumo-tabular-model-for-tabular-prediction/); [nvidia.github.io SDK model table](https://nvidia.github.io/structured-data-models/api/models.html); [Digital Today](https://www.digitaltoday.co.kr/en/view/110475/nvidia-unveils-kumo-tabular-open-source-tabular-data-foundation-model)
- Architecture: a single forward pass of in-context learning "inspired by TabICL and TabPFN", pretrained only on synthetic tables sampled from structural causal models. Tasks: classification and regression, numerical and categorical columns only. One summary notes weaker accuracy out of distribution. — [HF blog](https://huggingface.co/blog/nvidia/kumo-tabular); [unite.ai](https://www.unite.ai/nvidia-releases-open-kumo-tabular-model-for-tabular-prediction/)
- Size: the announcements give 3 sizes from 28M to 215M params. **This conflicts with** the SDK table, which lists 27.44M to 61.43M for the classification variant. — [HF blog](https://huggingface.co/blog/nvidia/kumo-tabular); [SDK table](https://nvidia.github.io/structured-data-models/api/models.html)
- Licence: **weights OpenMDW-1.1 (commercial use allowed), code Apache-2.0.** Inference runs through the open-source `structured-data-models` library, which downloads weights on first use. — [SDK table](https://nvidia.github.io/structured-data-models/api/models.html); [HF blog](https://huggingface.co/blog/nvidia/kumo-tabular); [TALENT PR #115](https://github.com/LAMDA-Tabular/TALENT/pull/115)
- Benchmark: NVIDIA reports "a new state-of-the-art on the accuracy-efficiency Pareto front" against TabArena (vendor claim). — [HF blog](https://huggingface.co/blog/nvidia/kumo-tabular)
- A Hugging Face repo "Zeroxdesignart/Kumo-Tabular" is a third-party mirror. Do not use it as the source. — [HF mirror](https://huggingface.co/Zeroxdesignart/Kumo-Tabular)

**Upgrade: LimiX-2 (TabTune has LimiX)**
- Stable AI and Tsinghua. Open-sourced 16 Sep 2026 as `LimiX-2.ckpt` plus inference code on GitHub and HF (`stable-ai/LimiX-2`). 400M params. One checkpoint does classification, regression and missing-value imputation, and the authors also claim causal-skeleton recovery. The architecture is "Contextual Mechanism Networks", which learn a joint structure over features and labels rather than predicting a fixed target. — [LimiX GitHub](https://github.com/limix-ldm-ai/LimiX); [Pandaily](https://pandaily.com/stable-ai-limix-2-structured-data-foundation-model); [alphaXiv 2609.17488](https://www.alphaxiv.org/abs/2609.17488); [aicybr](https://aicybr.com/blog/limix-2-400m-tabular-foundation-model)
- Vendor-reported Elo: **TabArena 1935, BCCO 1432, TALENT 1506.** — [ai-tldr](https://ai-tldr.dev/releases/stable-ai-limix-2/); [Pandaily](https://pandaily.com/stable-ai-limix-2-structured-data-foundation-model)
- Licence: **StableAI LimiX Non-Commercial License v1.0, so no commercial use.** — [Pandaily](https://pandaily.com/stable-ai-limix-2-structured-data-foundation-model); [LimiX GitHub](https://github.com/limix-ldm-ai/LimiX)
- **LimiX-2M** is a separate 2M-param lightweight model from the LimiX-16M line (arXiv 2606.04485). — [LimiX-2M paper](https://arxiv.org/html/2606.04485); [Show HN](https://news.ycombinator.com/item?id=45909608)

**Upgrade: TabDPT (TabTune has TabDPT)**
- Layer 6 AI. PyPI lists **v1.2 "TabDPT-Turbo" (17 Jun 2026)**, described as the fastest model overall in the authors' experiments, and **v1.3 (8 Sep 2026)**. The original release included inference code, weights on HF, training code and pretraining data. Whether v1.2/v1.3 keep the v1.1 licence is not stated. — [PyPI tabdpt](https://pypi.org/project/tabdpt/); [PyPI 1.3.1](https://pypi.org/project/tabdpt/1.3.1/); [TabDPT-inference GitHub](https://github.com/layer6ai-labs/TabDPT-inference); [TabDPT NeurIPS 2025 paper](https://www.cs.toronto.edu/~mvolkovs/NeurIPS2025_TabDPT.pdf); [arXiv 2410.18164](https://arxiv.org/abs/2410.18164)

**Variant: Real-TabPFN / RealTabPFN-2.5 (Prior Labs)**
- Real-TabPFN (2025, Garg et al.) continues pretraining TabPFN on curated real-world datasets. RealTabPFN weights carry a non-commercial licence (secondary 2026 blog). A 2026 TabArena-based blog called RealTabPFN-2.5 the top TFM on that board, but that snapshot appears to predate TabPFN-3/3.5. — [mindfulmodeler "State of TFMs 2026"](https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models); [awesome-tabular-foundation-models](https://github.com/christophM/awesome-tabular-foundation-models); [HF Prior-Labs/tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5)
- Wikipedia: TabPFN-2.5+ weights are non-commercial. The `tabpfn` **package code was relicensed to Apache-2.0 in v9.0.0 (Sep 2026)**. One source says Prior Labs is now part of SAP (unverified). — [Wikipedia TabPFN](https://en.wikipedia.org/wiki/TabPFN); [metavert](https://metavert.io/tabular-foundation-models)
- TabPFN-3 (May 2026) handles up to 1M rows; TabPFN-3.5 followed in Sep 2026. — [TabPFN-3 tech report, arXiv 2605.13986](https://arxiv.org/pdf/2605.13986); [metavert](https://metavert.io/tabular-foundation-models)

**Older or lower-value single-table ICL models (checked)**
- **TabForestPFN** (2024, den Breejen et al.): an ICL transformer fine-tuned after pretraining on a forest-based synthetic generator with complex decision boundaries. Its OpenReview entry is a **withdrawn** ICLR 2025 submission. — [OpenReview](https://openreview.net/forum?id=pE0UM18TQh); [Representation-learning survey](https://arxiv.org/pdf/2504.16109)
- **APT** (adversarially pre-trained transformer): zero-shot meta-learning, pretrained adversarially on synthetic data. Little detail turned up. — [awesome-tabular-foundation-models](https://github.com/christophM/awesome-tabular-foundation-models)
- **TabFlex** (Zeng et al., listed as a NeurIPS 2024 workshop paper): linear attention scales tabular ICL to millions of samples, reported about 2x faster than TabPFN and 1.5x faster than XGBoost. — [awesome-tabular-foundation-models](https://github.com/christophM/awesome-tabular-foundation-models); [LAMDA Tabular-Survey](https://github.com/LAMDA-Tabular/Tabular-Survey)
- **MotherNet** (Microsoft, arXiv Dec 2023, research prototype in `microsoft/ticl`): a hypernetwork that emits a small child MLP in one forward pass. It inherits TabPFN's quadratic memory cost. No 2026 Microsoft tabular release was found. — [microsoft/ticl](https://github.com/microsoft/ticl); [MotherNet arXiv 2312.08598](https://arxiv.org/pdf/2312.08598)

**New 2026 research models (weights unconfirmed)**
- **TAFFY** (arXiv 2610.07559, Oct 2026): adds an in-context diversity prior to a task-conditioned looped transformer. Weights and code not confirmed. — [TAFFY abs](https://arxiv.org/abs/2610.07559); [TAFFY html](https://arxiv.org/html/2610.07559v1)
- **TabSwift** (arXiv 2606.07345): "An Efficient Tabular Foundation Model with Row-Wise Attention". Only the title was confirmed. — [TabSwift PDF](https://arxiv.org/pdf/2606.07345)
- **TabGift** (arXiv 2610.16990): an in-context diffusion model for *generating* tabular data, outside clf/reg scope. — [TabGift](https://arxiv.org/abs/2610.16990)
- TabTune's own OrionMoE-v1.0 paper is on arXiv as 2610.20418 (Oct 2026). — [OrionMoE arXiv](https://arxiv.org/abs/2610.20418v1)

**Methods relevant to TabTune's fine-tuning / PEFT layer (techniques, not models)**
- **TFM-Retouche** (arXiv 2605.06047): a lightweight input-space adapter for TFMs, a PEFT-style method TabTune could offer. — [TFM-Retouche](https://arxiv.org/pdf/2605.06047)
- **GEAR** (arXiv 2608.18849): two-stage distillation of TFMs (generative expansion plus real anchoring). — [GEAR](https://arxiv.org/pdf/2608.18849)
- **Tab-AICL** (arXiv 2603.27385): active in-context learning, which optimises the labelled context rather than the weights. — [Tab-AICL](https://arxiv.org/abs/2603.27385)
- **Chunked TabPFN** (arXiv 2509.00326): exact, training-free ICL for long contexts. — [Chunked TabPFN](https://arxiv.org/pdf/2509.00326)

### Inferences
- Since TALENT's Sep 2026 refresh lists 7 TFMs and TabTune already has 6 of them (LimiX-2 being an upgrade), **Kumo Tabular is the main gap on the leaderboards** and the only new top-tier model with commercial-use weights. Its TabICL-like design should fit TabTune's existing ICL fine-tuning paths (meta-learning FT, SFT, LoRA), but whether NVIDIA's library exposes training hooks is not verified.
- LimiX-2 likely sits at or near the top of TabArena, but its non-commercial licence means it is best added as a "research" checkpoint behind TabTune's existing LimiX adapter.
- The licence situation splits clearly: Prior Labs 2.5+ and LimiX-2 are non-commercial, while Kumo Tabular (OpenMDW), TabICLv2 (BSD-3) and Mitra-v2 are permissive. Kumo Tabular strengthens TabTune's commercially usable lineup.
- TabForestPFN, APT, TabFlex and MotherNet are superseded on accuracy. They matter mainly as architectural variety (linear attention, hypernetwork) and are low priority.

### Gaps
- The live TabArena leaderboard (tabarena.ai) could not be fetched; the snapshots conflict (TabFM first vs LimiX-2 first). The live board needs checking.
- Not verified in this session: **GAMformer, KumoRFM's single-table mode, TP-BERTa, Tabby, Mamba/linear-attention tabular ICL beyond TabFlex, TabM**. Background knowledge, unverified: TabM (Yandex, ICLR 2025) is a parameter-efficient MLP ensemble trained per dataset, i.e. a strong baseline rather than an FM.
- No 2026 tabular FM from **Meta, Alibaba, ByteDance, Huawei or Microsoft** turned up in search ([search summary sources](https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models), [MDPI critical review](https://www.mdpi.com/2504-4990/8/8/244)). That does not prove none exist.
- Not confirmed: Kumo Tabular's row/feature/class limits, whether the SDK supports fine-tuning, and the true parameter range.
- TAFFY and TabSwift: no confirmed weights, licence or benchmark numbers.

## Q2. Relational / multi-table foundation models: maturity and open weights

### Takeaway
Relational FMs have taken off in 2026 and now have an open benchmark (**RelArena-α**, from Stanford STAR). There are three practical open routes. (1) **Stanford's Relational Transformer** family (RT 22M, RT-PluRel 86M, RT-J) has open HF checkpoints. (2) **Prior Labs' TabPFN-Rel** flattens the database with deep feature synthesis and feeds it to TabPFN-3; this is cheap for TabTune because TabPFN v3 is already supported. (3) **NVIDIA KumoRelational** is available, but its licence status is inconsistent. KumoRFM and KumoRFM-2 remain closed (API only). The field is still research-grade.

### Cited Findings
- **RelArena-α / TabPFN-Rel / RPI** (arXiv 2608.16319, Stanford STAR with others) is an open, reproducible relational-learning benchmark. TabPFN-Rel is described as Prior Labs' "relational harness for TabPFN-3 and the top-ranked model submission": it "flattens the database with deep feature synthesis and predicts in context with TabPFN-3". — [STAR RelArena page](https://star-project.stanford.edu/relarena/); [HF paper page](https://huggingface.co/papers/2608.16319); [arXiv PDF](https://arxiv.org/pdf/2608.16319)
- **Conflict:** the same STAR material also calls **RT-PluRel** "the #1 system" on its leaderboard. The two may be on different tracks; this is unresolved. — [STAR RelArena page](https://star-project.stanford.edu/relarena/)
- TabPFN-Rel code: Prior Labs GitHub has a public `tabpfn-rel` repo. Its extras are `tabpfn-rel-api` (featurisation runs locally, fitting and prediction on the hosted TabPFN API) and `tabpfn-rel-local` (local GPU). Weights are TabPFN-3's (non-commercial per Wikipedia); the `tabpfn` package itself is Apache-2.0 from v9.0.0. — [Prior Labs GitHub](https://github.com/priorlabs); [Prior Labs docs cookbook](https://docs.priorlabs.ai/cookbook/relational_predictions_tabpfn_rel); [Wikipedia TabPFN](https://en.wikipedia.org/wiki/TabPFN)
- **Relational Transformer (RT)** (Stanford, ICLR 2026, arXiv 2510.06377): 12 layers, d=256, 8 heads, about 22M params. It transfers across relational DBs without fine-tuning and reaches **93% of fully-supervised AUROC in one forward pass, beating a 27B LLM**. — [STAR RT page](https://star-project.stanford.edu/rt/); [arXiv 2510.06377](https://arxiv.org/pdf/2510.06377)
- **Open checkpoints:** STAR lists `rt-plurel` (86M, pretrained on PluRel synthetic databases) and the original 22M RT checkpoints. HF hosts `stanford-star/rt-j`, pretrained on PluRel and then on "the Join". Code, models and data are linked from `github.com/snap-stanford/relational-transformer`. — [STAR home](https://star-project.stanford.edu/); [HF stanford-star/rt-j](https://huggingface.co/stanford-star/rt-j); [STAR publications](https://star-project.stanford.edu/publications/)
- **PluRel** reports that synthetic data gives scaling laws for relational FMs. — [PluRel PDF](https://rishabh-ranjan.github.io/assets/2026_plurel.pdf)
- RT-J is also named in a ResearchGate listing, "RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions". — [ResearchGate](https://www.researchgate.net/publication/415822822_RT-J_Large-Scale_Pretraining_of_Relational_Transformers_for_Context-Efficient_Predictions)
- **KumoRFM / KumoRFM-2** (arXiv 2604.12596): closed weights. Papers note that API access does not let users control in-context examples, and that the KumoRFM-2 API arrived only shortly before their submission. — [KumoRFM-2 paper](https://arxiv.org/pdf/2604.12596); [OpenRFM paper](https://arxiv.org/html/2606.04320v1)
- **NVIDIA KumoRelational**: offered as an NVIDIA NIM. **Licence conflict:** the SDK table says OpenMDW-1.1, while the NIM model card says the NVIDIA Open Model License Agreement governs use. — [SDK table](https://nvidia.github.io/structured-data-models/api/models.html); [NIM reference](https://docs.api.nvidia.com/nim/re/reference/nvidia-kumo-relational); [build.nvidia.com model card](https://build.nvidia.com/nvidia/kumo-relational/modelcard)
- **Griffin** (May 2025, "Towards a Graph-Centric Relational Database Foundation Model"): pretrained over 150M+ nodes and listed as "Open-Source" on one leaderboard. It only pretrains the encoder and needs fine-tuning. Weights and licence are unconfirmed. — [SOTA2 RelBench leaderboard](https://www.sota2.com/research/sota/entity-regression-on-relbench-v1); [OpenRFM paper](https://arxiv.org/html/2606.04320v1)
- **RDBLearn** (arXiv 2602.18495): simple in-context prediction over RDBs. Reported as the best of the compared FMs, including closed KumoRFM, on regression. — [alphaXiv RDBLearn](https://www.alphaxiv.org/abs/2602.18495)
- **OpenRFM** (arXiv 2606.04320): an analysis paper on relational ICL. — [OpenRFM](https://arxiv.org/html/2606.04320v1)
- **RelGT** is supervised, not an FM (Dwivedi et al., ICLR 2026). RelGT-AC (arXiv 2606.03040) applies it to autocomplete tasks. — [SOTA2 leaderboard](https://www.sota2.com/research/sota/entity-regression-on-relbench-v1); [RelGT-AC](https://arxiv.org/html/2606.03040v1)
- Related tooling: RelAgent (LLM agents for relational learning, arXiv 2605.07840) and Relational Task Generation Language (arXiv 2609.01292). — [RelAgent](https://arxiv.org/html/2605.07840); [RTGL](https://arxiv.org/pdf/2609.01292)

### Inferences
- The cheapest relational win for TabTune is a **DFS-flattening front-end (TabPFN-Rel style) feeding any single-table FM it already supports**. That reuses all existing fine-tuning paths and adds no new model weights.
- The first native relational model worth adding is **RT-PluRel / RT-J**: open checkpoints, small (22 to 86M), with a strong benchmark position. TabTune's fit/predict API would need a multi-table input abstraction (entity table plus linked tables plus timestamps).
- KumoRFM cannot be integrated because there are no weights. KumoRelational might be, depending on how its licence conflict resolves.

### Gaps
- No licence was found for the RT / RT-PluRel / RT-J checkpoints or for Griffin.
- The ranking conflict on RelArena-α (TabPFN-Rel "top-ranked submission" vs RT-PluRel "#1") is unresolved.
- Whether KumoRelational weights can be downloaded (vs NIM only) is not confirmed.

## Q3. LLM-based / text-serialisation and multimodal tabular models: open weights and practical value

### Takeaway
**TabSTAR** (MIT code, 47M-param HF checkpoint, NeurIPS 2025) is the most practical open text-aware tabular FM and would complement ContextTab. **TabuLa-8B** (Llama-3-8B, Llama 3 licence) is still the main open-weight LLM serialisation predictor; it has had no update since mid-2024 and is costly to run. No open-weight FM handles tables plus images natively. MultiModalPFN (CVPR 2026) is the closest, with weights unconfirmed, and MulTaBench (NeurIPS 2026 Spotlight) is the benchmark to track.

### Cited Findings
- **TabSTAR** (arXiv 2505.18125, NeurIPS 2025; Arazi, Shapira et al., Technion): a tabular FM that handles text fields through an *unfrozen* pretrained text encoder with semantically target-aware representations. The GitHub repo and model page say **MIT**. HF `alana89/TabSTAR` is **47.3M params** (safetensors). **Licence inconsistency:** an older HF discussion diff shows cc-by-4.0. Reported to beat CARTE on 93% of classification splits and 36% of regression splits (≤10K examples). — [TabSTAR GitHub](https://github.com/alanarazi7/TabSTAR); [HF model](https://huggingface.co/alana89/TabSTAR); [HF discussion](https://huggingface.co/alana89/TabSTAR/discussions/1/files); [arXiv html](https://arxiv.org/html/2505.18125)
- **CARTE** is the earlier graph-based baseline; TabSTAR's benchmark is partly built from the CARTE paper. **TARTE** is a knowledge-pretrained transformer that multiplies cell values by FastText column-name embeddings. Licences for both were not found. — [TabSTAR arXiv html](https://arxiv.org/html/2505.18125); [emergentmind TabSTAR](https://www.emergentmind.com/topics/tabstar)
- **TabuLa-8B** (mlfoundations; arXiv 2406.12031, Jun 2024): fine-tuned from Llama-3-8B and released under the **Llama 3 licence**. It does classification and *binned* regression, serialises rows as "the <key> is <value>", uses row-causal masking for few-shot prediction, and was trained on the T4 dataset drawn from TabLib. It loads through `transformers`. No 2026 successor was found. — [HF tabula-8b](https://huggingface.co/mlfoundations/tabula-8b); [HF README](https://huggingface.co/mlfoundations/tabula-8b/raw/main/README.md); [arXiv 2406.12031](https://arxiv.org/pdf/2406.12031)
- "Data Language Models" (arXiv 2605.06290, 2026) groups TabuLa-8B with TableGPT2 and TabLLM as serialisation approaches. It finds the semantics-aware models (TabuLa-8B, ConTextTab) **lose 0.0709 and 0.0748 mean ROC-AUC when column names are removed**. — [Data Language Models](https://arxiv.org/html/2605.06290v1)
- A benchmark for TFMs on tabular data with text exists. — [arXiv 2507.07829](https://arxiv.org/pdf/2507.07829)
- **MulTaBench** (arXiv 2605.10616, NeurIPS 2026 Spotlight, 40 datasets) says the field "still lacks a true multimodal foundation model for tabular data with text and images" and that TFMs depend on frozen embeddings. It treats TabSTAR and ConTextTab as end-to-end text-capable models. — [MulTaBench abs](https://arxiv.org/abs/2605.10616); [MulTaBench html](https://arxiv.org/html/2605.10616v1)
- **MultiModalPFN** (CVPR 2026): extends TabPFN to tabular plus image or tabular plus text through a trained modality projector and partial TabPFN fine-tuning, evaluated on 9 datasets. Weights unconfirmed. — [MulTaBench html](https://arxiv.org/html/2605.10616v1); [cross-modality TFM evaluation, arXiv 2606.02106](https://arxiv.org/html/2606.02106v1)
- A systematic evaluation covers how TFMs transfer across 95 datasets and 7 modalities. — [arXiv 2606.02106](https://arxiv.org/html/2606.02106v1)
- Using TFMs for non-tabular tasks is also being studied. — [arXiv 2608.22594](https://arxiv.org/pdf/2608.22594)

### Inferences
- TabSTAR adds something TabTune lacks: text-cell support that is fine-tunable end to end and permissively licensed (ContextTab is its only current text-aware model). Its small size suits TabTune's SFT and LoRA workflows.
- TabuLa-8B would exercise TabTune's LoRA/PEFT path on an LLM backbone. The cost is high (8B params, row-serialised context) and its accuracy relative to current TFMs is not established, so it is a mid-priority, differentiating feature rather than an accuracy play.
- For multimodal input, the realistic near-term step is a "frozen embeddings into TFM" utility, with MultiModalPFN as the model to watch.

### Gaps
- CARTE and TARTE licences and current weight availability were not verified. Background knowledge, unverified: CARTE is from Inria (soda-inria) and installs via pip.
- TP-BERTa, TableGPT2 and other 2025/26 LLM tabular predictors were not checked for weights or licences.
- MultiModalPFN code/weights and licence are unknown.

## Q4. Licences, maintenance, and ranked recommendations for TabTune

### Takeaway
Among the candidates, the permissive ones (commercial use OK) are **Kumo Tabular (OpenMDW-1.1 weights, Apache-2.0 code)**, **TabSTAR (MIT, with an inconsistency noted)** and **TabuLa-8B (Llama 3 licence, commercial with conditions)**. **LimiX-2** and **TabPFN-2.5+ / TabPFN-Rel weights** are non-commercial. Active 2026 maintenance is confirmed for Kumo Tabular, LimiX-2, TabDPT (v1.3 in Sep 2026), TabPFN-Rel and Stanford RT. Kumo Tabular ranks first.

### Cited Findings
- Kumo Tabular: weights OpenMDW-1.1, code Apache-2.0, commercial use explicitly allowed. — [SDK table](https://nvidia.github.io/structured-data-models/api/models.html); [HF blog](https://huggingface.co/blog/nvidia/kumo-tabular)
- LimiX-2: StableAI LimiX Non-Commercial License v1.0. — [Pandaily](https://pandaily.com/stable-ai-limix-2-structured-data-foundation-model)
- TabPFN-2.5+ weights are non-commercial; `tabpfn` code is Apache-2.0 from v9.0.0 (Sep 2026). — [Wikipedia TabPFN](https://en.wikipedia.org/wiki/TabPFN)
- TabICLv2 (already in TabTune) weights are BSD-3 with no gating, a useful reference point. — [MulTaBench-related search summary; metavert](https://metavert.io/tabular-foundation-models)
- TabSTAR: MIT per GitHub; cc-by-4.0 per an older HF diff. — [GitHub](https://github.com/alanarazi7/TabSTAR); [HF discussion](https://huggingface.co/alana89/TabSTAR/discussions/1/files)
- TabuLa-8B: Llama 3 licence. — [HF tabula-8b](https://huggingface.co/mlfoundations/tabula-8b)
- TabDPT releases v1.2 (Jun 2026) and v1.3 (Sep 2026). — [PyPI tabdpt](https://pypi.org/project/tabdpt/)

### Candidate catalogue (summary table; sources are in the sections above)

| Candidate | Org | Release | Type | Tasks | Size / limits | Weights licence | Benchmark standing | FT support | Status |
|---|---|---|---|---|---|---|---|---|---|
| **NVIDIA Kumo Tabular** | NVIDIA (ex-Kumo AI) | 2026-09-28/29 | Single-table ICL (TabICL/TabPFN-style, SCM synthetic prior) | clf, reg | 3 sizes, 28M–215M (SDK: 27–61M clf); num+cat only; row/feature caps unverified | **OpenMDW-1.1 (commercial OK)**, code Apache-2.0 | Vendor: Pareto SOTA vs TabArena; in TALENT Sep-2026 wave | Unverified | **New** |
| LimiX-2 | Stable AI + Tsinghua | 2026-09-16 | Contextual Mechanism Network | clf, reg, imputation, causal skeleton | 400M | **Non-commercial** | Vendor Elo: TabArena 1935, TALENT 1506; one Oct-2026 post has it 1st (1971) | Unverified | Upgrade (LimiX) |
| LimiX-2M | (LimiX line) | 2026 (arXiv 2606.04485) | Lightweight TFM | clf/reg | 2M | Unverified | — | — | Upgrade/variant |
| TabDPT v1.2-Turbo / v1.3 | Layer 6 AI | Jun / Sep 2026 | Real-data-pretrained ICL | clf, reg | — | Unverified for v1.2+ | Turbo: "fastest overall" (authors) | Training code released (v1) | Upgrade (TabDPT) |
| RealTabPFN-2.5 | Prior Labs | 2025 | TabPFN plus real-data continued pretraining | clf, reg | — | Non-commercial | Once top TFM on TabArena (older snapshot) | — | Variant |
| **TabSTAR** | Technion (Arazi et al.) | May 2025 / NeurIPS 2025 | Unfrozen text encoder plus target-aware tabular transformer | clf, reg (text fields) | 47.3M; benchmarked ≤10K rows | **MIT** (cc-by-4.0 inconsistency) | Beats CARTE on 93% clf / 36% reg splits | Designed for per-dataset FT | **New** |
| CARTE / TARTE | Inria (CARTE) | 2024 / 2025 | Graph-based / knowledge-pretrained (FastText column embeddings) | clf, reg | — | Unverified | Below TabSTAR | FT required | New (low) |
| **TabuLa-8B** | mlfoundations (UW et al.) | Jun 2024 | Llama-3-8B, row serialisation | clf, binned reg | 8B; context-limited | Llama 3 licence | Older; column-name sensitive | LLM FT/LoRA | **New** |
| **Relational Transformer (RT / RT-PluRel / RT-J)** | Stanford STAR / SNAP | ICLR 2026; 2026 | Relational ICL transformer | Relational clf/reg | 22M / 86M | Unverified | RT-PluRel "#1" on RelArena-α (STAR) | Zero-shot; FT possible | **New (relational)** |
| **TabPFN-Rel** | Prior Labs | 2026 | DFS flattening plus TabPFN-3 | Relational clf/reg | TabPFN-3 limits (≤1M rows) | TabPFN-3 weights non-commercial; package Apache-2.0 | "Top-ranked submission" on RelArena-α | Inherits TabPFN-3 FT | **New harness on an existing model** |
| NVIDIA KumoRelational | NVIDIA | 2026 | Relational FM (NIM) | Relational | — | **Conflict**: OpenMDW-1.1 vs NVIDIA Open Model License | — | — | New (unclear) |
| KumoRFM / KumoRFM-2 | Kumo AI | 2025 / 2026 | Relational ICL (graph) | Relational | — | **Closed (API only)** | — | No | Not integrable |
| Griffin | (Amazon-affiliated per query; unverified) | May 2025 | Graph-centric RDB FM | Relational | 150M+ node pretraining | Unverified | — | FT required | Watch |
| RDBLearn | — | Feb 2026 | Simple relational ICL | Relational | — | Unverified | Beat KumoRFM on regression (authors) | — | Watch |
| TabFlex | Zeng et al. | 2024 | Linear-attention ICL | clf | Millions of rows | Unverified | Speed claims only | — | Low |
| MotherNet | Microsoft | Dec 2023 | Hypernetwork that emits an MLP | clf | TabPFN-like limits | Unverified (`microsoft/ticl`) | Superseded | — | Low |
| TabForestPFN | den Breejen et al. | 2024 | Forest-prior ICL | clf | — | Unverified | Withdrawn ICLR'25 | FT-centric | Low |
| APT | — | 2025 | Adversarially pretrained ICL | clf/reg | — | Unverified | — | — | Low |
| TAFFY | — | Oct 2026 | Looped transformer plus diversity prior | clf/reg | — | Unknown | — | — | Watch |
| TabSwift | — | Jun 2026 | Row-wise attention TFM | — | — | Unknown | — | — | Watch |
| MultiModalPFN | — | CVPR 2026 | TabPFN plus modality projector | Multimodal clf/reg | — | Unknown | — | Partial FT | Watch |

### Ranked top 8 by value for TabTune
1. **NVIDIA Kumo Tabular**: the only new near-SOTA single-table FM with commercial-use weights; already in TALENT's Sep 2026 set; ICL architecture matches TabTune's FT recipes.
2. **LimiX-2 (upgrade to the LimiX adapter)**: possibly #1 on TabArena (vendor 1935 Elo; one post says 1971), adds imputation. Ship as non-commercial/research.
3. **TabSTAR**: MIT, small, text-native and FT-native; fills the text-column gap next to ContextTab.
4. **TabPFN-Rel-style relational harness**: low engineering cost on top of TabPFN v3 (DFS flattening), top-ranked on RelArena-α, and generalises to every TabTune backbone, including the commercially usable ones.
5. **Relational Transformer (RT-PluRel / RT-J)**: the best open-checkpoint native relational FM (22M to 86M); opens a relational track. Check its licence first.
6. **TabDPT v1.2-Turbo / v1.3 (upgrade)**: cheap checkpoint bump with a speed gain.
7. **TabuLa-8B**: the only mature open-weight LLM-serialisation predictor; exercises TabTune's LoRA path on an LLM; Llama 3 licence. Accuracy relative to current TFMs is unproven.
8. **NVIDIA KumoRelational**: add only if the licence conflict resolves to permissive and weights can be downloaded.
- Watchlist: TAFFY, TabSwift, LimiX-2M, MultiModalPFN, Griffin, RDBLearn, CARTE/TARTE. Fine-tuning features to consider: the TFM-Retouche input adapter and GEAR distillation.

### Inferences
- The ranking weighs (a) leaderboard strength, (b) licence permissiveness, (c) integration cost given TabTune's existing adapters, and (d) whether the model opens a new capability (text, relational, LLM). Items 1, 3 and 4 score well on all four.
- Because the Sep 2026 models arrived in one burst and TALENT had to fix wrappers that "silently ran a different model", TabTune should add parity tests against the reference implementations for each new adapter.

### Gaps
- Kumo Tabular fine-tuning hooks and size limits; RT and Griffin licences; TabDPT v1.2+ licence; live TabArena positions for Kumo Tabular and LimiX-2 (independent rather than vendor numbers).
- Network block: arxiv.org, tabarena.ai and thetabularguy.substack.com must be allowed in the session's network settings so primary pages can be checked directly.
