# TabTune demo notebooks

Four notebooks, all on **TabTune open source**, all in the same ten-section format. The Lexsi platform is shown
as reference (options dictionary, equivalent SDK calls, API contract); nothing calls the platform backend.

| Notebook | Task | Use case |
|---|---|---|
| `01_lending_club_classification.ipynb` | Classification | Lending Club credit default |
| `02_home_credit_regression.ipynb` | Regression | Home Credit loan amount (`AMT_CREDIT`) |
| `03_enterprise_timeseries_anomaly.ipynb` | Forecasting + anomaly detection | Engineered multi-entity bank ledger (about 1M postings, 32 series) |
| `04_real_world_cashflow_treasury.ipynb` | Forecasting + anomaly replay | US Daily Treasury Statement (490k rows, 18 lines) |

## The format (every notebook)

1. **Onboarding** – install TabTune and the Lexsi SDK; model dictionary read live from the TabTune registry; Lexsi platform options
2. **Product options** – problem statement, training and test data, development vs production
3. **Data preparation** – profile dashboards, a data preparation agent with human approval gates, feature engineering, conclusions
4. **Model training** – AutoML race, hyperparameter tuning, selection rule, global explainability
5. **Model testing** – evaluation dashboards on held-out data, interactive controls
6. **Batch inference** – score a file, throughput, Lexsi equivalent
7. **Model risk** – conformal uncertainty with measured coverage, fairness (tabular), accuracy guards, regulatory evidence map
8. **Artefacts before production** – model card written by an open-source LLM (Qwen 2.5 7B) with trace and a numeric grounding check; API contract with a live round trip
9. **Model in production** – drift monitoring, retraining agent with human approval, regenerated model card with change log
10. **Conclusions** – plain-English summary generated from the recorded facts, plus the audit trail

## Running

* Colab: *Runtime → Change runtime type → A100 GPU, High-RAM*, then *Run all*.
* 01 and 02 mount Drive and read `MyDrive/datasets/lc/` and `MyDrive/datasets/home-credit-default-risk/processed/`.
  03 generates its data; 04 downloads from the Treasury API (about two minutes, cached).
* `AUTO_APPROVE = True` (imports cell) lets the agents' approval gates pass automatically and logs them as auto-approved.
  Set it to `False` to be asked at each gate during a live demo.
* The model card uses `Qwen/Qwen2.5-7B-Instruct` on the GPU (about 15 GB download on first run). Set `LLM_MODEL = None`
  to use the deterministic template writer instead; the notebook also falls back to it if the LLM fails to load.
* `TABTUNE_DEMO_FAST=1` runs a small smoke test without a GPU.

## What has and has not been executed

All four notebooks were executed end to end in fast mode on CPU with real TabTune models
(tabular: OrionMSP, TabICLv2, TabPFN zero-shot, OrionMSP LoRA, `TabularEnsemble`, conformal prediction;
time series: baselines and the tree forecaster). Not yet executed: the full-size A100 settings, the Chronos
foundation models inside the new sections of 03 and 04, and the Qwen model-card writer (fast mode uses the template writer).
Run each notebook once on the A100 before the demo.

`EXTRA_CELLS_sdk_notebooks.md` and `EXTRA_CELL_timeseries_models.md` are earlier add-on cells, kept for reference.
