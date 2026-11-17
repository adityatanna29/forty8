# TabTune demo notebooks

Four notebooks, all on **TabTune open source**, all in the same ten-section format. In 01 and 02 the Lexsi platform
is shown as reference text (options dictionary, equivalent calls); nothing calls the platform backend. 03 and 04 are
library-only: time series is not on the platform, so apart from the LLM gateway used for the model card they carry no platform or SDK references.

| Notebook | Task | Use case |
|---|---|---|
| `01_lending_club_classification.ipynb` | Classification | Lending Club credit default |
| `02_home_credit_regression.ipynb` | Regression | Home Credit loan amount (`AMT_CREDIT`) |
| `03_enterprise_timeseries_anomaly.ipynb` | Forecasting + anomaly detection | Engineered multi-entity bank ledger (about 1M postings, 32 series) |
| `04_real_world_cashflow_treasury.ipynb` | Forecasting + anomaly replay | US Daily Treasury Statement (490k rows, 18 lines) |

## How the notebooks read

* **Short visible cells are the calls that matter**: `TabularPipeline(...)`, `fit`, `predict` / `predict_proba`, `evaluate`,
  `save` / `load`, conformal `calibrate` / `predict_set` / `predict_interval`, `TimeSeriesSchema`, `split_horizon`,
  `TimeSeriesLeaderboard.run`, `TimeSeriesPipeline.fit / predict / evaluate / calibrate`, the agents and the webhook.
* **Collapsed cells** (Colab forms, ▸ *Show code*) hold plotting and helper code. Their output still shows.

## The format (every notebook)

1. **Onboarding** – install TabTune; model dictionary read live from the TabTune registry; Lexsi platform options (01 and 02 only)
2. **Product options** – problem statement, training and test data, development vs production
3. **Data preparation** – profile dashboards; a data preparation agent with human approval gates, including a
   "too good to be true" leakage probe (tabular); feature engineering; conclusions
4. **Model training** – one model step by step, AutoML race, hyperparameter tuning, selection rule, global explainability
5. **Model testing** – evaluation dashboards on held-out data, interactive controls
6. **Batch inference** – save, reload and score a file
7. **Model risk** – conformal uncertainty with measured coverage, fairness (tabular), accuracy guards, regulatory evidence map
8. **Artefacts before production** – model card written through the **Lexsi LLM gateway**
   (`/gateway/v1/chat/completions`, any provider registered in the Lexsi project) from a documented prompt: role, citation
   and number rules, what each of the 14 sections must contain, the fact sheet with a key per fact, and metric notes.
   Automatic checks (citation keys, number matching, sections, forbidden content) and one revision pass; API contract
9. **Model in production** – monitoring options (data drift, prediction drift, performance, calibration, data quality,
   fairness; series drift, band coverage and scheduled review for time series), alert rules with warning and critical
   levels, a signed **webhook** that receives alerts and queues jobs for the **retraining agent**, champion vs challenger
   with human approval, regenerated model card with change log
10. **Conclusions** – plain-English summary generated from the recorded facts, plus the audit trail

## Running

* Colab: *Runtime → Change runtime type → A100 or H100, High-RAM*, then *Run all*.
* 01 and 02 mount Drive and read `MyDrive/datasets/lc/` and `MyDrive/datasets/home-credit-default-risk/processed/`.
  03 generates its data; 04 downloads from the Treasury API (about two minutes, cached).
* `AUTO_APPROVE = True` (imports cell) lets the agents' approval gates pass automatically and logs them as auto-approved.
  Set it to `False` to be asked at each gate during a live demo.
* Model-card writer (section 8.1): preset to Claude Sonnet 5.5 through Open Router (`anthropic-claude-sonnet-5.5_v1`,
  registered in Lexsi project `tabtunedemo_WJQPD747O8`), called through the Lexsi gateway's `/chat/completions` with a
  32,000-token budget and short reasoning. Add a Colab secret `LEXSI_API_TOKEN`. Only the aggregate fact sheet is sent.
  `/text/generation` is available as `LEXSI_ENDPOINT = "text"` for Lexsi-hosted models. Without the token the notebook
  falls back to open weights on the GPU, then to a deterministic template.
* The webhook runs inside the notebook on a free local port; in production it would run as a service behind the API gateway.
* `TABTUNE_DEMO_FAST=1` runs a small smoke test without a GPU.

## What has and has not been executed

All four notebooks were executed end to end in fast mode on CPU with real TabTune models and stand-in data
(the regression stand-in includes deliberately leaky ratio features, which the data preparation agent removes).
The model-card path was run against a local stand-in for the Lexsi gateway that mirrors its request checks and
response format; it has not been run against the live gateway from this environment. Not yet executed in this
version: the full-size GPU settings on the real data and the Chronos models inside the new monitoring sections.

`EXTRA_CELLS_sdk_notebooks.md` and `EXTRA_CELL_timeseries_models.md` are earlier add-on cells, kept for reference.
