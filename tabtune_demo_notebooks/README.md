# TabTune demo notebooks

Six notebooks in the same ten-section format.

* **01–04 run on TabTune open source.** In 01 and 02 the Lexsi platform is shown as reference text; nothing calls the
  platform backend except the LLM gateway used for the model card. 03 and 04 are library-only (time series is not on the platform).
* **05 and 06 run the same flow on the Lexsi platform** through `lexsi-sdk`: training, scoring, explanations, observations
  and policies, drift dashboards, monitors and alerts, and the LLM gateway behind every agent. See "Platform notebooks" below.

| Notebook | Task | Use case |
|---|---|---|
| `01_lending_club_classification.ipynb` | Classification | Lending Club credit default |
| `02_home_credit_regression.ipynb` | Regression | Home Credit loan amount (`AMT_CREDIT`) |
| `03_enterprise_timeseries_anomaly.ipynb` | Forecasting + anomaly detection | Engineered multi-entity bank ledger (about 1M postings, 32 series) |
| `04_real_world_cashflow_treasury.ipynb` | Forecasting + anomaly replay | US Daily Treasury Statement (490k rows, 18 lines) |
| `05_lending_club_classification_sdk.ipynb` | Classification, on the Lexsi platform | Lending Club credit default |
| `06_home_credit_regression_sdk.ipynb` | Regression, on the Lexsi platform | Home Credit loan amount (`AMT_CREDIT`) |

## How the notebooks read (01–04; 05 and 06 follow the same convention with platform calls)

* **Short visible cells are the calls that matter**: `TabularPipeline(...)`, `fit`, `predict` / `predict_proba`, `evaluate`,
  `save` / `load`, conformal `calibrate` / `predict_set` / `predict_interval`, `TimeSeriesSchema`, `split_horizon`,
  `TimeSeriesLeaderboard.run`, `TimeSeriesPipeline.fit / predict / evaluate / calibrate`, the agents and the webhook.
* **Collapsed cells** (Colab forms, ▸ *Show code*) hold plotting and helper code. Their output still shows.

## The format (01–04)

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

Notebooks 01–04 were executed end to end in fast mode on CPU with real TabTune models and stand-in data
(the regression stand-in includes deliberately leaky ratio features, which the data preparation agent removes).
The model-card path was run against a local stand-in for the Lexsi gateway that mirrors its request checks and
response format; it has not been run against the live gateway from this environment. Not yet executed in this
version: the full-size GPU settings on the real data and the Chronos models inside the new monitoring sections.

`EXTRA_CELLS_sdk_notebooks.md` and `EXTRA_CELL_timeseries_models.md` are earlier add-on cells, kept for reference.

## Platform notebooks (05, 06)

Same ten sections, with the work done on Lexsi endpoints. Colab needs no GPU: models run on platform compute.

| Section | What runs on the platform |
|---|---|
| 1 Onboarding | `lexsi.login()`, organization → workspace → project; a table mapping every SDK call to its REST endpoint; the LLM gateway (`/gateway/v1/chat/completions`) with a role → (provider, model) table, a side-by-side model comparison, model search and registration |
| 3 Data preparation | The data-prep agent's checks run in the notebook; each proposal is reviewed by an LLM through the gateway, then a human gate. Upload by tag (`Training`, `Test`, `Prod_M1…M4`), `data_observations`, `data_warnings` |
| 4 Training | `train_model`: zero-shot foundation models, ensemble, leaderboard, distillation; selection from the platform's metrics; global SHAP / LIME from `model_summary` |
| 5–6 Testing, batch | `model_inference` on the Test tag and on a production tag |
| 7 Model risk | Conformal coverage and fairness on platform scores; **observations** (`create_observation`) and **policies** (`create_policy`) |
| 8 Artefacts | Model card through the gateway with the same prompt and automatic checks as 01–03; `case_predict(..., risk_policies=True)` with SHAP, LIME, observations and policies |
| 9 Production | `get_data_drift_dashboard`, `get_target_drift_dashboard`, `data_drift_diagnosis`, `create_monitor`, `alerts`; alerts go to the signed webhook; the retraining agent (LLM review + human gates) retrains with `train_model`, verifies with `model_inference`, promotes with `activate_model` |
| 10 Traces | Every LLM call per agent (model, tokens, seconds, decision, rationale), and the gateway's own record: `sessions()` → `messages(session_id)` → `traces(trace_id)` |

**Agents and LLMs.** The data-prep agent, the retraining agent and the model-card writer each call an LLM through the gateway. The
model is whatever `GATEWAY_MODELS` names for the role; it is preset to the model registered in the gateway project (Claude Sonnet 5.5
through Open Router). To run the agents on an open-weights model, register one in the project (cell 1.3c) and point the `agents`
role at it. Deterministic checks compute the evidence and stay in charge of policy-mandated steps; without a gateway the agents
use their rule defaults and say so.

**Settings to check before running** (config cell in section 1): `ORGANIZATION`, `WORKSPACE`, `PROJECT`, `COMPUTE`,
`LLM_PROJECT` / `LLM_PROJECT_ID` / `LLM_CLIENT_ID`, `GATEWAY_MODELS`, `ROSTER`. `ALERT_EMAILS` is empty by default, so monitors
send no e-mail. Add the `LEXSI_API_TOKEN` Colab secret: the gateway is called through the SDK first and through REST with this
token if the SDK route is not available.

**Expression rules for observations and policies** (from the SDK's parser): comparison values must be whole words or whole
numbers (no decimals, no minus sign); the digits of a value must not occur inside its feature name; a feature after `and` / `or`
must not start with `in`, `notin` or `contains`. The notebooks pick thresholds and features that respect these rules.

**What has and has not been executed.** Both notebooks were run end to end against a local stand-in for the SDK and the gateway
(SDK gateway route, REST route, SDK route failing over to REST, no gateway at all, platform alerts present, promotion and
no-promotion outcomes). The stand-in uses the real SDK's expression parser and the method signatures of `lexsi-sdk` 0.1.69. They
have **not** been run against the live Lexsi backend from this environment. Every platform call is wrapped: a call the account
cannot run prints the platform's message and the notebook carries on.
