# TabTune demo notebooks

| Notebook | Stack | Use case |
|---|---|---|
| `01_lending_club_classification_sdk.ipynb` | Lexsi SDK `0.1.69` (platform) | Lending Club credit default |
| `02_home_credit_regression_sdk.ipynb` | Lexsi SDK `0.1.69` (platform) | Home Credit loan amount (`AMT_CREDIT`) |
| `03_enterprise_timeseries_anomaly.ipynb` | TabTune open source (`tabtune.TimeSeries`) | Engineered multi-entity bank ledger (about 1M postings): data preparation, feature engineering, liquidity forecasting, anomaly detection |
| `04_real_world_cashflow_treasury.ipynb` | TabTune open source (`tabtune.TimeSeries`) | Real data: US Daily Treasury Statement (490k rows), label harmonisation, cash-flow forecasting, COVID anomaly replay |

## Running

* **01 and 02 (SDK)**: open in Colab, *Run all*. They mount Drive and read the processed files from
  `MyDrive/datasets/lc/` and `MyDrive/datasets/home-credit-default-risk/processed/`, then prompt for a
  Lexsi SDK token. Edit only the configuration cell (organization, workspace, project name, roster, flags).
* **03 and 04 (open source)**: Colab Pro+ with a GPU runtime, *Run all*. `TABTUNE_DEMO_FAST=1` runs it without
  a GPU using weight-free models.

## What the SDK notebooks rely on (checked against `lexsi-backend-prod_v4` and the SDK source)

* Strategies: `simple`, `ensemble`, `distillation`, `leaderboard`; advanced builds accept `tunning_strategy="inference"` only.
* Every build passes `tags=["Training"]`, `test_tags=["Test"]` and the feature exclusions explicitly; the SDK
  does not carry the project's exclusions into a build.
* The backend drops any column with more than 95% unique values; the notebooks round such numeric
  columns first (pre-flight cell) so continuous features survive.
* Builds are capped at 10,000 rows (train + test) unless `sample_percentage` is set.
* Support set: classification only, inference only, models TabICL, OrionMSP, OrionBix, TabPFNv26, Limix,
  TabPFNv3; it must be enabled on the account plan. The notebooks fall back to SHAP and LIME automatically.
* Ensembles are explained with LIME (SHAP is not allowed for ensembles). TabPFNv3 needs `tabpfn_token`.
* Inference output columns: `Predicted_value_AutoML`, `Prediction_category_AutoML`, `pred_proba_AutoML`
  (class-1 probability). `Prediction_category_AutoML` is overwritten with the true label when the scored
  tag contains the target, so the notebooks evaluate from `pred_proba_AutoML`.
* Fairness, cross-validation and the model-performance dashboard are not produced for TFM builds through
  the API; the classification notebook computes a group check locally.

The SDK notebooks were executed end to end against a test double that mimics these payloads; they have
not yet been run against the live backend from this environment.
