# TabTune demo notebooks

| Notebook | Stack | Status |
|---|---|---|
| `01_classification_lending_club.ipynb` | TabTune SDK | waiting for SDK backend code and the earlier Lending Club notebook |
| `02_regression.ipynb` | TabTune SDK | waiting for SDK backend code |
| `03_enterprise_timeseries_anomaly.ipynb` | TabTune open source (`tabtune.TimeSeries`) | ready for Colab Pro+ |

## 03 · Enterprise time series and anomaly detection

Open in Colab, choose a GPU runtime (A100 or L4), then *Run all*. The first cell installs TabTune from
GitHub; model weights (Chronos-2, Chronos-Bolt) download on first use.

- Data: a synthetic 20-series GCC banking panel built in the notebook (branch cash, corporate payment
  outflows, card spend) with Ramadan, Eid, payday and Friday–Saturday weekend effects. No customer data.
- Sections: exploratory dashboard, leaderboard on rolling backtests, 4-week forecast with calendar
  covariates, conformal calibration, executive liquidity dashboard, ensemble, embedding similarity,
  anomaly detection with labelled incidents and an alert dashboard.
- `TABTUNE_DEMO_FAST=1` swaps foundation models for weight-free models, for a quick smoke test without a GPU.
- `RUN_FINETUNE = True` in the config cell adds an optional LoRA fine-tuning comparison.
