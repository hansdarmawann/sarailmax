# sarailmax — Superstore Sales Forecasting

Monthly sales forecasting for the Superstore dataset using a SARIMAX time series model.

## Project Layout

```
sarailmax/
├── notebooks/
│   └── Superstore_Forecast_Local.ipynb   # the notebook you run
├── src/                                   # pipeline logic used by the notebook
│   ├── config.py       # paths & constants (experiment name, validation/forecast horizon)
│   ├── setup.py        # create data/output/mlruns dirs, verify the data file
│   ├── data.py         # download/load/preprocess/aggregate the Superstore data
│   ├── modeling.py     # ADF stationarity tests, SARIMAX grid search, training
│   ├── evaluation.py   # forecast generation, validation, MAPE
│   ├── export.py       # build & save the actual-vs-forecast CSV
│   └── tracking.py     # MLflow experiment tracking helpers
├── data/raw/            # input data directory (place Superstore.xlsx here)
├── output/              # forecast results (sales_actual_forecast.csv)
├── mlruns/              # local MLflow experiment tracking store
└── requirements.txt
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place `Superstore.xlsx` in `data/raw/` (the notebook will also attempt to download the demo dataset automatically if it's missing and `IS_CUSTOM_DATA = False` in `src/config.py`).
3. Open and run `notebooks/superstore_forecast.ipynb` top to bottom. The first two cells add the repo root to `sys.path` and create the required directories, so no separate setup script is needed.

## What the Notebook Does

1. **Load & clean** the Superstore Excel export (9,994 rows) and aggregate daily sales into a monthly series (`src/data.py`).
2. **Exploratory analysis** — trend/seasonality visualization and decomposition (inline in the notebook).
3. **Model selection** — ADF stationarity tests determine the differencing orders (d, D), then a grid search over the remaining SARIMAX parameters picks the lowest-AIC configuration (`src/modeling.py`).
4. **Train & validate** the final SARIMAX model, holding out the last 12 months to compute MAPE (`src/evaluation.py`).
5. **Export** actual + forecast values (with confidence intervals) to `output/sales_actual_forecast.csv`, and log the run (parameters, MAPE, AIC, model artifact) to MLflow under `mlruns/` (`src/export.py`, `src/tracking.py`).

## Known Limitations

- Only ~4 years of monthly data (~48 points) is available, and the model is validated with a single 12-month holdout rather than rolling-origin cross-validation, so the reported MAPE is an approximate accuracy estimate.
- Forecasts are for total company-wide sales; there is no per-category/per-region breakdown.
- No external regressors (holidays, promotions, macro indicators) are included.

## Current Evaluation Status

- The selected model remains SARIMAX `(0, 0, 0) x (0, 1, 2, 12)` with `d = 0` and `D = 1`.
- Validation MAPE is approximately **24.52%**, so the current result does not meet the **20%** target threshold.
- Explicitly testing `(d, D)` combinations showed that `(0, 1)` remains the preferred configuration by AIC; `(1, 1)` tied on AIC but did not provide a clear improvement.
- Residual diagnostics do not show significant first-order autocorrelation (Ljung–Box p-value `0.46`).
