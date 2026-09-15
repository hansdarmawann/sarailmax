# sarailmax — Superstore Sales Forecasting

Monthly sales forecasting for the Superstore dataset using a SARIMAX time-series model.

## Project Layout

```text
sarailmax/
├── notebooks/superstore_forecast.ipynb  # exploratory analysis and workflow notebook
├── src/                                 # reusable pipeline logic
├── tests/                               # unit and integration tests
├── .github/workflows/tests.yml          # GitHub Actions CI workflow
├── pytest.ini                           # pytest configuration
├── TESTING.md                           # testing guide
├── data/                                # local input data, ignored by Git
├── output/                              # generated results, ignored by Git
├── mlruns/                              # local MLflow store, ignored by Git
└── requirements.txt
```

The main source modules are `config.py`, `setup.py`, `data.py`, `modeling.py`, `evaluation.py`, `export.py`, `pipeline.py`, and `tracking.py`.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place `Superstore.xlsx` in `data/raw/`. If it is missing, the pipeline/notebook attempts to download the demo dataset when `IS_CUSTOM_DATA = False` in `src/config.py`.

3. Run the end-to-end pipeline:

   ```bash
   python -m src.pipeline
   ```

4. Alternatively, open and run `notebooks/superstore_forecast.ipynb` from top to bottom.

## Testing

Run the automated unit and integration test suite with:

```bash
python -m pytest
```

Tests cover configuration, data processing, SARIMAX modeling, evaluation, export, setup, and the end-to-end pipeline. GitHub Actions runs the same test suite automatically on every push and pull request through `.github/workflows/tests.yml`.

## What the Pipeline Does

1. **Load and clean** the Superstore Excel export and aggregate daily sales into a monthly series (`src/data.py`).
2. **Explore** trend and seasonality through visualizations and decomposition in the notebook.
3. **Select a model** using ADF stationarity tests and a SARIMAX parameter grid search (`src/modeling.py`).
4. **Train and validate** using a time-ordered holdout of the last 12 months (`src/evaluation.py`).
5. **Export** actual and forecast values with confidence intervals to `output/sales_actual_forecast.csv`, with optional MLflow tracking under `mlruns/` (`src/export.py`, `src/tracking.py`).

## Known Limitations

- Only approximately four years of monthly data is available, and validation uses a single 12-month holdout rather than rolling-origin cross-validation.
- Forecasts cover total company-wide sales; there is no per-category or per-region breakdown.
- No external regressors such as holidays, promotions, or macro indicators are included.
- The full test suite includes SARIMAX fitting and can take several minutes to complete.

## Current Evaluation Status

- The selected model is SARIMAX `(0, 0, 0) x (0, 1, 2, 12)` with `d = 0` and `D = 1`.
- Validation MAPE is approximately **24.52%**, above the **20%** target threshold.
- Explicit testing of `(d, D)` combinations preferred `(0, 1)` by AIC; `(1, 1)` tied on AIC without a clear improvement.
- Residual diagnostics did not show significant first-order autocorrelation (Ljung–Box p-value `0.46`).

The evaluation figures above are the latest recorded results and should be refreshed when the data or modeling approach changes.
