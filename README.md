# sarailmax — Superstore Sales Forecasting

Monthly sales forecasting for the Superstore dataset using a SARIMAX time series model, adapted from a Microsoft Fabric sample notebook so it can run locally without Spark or a Fabric workspace.

## Project Layout

| Path | Purpose |
|---|---|
| `AIsample - Superstore Forecast v1.2 (2).ipynb` | Original Microsoft Fabric sample notebook (Spark/lakehouse-specific). Reference only — not run directly. |
| `transform_notebook.py` | Converts the Fabric notebook into a locally runnable version by rewriting lakehouse paths, `display()` calls, Spark writes, and MLflow tracking URI. |
| `Superstore_Forecast_Local.ipynb` | The notebook you actually run. Generated from the file above, with local fixes on top. |
| `setup_local.py` | Creates `data/`, `output/`, `mlruns/` and checks that `data/raw/Superstore.xlsx` exists. |
| `requirements.txt` | Python dependencies. |
| `data/raw/` | Input data directory (place `Superstore.xlsx` here). |
| `output/` | Forecast results (`sales_actual_forecast.csv`). |
| `mlruns/` | Local MLflow experiment tracking store. |

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place `Superstore.xlsx` in `data/raw/` (the notebook will also attempt to download the demo dataset automatically if it's missing and `IS_CUSTOM_DATA = False`).
3. Run the setup script to create the required directories and verify the data file:
   ```
   python setup_local.py
   ```
4. Open and run `Superstore_Forecast_Local.ipynb` top to bottom.

## What the Notebook Does

1. **Load & clean** the Superstore Excel export (9,994 rows) and aggregate daily sales into a monthly series.
2. **Exploratory analysis** — trend/seasonality visualization and decomposition.
3. **Model selection** — ADF stationarity tests determine the differencing orders (d, D), then a grid search over the remaining SARIMAX parameters picks the lowest-AIC configuration.
4. **Train & validate** the final SARIMAX model, holding out the last 12 months to compute MAPE.
5. **Export** actual + forecast values (with confidence intervals) to `output/sales_actual_forecast.csv`, and log the run (parameters, MAPE, AIC, model artifact) to MLflow under `mlruns/`.

## Regenerating the Local Notebook

If the original Fabric sample notebook changes, regenerate the local version with:
```
python transform_notebook.py
```
This only handles Fabric → local path/API rewrites — any manual fixes made directly in `Superstore_Forecast_Local.ipynb` are not carried over automatically and would need to be reapplied.

## Known Limitations

- Only ~4 years of monthly data (~48 points) is available, and the model is validated with a single 12-month holdout rather than rolling-origin cross-validation, so the reported MAPE is an approximate accuracy estimate.
- Forecasts are for total company-wide sales; there is no per-category/per-region breakdown.
- No external regressors (holidays, promotions, macro indicators) are included.
