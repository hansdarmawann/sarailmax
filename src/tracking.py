"""MLflow experiment tracking helpers.

Imports mlflow lazily inside each function (rather than at module load time)
so this module can still be imported when MLflow isn't installed - the
notebook only calls these functions inside its `if MLFLOW_AVAILABLE:` branch.
"""

from pathlib import Path


def init_mlflow(experiment_name: str, tracking_dir) -> None:
    import mlflow

    # Windows absolute paths (e.g. "C:\...") are misread as a URI scheme by
    # mlflow unless converted to a proper file:// URI first.
    mlflow.set_tracking_uri(Path(tracking_dir).resolve().as_uri())
    mlflow.set_experiment(experiment_name)
    mlflow.autolog(disable=True)


def log_sarimax_run(results, order: tuple, seasonal_order: tuple, experiment_name: str) -> str:
    import mlflow
    import mlflow.statsmodels

    model_name = f"{experiment_name}-Sarimax"
    with mlflow.start_run(run_name="Sarimax") as run:
        mlflow.statsmodels.log_model(results, model_name, registered_model_name=model_name)
        mlflow.log_params({
            "order": order,
            "seasonal_order": seasonal_order,
            "enforce_stationarity": False,
            "enforce_invertibility": False,
        })
        model_uri = f"runs:/{run.info.run_id}/{model_name}"

    return model_uri


def log_forecast_run(results, order: tuple, seasonal_order: tuple, mape: float,
                      validation_months: int, forecast_months: int) -> str:
    import mlflow
    import mlflow.statsmodels

    with mlflow.start_run(run_name="SARIMAX_Forecast") as run:
        mlflow.log_params({
            "order": order,
            "seasonal_order": seasonal_order,
            "validation_months": validation_months,
            "forecast_months": forecast_months,
        })
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("aic", results.aic)
        mlflow.statsmodels.log_model(results, "sarimax_model")

    return run.info.run_id
