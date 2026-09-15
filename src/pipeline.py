"""Leakage-free training, validation, and future forecasting pipeline."""

from src import data, evaluation, modeling
from src.config import FORECAST_MONTHS, VALIDATION_MONTHS


def run(y, validation_months=VALIDATION_MONTHS, forecast_months=FORECAST_MONTHS):
    """Evaluate on a time-ordered holdout, then retrain and forecast.

    Model selection and parameter estimation use only the training portion.
    The validation portion is used once for an honest out-of-sample score.
    """
    if len(y) <= validation_months:
        raise ValueError("The series must be longer than the validation horizon.")

    train = y.iloc[:-validation_months].copy()
    validation = y.iloc[-validation_months:].copy()

    d, D, diagnostics = modeling.determine_differencing(train)
    aic_df = modeling.grid_search(train, d, D)
    order, seasonal_order = modeling.select_best(aic_df)

    validation_model = modeling.train_sarimax(train, order, seasonal_order)
    validation_prediction = validation_model.get_forecast(steps=validation_months)
    mape = evaluation.compute_mape(validation, validation_prediction)

    final_model = modeling.train_sarimax(y, order, seasonal_order)
    future_prediction, future_ci = evaluation.forecast_future(
        final_model, forecast_months
    )

    return {
        "train": train,
        "validation": validation,
        "order": order,
        "seasonal_order": seasonal_order,
        "diagnostics": diagnostics,
        "aic": aic_df,
        "validation_prediction": validation_prediction,
        "mape": mape,
        "results": final_model,
        "prediction": future_prediction,
        "prediction_ci": future_ci,
    }
