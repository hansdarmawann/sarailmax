"""Forecast generation and accuracy evaluation."""

import warnings

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


def forecast_future(results, horizon_months: int):
    """Forecast periods strictly after the end of the fitted series."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        pred = results.get_forecast(steps=horizon_months)
        return pred, pred.conf_int()


def forecast(results, anchor_date, horizon_months: int):
    """Backward-compatible forecast API used by the notebook.

    ``anchor_date`` is retained for compatibility with older notebook code;
    statsmodels derives the forecast start from the end of the fitted series.
    """
    del anchor_date
    return forecast_future(results, horizon_months)


def validate(results, anchor_date, validation_months: int):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        return results.get_prediction(
            start=anchor_date - pd.DateOffset(months=validation_months - 1),
            end=anchor_date,
            dynamic=False
        )


def compute_mape(y_true: pd.Series, predictions) -> float:
    y_true = y_true.reindex(predictions.predicted_mean.index)
    y_pred = predictions.predicted_mean
    return mean_absolute_percentage_error(y_true.squeeze(), y_pred) * 100
