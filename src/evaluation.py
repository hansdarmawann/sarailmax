"""Forecast generation and accuracy evaluation."""

import warnings

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


def forecast(results, anchor_date, horizon_months: int):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        pred = results.get_prediction(
            start=anchor_date,
            end=anchor_date + pd.DateOffset(months=horizon_months),
            dynamic=False
        )
        return pred, pred.conf_int()


def validate(results, anchor_date, validation_months: int):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        return results.get_prediction(
            start=anchor_date - pd.DateOffset(months=validation_months - 1),
            end=anchor_date,
            dynamic=False
        )


def compute_mape(y: pd.Series, predictions) -> float:
    y_true = y.loc[predictions.predicted_mean.index]
    y_pred = predictions.predicted_mean
    return mean_absolute_percentage_error(y_true.squeeze(), y_pred) * 100
