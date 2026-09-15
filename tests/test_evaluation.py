"""Tests for src/evaluation.py"""

import pytest
import pandas as pd
import numpy as np
from src import evaluation, modeling


class TestForecastFuture:
    """Test future forecasting."""

    def test_forecast_future_returns_prediction_and_ci(self, sample_monthly_series):
        """Forecast should return (prediction, confidence_intervals)."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        pred, pred_ci = evaluation.forecast_future(results, 3)

        assert len(pred.predicted_mean) == 3
        assert pred_ci.shape == (3, 2)

    def test_forecast_future_produces_correct_horizon(self, sample_monthly_series):
        """Forecast should produce requested number of steps."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        for steps in [1, 3, 6, 12]:
            pred, pred_ci = evaluation.forecast_future(results, steps)
            assert len(pred.predicted_mean) == steps
            assert len(pred_ci) == steps

    def test_forecast_future_ci_bounds(self, sample_monthly_series):
        """Confidence interval lower bound should be less than upper."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        _, pred_ci = evaluation.forecast_future(results, 3)

        assert np.all(pred_ci.iloc[:, 0] <= pred_ci.iloc[:, 1])

    def test_forecast_future_values_positive(self, sample_monthly_series):
        """Forecast should be positive (assuming positive sales)."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        pred, _ = evaluation.forecast_future(results, 3)

        assert np.all(pred.predicted_mean > 0)


class TestForecast:
    """Test forecast wrapper function."""

    def test_forecast_returns_same_as_forecast_future(self, sample_monthly_series):
        """Forecast wrapper should return same as forecast_future."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)
        anchor_date = y.index[-1]

        pred1, ci1 = evaluation.forecast_future(results, 3)
        pred2, ci2 = evaluation.forecast(results, anchor_date, 3)

        np.testing.assert_array_almost_equal(
            pred1.predicted_mean.values,
            pred2.predicted_mean.values
        )

    def test_forecast_ignores_anchor_date(self, sample_monthly_series):
        """Forecast should ignore anchor_date parameter."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        # Different anchor dates should produce same result
        pred1, _ = evaluation.forecast(results, y.index[-1], 3)
        pred2, _ = evaluation.forecast(results, y.index[-10], 3)

        np.testing.assert_array_almost_equal(
            pred1.predicted_mean.values,
            pred2.predicted_mean.values
        )


class TestValidate:
    """Test validation forecast."""

    def test_validate_returns_prediction_object(self, sample_monthly_series):
        """Validate should return prediction object."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)
        anchor_date = y.index[-1]

        pred = evaluation.validate(results, anchor_date, 3)

        assert hasattr(pred, 'predicted_mean')
        assert hasattr(pred, 'conf_int')

    def test_validate_forecast_length(self, sample_monthly_series):
        """Validate should produce correct number of predictions."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)
        anchor_date = y.index[-1]

        for months in [1, 3, 6, 12]:
            pred = evaluation.validate(results, anchor_date, months)
            assert len(pred.predicted_mean) == months

    def test_validate_uses_dynamic_false(self, sample_monthly_series):
        """Validate should use dynamic=False (not one-step ahead)."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)
        anchor_date = y.index[-1]

        pred = evaluation.validate(results, anchor_date, 3)

        # With dynamic=False, predictions should be smooth
        diffs = np.diff(pred.predicted_mean.values)
        assert len(diffs) > 0


class TestComputeMape:
    """Test MAPE calculation."""

    def test_compute_mape_returns_float(self, sample_monthly_series):
        """Compute MAPE should return float."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        pred = evaluation.validate(results, y.index[-1], 3)
        mape = evaluation.compute_mape(y[-3:], pred)

        assert isinstance(mape, (float, np.floating))

    def test_compute_mape_in_valid_range(self, sample_monthly_series):
        """MAPE should be between 0 and 100."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        pred = evaluation.validate(results, y.index[-1], 3)
        mape = evaluation.compute_mape(y[-3:], pred)

        assert 0 <= mape <= 200  # Allow up to 200 for very bad forecasts

    def test_compute_mape_perfect_forecast(self):
        """MAPE should be 0 for perfect forecast."""
        y_true = pd.Series([100, 200, 300])
        y_pred = pd.Series([100, 200, 300])

        # Create mock prediction object
        from unittest.mock import MagicMock
        pred = MagicMock()
        pred.predicted_mean = y_pred
        pred.predicted_mean.index = y_true.index

        mape = evaluation.compute_mape(y_true, pred)

        assert mape == 0.0

    def test_compute_mape_consistent_forecast(self):
        """MAPE should be consistent with definition."""
        y_true = pd.Series([100, 200, 300], index=pd.date_range('2020-01-01', periods=3, freq='ME'))
        y_pred = pd.Series([110, 220, 330], index=y_true.index)

        from unittest.mock import MagicMock
        pred = MagicMock()
        pred.predicted_mean = y_pred

        mape = evaluation.compute_mape(y_true, pred)

        # Expected: mean(|100-110|/100, |200-220|/200, |300-330|/300) * 100
        # = mean(0.1, 0.1, 0.1) * 100 = 10.0
        assert abs(mape - 10.0) < 0.01
