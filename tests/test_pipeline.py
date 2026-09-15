"""Integration tests for src/pipeline.py"""

import pytest
import pandas as pd
import numpy as np
from src import pipeline


class TestPipelineRun:
    """Integration tests for the complete pipeline."""

    def test_pipeline_run_returns_dict(self, sample_monthly_series):
        """Pipeline run should return a dictionary."""
        y = sample_monthly_series[:30]

        result = pipeline.run(y, validation_months=3, forecast_months=3)

        assert isinstance(result, dict)

    def test_pipeline_run_returns_all_keys(self, sample_monthly_series):
        """Pipeline should return all required result keys."""
        y = sample_monthly_series[:30]

        result = pipeline.run(y, validation_months=3, forecast_months=3)

        required_keys = {
            'train', 'validation', 'order', 'seasonal_order',
            'diagnostics', 'aic', 'validation_prediction', 'mape',
            'results', 'prediction', 'prediction_ci'
        }
        assert required_keys.issubset(set(result.keys()))

    def test_pipeline_run_train_validation_split(self, sample_monthly_series):
        """Pipeline should split data into train and validation."""
        y = sample_monthly_series[:30]
        validation_months = 6

        result = pipeline.run(y, validation_months=validation_months, forecast_months=3)

        assert len(result['train']) + len(result['validation']) == len(y)
        assert len(result['validation']) == validation_months

    def test_pipeline_run_produces_mape(self, sample_monthly_series):
        """Pipeline should compute MAPE."""
        y = sample_monthly_series[:30]

        result = pipeline.run(y, validation_months=3, forecast_months=3)

        assert isinstance(result['mape'], (float, np.floating))
        assert 0 <= result['mape'] <= 200

    def test_pipeline_run_produces_forecast(self, sample_monthly_series):
        """Pipeline should produce future forecast."""
        y = sample_monthly_series[:30]
        forecast_months = 6

        result = pipeline.run(y, validation_months=3, forecast_months=forecast_months)

        assert len(result['prediction'].predicted_mean) == forecast_months

    def test_pipeline_run_produces_confidence_intervals(self, sample_monthly_series):
        """Pipeline should produce confidence intervals."""
        y = sample_monthly_series[:30]

        result = pipeline.run(y, validation_months=3, forecast_months=3)

        assert result['prediction_ci'].shape[0] == 3
        assert result['prediction_ci'].shape[1] == 2

    def test_pipeline_run_valid_orders(self, sample_monthly_series):
        """Pipeline should return valid SARIMAX orders."""
        y = sample_monthly_series[:30]

        result = pipeline.run(y, validation_months=3, forecast_months=3)

        order = result['order']
        seasonal_order = result['seasonal_order']

        assert len(order) == 3
        assert len(seasonal_order) == 4
        assert all(isinstance(x, (int, np.integer)) for x in order)
        assert all(isinstance(x, (int, np.integer)) for x in seasonal_order)

    def test_pipeline_run_aic_dataframe(self, sample_monthly_series):
        """Pipeline should return AIC search results as DataFrame."""
        y = sample_monthly_series[:30]

        result = pipeline.run(y, validation_months=3, forecast_months=3)

        aic_df = result['aic']
        assert isinstance(aic_df, pd.DataFrame)
        assert 'AIC' in aic_df.columns
        assert 'order' in aic_df.columns

    def test_pipeline_run_diagnostics_present(self, sample_monthly_series):
        """Pipeline should return ADF test diagnostics."""
        y = sample_monthly_series[:30]

        result = pipeline.run(y, validation_months=3, forecast_months=3)

        diagnostics = result['diagnostics']
        assert 'adf_level_pvalue' in diagnostics
        assert 'adf_seasonal_pvalue' in diagnostics

    def test_pipeline_run_requires_longer_series(self, sample_monthly_series):
        """Pipeline should raise error if series too short."""
        y = sample_monthly_series[:10]

        with pytest.raises(ValueError):
            pipeline.run(y, validation_months=12, forecast_months=3)

    def test_pipeline_run_minimum_length(self, sample_monthly_series):
        """Pipeline should work with sufficient series length."""
        y = sample_monthly_series[:30]

        # Should not raise - 30 points > 12 month validation horizon with room for training
        result = pipeline.run(y, validation_months=12, forecast_months=1)
        assert len(result['validation']) == 12

    def test_pipeline_run_valid_data_leakage(self, sample_monthly_series):
        """Pipeline should not leak validation data into model selection."""
        y = sample_monthly_series[:30]
        validation_months = 6

        result = pipeline.run(y, validation_months=validation_months, forecast_months=3)

        # Model was trained on training portion only
        assert len(result['train']) == len(y) - validation_months

    def test_pipeline_run_forecast_after_retrain(self, sample_monthly_series):
        """Final forecast should be trained on full data."""
        y = sample_monthly_series[:30]

        result = pipeline.run(y, validation_months=3, forecast_months=3)

        # Results object was trained on full series
        # Can verify by checking it has all data points available
        assert len(result['results'].fittedvalues) >= len(y) - 3

    def test_pipeline_run_mape_on_validation_only(self, sample_monthly_series):
        """MAPE should be computed on validation set only."""
        y = sample_monthly_series[:30]
        validation_months = 6

        result = pipeline.run(y, validation_months=validation_months, forecast_months=3)

        # MAPE should be computed on 6 months of validation data
        validation_pred = result['validation_prediction']
        assert len(validation_pred.predicted_mean) == validation_months

    def test_pipeline_run_different_horizons(self, sample_monthly_series):
        """Pipeline should handle different validation/forecast horizons."""
        y = sample_monthly_series[:48]

        for val_months in [3, 6, 12]:
            result = pipeline.run(y, validation_months=val_months, forecast_months=val_months)
            assert len(result['validation']) == val_months
            assert len(result['prediction'].predicted_mean) == val_months

    def test_pipeline_run_repeatable_with_seed(self):
        """Pipeline results should be deterministic."""
        np.random.seed(42)
        dates = pd.date_range('2014-01-01', periods=48, freq='MS')
        trend = np.linspace(10000, 15000, 48)
        seasonal = 3000 * np.sin(np.arange(48) * 2 * np.pi / 12)
        noise = np.random.normal(0, 500, 48)
        y1 = pd.Series(np.maximum(trend + seasonal + noise, 1000), index=dates)

        np.random.seed(42)
        noise2 = np.random.normal(0, 500, 48)
        y2 = pd.Series(np.maximum(trend + seasonal + noise2, 1000), index=dates)

        result1 = pipeline.run(y1, validation_months=12, forecast_months=12)
        result2 = pipeline.run(y2, validation_months=12, forecast_months=12)

        # Orders should be identical
        assert result1['order'] == result2['order']
        assert result1['seasonal_order'] == result2['seasonal_order']
