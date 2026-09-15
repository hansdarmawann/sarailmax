"""Tests for src/modeling.py"""

import pytest
import pandas as pd
import numpy as np
from src import modeling


class TestDetermineDifferencing:
    """Test ADF stationarity tests and differencing determination."""

    def test_determine_differencing_returns_tuple(self, sample_monthly_series):
        """Determine differencing should return (d, D, diagnostics)."""
        d, D, diagnostics = modeling.determine_differencing(sample_monthly_series)

        assert isinstance(d, (int, np.integer))
        assert isinstance(D, (int, np.integer))
        assert isinstance(diagnostics, dict)

    def test_determine_differencing_orders_are_valid(self, sample_monthly_series):
        """d and D should be 0 or 1."""
        d, D, _ = modeling.determine_differencing(sample_monthly_series)

        assert d in [0, 1]
        assert D in [0, 1]

    def test_determine_differencing_diagnostics_keys(self, sample_monthly_series):
        """Diagnostics should contain ADF statistics and p-values."""
        _, _, diagnostics = modeling.determine_differencing(sample_monthly_series)

        expected_keys = {
            'adf_level_statistic',
            'adf_level_pvalue',
            'adf_seasonal_statistic',
            'adf_seasonal_pvalue',
        }
        assert set(diagnostics.keys()) == expected_keys

    def test_determine_differencing_pvalues_valid_range(self, sample_monthly_series):
        """ADF p-values should be between 0 and 1."""
        _, _, diagnostics = modeling.determine_differencing(sample_monthly_series)

        assert 0 <= diagnostics['adf_level_pvalue'] <= 1
        assert 0 <= diagnostics['adf_seasonal_pvalue'] <= 1

    def test_determine_differencing_seasonal_period(self):
        """SEASONAL_PERIOD should be 12 for monthly data."""
        assert modeling.SEASONAL_PERIOD == 12


class TestGridSearch:
    """Test SARIMAX grid search."""

    def test_grid_search_returns_dataframe(self, sample_monthly_series):
        """Grid search should return a DataFrame."""
        y = sample_monthly_series[:30]
        d, D, _ = modeling.determine_differencing(y)

        result = modeling.grid_search(y, d, D, p_range=range(0, 2), q_range=range(0, 2))

        assert isinstance(result, pd.DataFrame)

    def test_grid_search_has_required_columns(self, sample_monthly_series):
        """Result should have order, seasonal_order, AIC columns."""
        y = sample_monthly_series[:30]
        d, D, _ = modeling.determine_differencing(y)

        result = modeling.grid_search(y, d, D, p_range=range(0, 2), q_range=range(0, 2))

        assert 'order' in result.columns
        assert 'seasonal_order' in result.columns
        assert 'AIC' in result.columns

    def test_grid_search_sorted_by_aic(self, sample_monthly_series):
        """Result should be sorted by AIC ascending."""
        y = sample_monthly_series[:30]
        d, D, _ = modeling.determine_differencing(y)

        result = modeling.grid_search(y, d, D, p_range=range(0, 2), q_range=range(0, 2))

        aic_values = result['AIC'].values
        assert np.all(aic_values[:-1] <= aic_values[1:])

    def test_grid_search_respects_ranges(self, sample_monthly_series):
        """Grid search should respect provided p and q ranges."""
        y = sample_monthly_series[:30]
        d, D, _ = modeling.determine_differencing(y)
        p_range = range(0, 2)
        q_range = range(0, 2)

        result = modeling.grid_search(y, d, D, p_range=p_range, q_range=q_range)

        for order in result['order']:
            p, d_val, q = order
            assert p in p_range
            assert q in q_range

    def test_grid_search_aic_values_positive(self, sample_monthly_series):
        """AIC values should be valid numbers."""
        y = sample_monthly_series[:30]
        d, D, _ = modeling.determine_differencing(y)

        result = modeling.grid_search(y, d, D, p_range=range(0, 2), q_range=range(0, 2))

        assert all(pd.notna(result['AIC']))
        assert all(np.isfinite(result['AIC']))

    def test_grid_search_not_empty(self, sample_monthly_series):
        """Grid search should return at least one result."""
        y = sample_monthly_series[:30]
        d, D, _ = modeling.determine_differencing(y)

        result = modeling.grid_search(y, d, D)

        assert len(result) > 0


class TestSelectBest:
    """Test best model selection."""

    def test_select_best_returns_tuple_of_tuples(self, sample_monthly_series):
        """Select best should return (order, seasonal_order)."""
        y = sample_monthly_series[:30]
        d, D, _ = modeling.determine_differencing(y)
        aic_df = modeling.grid_search(y, d, D, p_range=range(0, 2), q_range=range(0, 2))

        order, seasonal_order = modeling.select_best(aic_df)

        assert isinstance(order, tuple)
        assert isinstance(seasonal_order, tuple)

    def test_select_best_picks_lowest_aic(self, sample_monthly_series):
        """Select best should pick row with lowest AIC."""
        y = sample_monthly_series[:30]
        d, D, _ = modeling.determine_differencing(y)
        aic_df = modeling.grid_search(y, d, D, p_range=range(0, 2), q_range=range(0, 2))

        order, seasonal_order = modeling.select_best(aic_df)
        best_row = aic_df.iloc[0]

        assert order == best_row['order']
        assert seasonal_order == best_row['seasonal_order']

    def test_select_best_order_has_three_elements(self, sample_monthly_series):
        """Order should have (p, d, q)."""
        y = sample_monthly_series[:30]
        d, D, _ = modeling.determine_differencing(y)
        aic_df = modeling.grid_search(y, d, D, p_range=range(0, 2), q_range=range(0, 2))

        order, _ = modeling.select_best(aic_df)

        assert len(order) == 3

    def test_select_best_seasonal_order_has_four_elements(self, sample_monthly_series):
        """Seasonal order should have (P, D, Q, s)."""
        y = sample_monthly_series[:30]
        d, D, _ = modeling.determine_differencing(y)
        aic_df = modeling.grid_search(y, d, D, p_range=range(0, 2), q_range=range(0, 2))

        _, seasonal_order = modeling.select_best(aic_df)

        assert len(seasonal_order) == 4


class TestTrainSarimax:
    """Test SARIMAX model training."""

    def test_train_sarimax_returns_results(self, sample_monthly_series):
        """Train should return statsmodels SARIMAX results object."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)

        results = modeling.train_sarimax(y, order, seasonal_order)

        assert hasattr(results, 'aic')
        assert hasattr(results, 'predict')
        assert hasattr(results, 'get_forecast')

    def test_train_sarimax_aic_is_valid(self, sample_monthly_series):
        """Trained model should have valid AIC."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)

        results = modeling.train_sarimax(y, order, seasonal_order)

        assert pd.notna(results.aic)
        assert np.isfinite(results.aic)

    def test_train_sarimax_can_forecast(self, sample_monthly_series):
        """Trained model should be able to forecast."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)

        results = modeling.train_sarimax(y, order, seasonal_order)
        forecast = results.get_forecast(steps=3)

        assert len(forecast.predicted_mean) == 3
