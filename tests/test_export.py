"""Tests for src/export.py"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from src import export, evaluation, modeling


class TestBuildOutputFrame:
    """Test output dataframe building."""

    def test_build_output_frame_returns_dataframe(self, sample_monthly_series):
        """Build output frame should return a DataFrame."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        pred, pred_ci = evaluation.forecast_future(results, 3)
        output = export.build_output_frame(y, pred, pred_ci, 25.0)

        assert isinstance(output, pd.DataFrame)

    def test_build_output_frame_has_required_columns(self, sample_monthly_series):
        """Output should have all required columns."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        pred, pred_ci = evaluation.forecast_future(results, 3)
        output = export.build_output_frame(y, pred, pred_ci, 25.0)

        required_columns = {
            'periode', 'sales', 'forecast_sales', 'lower_ci',
            'upper_ci', 'mape', 'type'
        }
        assert required_columns.issubset(set(output.columns))

    def test_build_output_frame_correct_length(self, sample_monthly_series):
        """Output should have actual + forecast rows."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        forecast_months = 3
        pred, pred_ci = evaluation.forecast_future(results, forecast_months)
        output = export.build_output_frame(y, pred, pred_ci, 25.0)

        expected_length = len(y) + forecast_months
        assert len(output) == expected_length

    def test_build_output_frame_actual_has_sales(self, sample_monthly_series):
        """Actual rows should have sales values."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        pred, pred_ci = evaluation.forecast_future(results, 3)
        output = export.build_output_frame(y, pred, pred_ci, 25.0)

        actual_rows = output[output['type'] == 'actual']
        assert actual_rows['sales'].notna().all()
        assert actual_rows['forecast_sales'].isna().all()

    def test_build_output_frame_forecast_has_forecast(self, sample_monthly_series):
        """Forecast rows should have forecast_sales values."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        pred, pred_ci = evaluation.forecast_future(results, 3)
        output = export.build_output_frame(y, pred, pred_ci, 25.0)

        forecast_rows = output[output['type'] == 'forecast']
        assert forecast_rows['forecast_sales'].notna().all()
        assert forecast_rows['sales'].isna().all()

    def test_build_output_frame_sorted_by_date(self, sample_monthly_series):
        """Output should be sorted by periode."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        pred, pred_ci = evaluation.forecast_future(results, 3)
        output = export.build_output_frame(y, pred, pred_ci, 25.0)

        dates = pd.to_datetime(output['periode'])
        assert (dates.iloc[:-1].values <= dates.iloc[1:].values).all()

    def test_build_output_frame_mape_constant(self, sample_monthly_series):
        """All rows should have same MAPE value."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        mape_value = 25.5
        pred, pred_ci = evaluation.forecast_future(results, 3)
        output = export.build_output_frame(y, pred, pred_ci, mape_value)

        assert (output['mape'] == mape_value).all()

    def test_build_output_frame_ci_bounds(self, sample_monthly_series):
        """Lower CI should be less than upper CI."""
        y = sample_monthly_series[:30]
        order = (0, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = modeling.train_sarimax(y, order, seasonal_order)

        pred, pred_ci = evaluation.forecast_future(results, 3)
        output = export.build_output_frame(y, pred, pred_ci, 25.0)

        forecast_rows = output[output['type'] == 'forecast']
        assert (forecast_rows['lower_ci'] <= forecast_rows['upper_ci']).all()


class TestSaveCsv:
    """Test CSV saving."""

    def test_save_csv_creates_file(self, tmp_path):
        """Save CSV should create output file."""
        df = pd.DataFrame({
            'periode': pd.date_range('2020-01-01', periods=3),
            'sales': [100, 200, 300],
            'forecast_sales': [np.nan, np.nan, np.nan],
            'type': ['actual', 'actual', 'actual']
        })

        output_path = tmp_path / "test_output.csv"

        with patch('src.export.OUTPUT_DIR', tmp_path):
            with patch('src.export.OUTPUT_PATH', output_path):
                export.save_csv(df, output_path)

        assert output_path.exists()

    def test_save_csv_output_is_valid_csv(self, tmp_path):
        """Saved file should be valid CSV."""
        df = pd.DataFrame({
            'periode': pd.date_range('2020-01-01', periods=3),
            'sales': [100, 200, 300],
            'forecast_sales': [np.nan, np.nan, np.nan],
            'type': ['actual', 'actual', 'actual']
        })

        output_path = tmp_path / "test_output.csv"

        with patch('src.export.OUTPUT_DIR', tmp_path):
            export.save_csv(df, output_path)

        # Read back and verify
        df_read = pd.read_csv(output_path)
        assert len(df_read) == len(df)
        assert list(df_read.columns) == list(df.columns)

    def test_save_csv_preserves_data(self, tmp_path):
        """Saved CSV should preserve all data."""
        df = pd.DataFrame({
            'periode': pd.date_range('2020-01-01', periods=3),
            'sales': [100.5, 200.7, 300.2],
            'forecast_sales': [105.0, 210.0, 320.0],
            'type': ['actual', 'actual', 'forecast']
        })

        output_path = tmp_path / "test_output.csv"

        with patch('src.export.OUTPUT_DIR', tmp_path):
            export.save_csv(df, output_path)

        df_read = pd.read_csv(output_path)
        assert df_read['sales'].iloc[0] == pytest.approx(100.5)
        assert df_read['forecast_sales'].iloc[2] == pytest.approx(320.0)
