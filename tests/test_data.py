"""Tests for src/data.py"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src import data


class TestPreprocess:
    """Test data preprocessing."""

    def test_preprocess_drops_non_essential_columns(self, sample_raw_dataframe):
        """Preprocess should drop non-essential columns."""
        df = sample_raw_dataframe
        result = data.preprocess(df)

        for col in data.NON_ESSENTIAL_COLUMNS:
            assert col not in result.columns

    def test_preprocess_keeps_essential_columns(self, sample_raw_dataframe):
        """Preprocess should keep Order Date and Sales."""
        df = sample_raw_dataframe
        result = data.preprocess(df)

        assert 'Order Date' in result.columns
        assert 'Sales' in result.columns

    def test_preprocess_sorts_by_date(self, sample_raw_dataframe):
        """Preprocess should sort dataframe by Order Date."""
        df = sample_raw_dataframe
        result = data.preprocess(df)

        dates = result['Order Date'].values
        assert np.all(dates[:-1] <= dates[1:])

    def test_preprocess_returns_dataframe(self, sample_raw_dataframe):
        """Preprocess should return a DataFrame."""
        result = data.preprocess(sample_raw_dataframe)
        assert isinstance(result, pd.DataFrame)


class TestAggregateMonthly:
    """Test monthly aggregation."""

    def test_aggregate_monthly_returns_tuple(self, sample_daily_dataframe):
        """Aggregate monthly should return (series, max_date)."""
        y, max_date = data.aggregate_monthly(sample_daily_dataframe)
        assert isinstance(y, pd.Series)
        assert isinstance(max_date, pd.Timestamp)

    def test_aggregate_monthly_index_is_datetime(self, sample_daily_dataframe):
        """Aggregated series should have datetime index."""
        y, _ = data.aggregate_monthly(sample_daily_dataframe)
        assert isinstance(y.index, pd.DatetimeIndex)

    def test_aggregate_monthly_all_positive(self, sample_daily_dataframe):
        """Aggregated sales should be positive."""
        y, _ = data.aggregate_monthly(sample_daily_dataframe)
        assert (y >= 0).all()

    def test_aggregate_monthly_max_date_is_correct(self, sample_daily_dataframe):
        """Max date should match last date in aggregated series."""
        y, max_date = data.aggregate_monthly(sample_daily_dataframe)
        assert max_date == y.index.max()

    def test_aggregate_monthly_frequency(self, sample_daily_dataframe):
        """Aggregated series should have monthly frequency."""
        y, _ = data.aggregate_monthly(sample_daily_dataframe)
        assert y.index.inferred_freq == 'MS' or str(y.index.freq) == '<MonthBegin>'


class TestLoadRawData:
    """Test raw data loading."""

    @patch('src.data.pd.read_excel')
    def test_load_raw_data_calls_read_excel(self, mock_read):
        """Load raw data should call pd.read_excel."""
        mock_read.return_value = pd.DataFrame({'col': [1, 2, 3]})

        result = data.load_raw_data()

        mock_read.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    @patch('src.data.pd.read_excel')
    def test_load_raw_data_uses_correct_path(self, mock_read, sample_raw_dataframe):
        """Load raw data should use RAW_DATA_PATH."""
        mock_read.return_value = sample_raw_dataframe

        data.load_raw_data()

        called_path = mock_read.call_args[0][0]
        assert str(called_path).endswith('.xlsx')


class TestDownloadDemoDataset:
    """Test demo dataset download."""

    @patch('src.data.requests.get')
    @patch('src.data.RAW_DATA_PATH')
    @patch('src.data.RAW_DATA_DIR')
    def test_download_creates_directory(self, mock_dir, mock_path, mock_get):
        """Download should create data directory."""
        mock_dir.mkdir = MagicMock()
        mock_dir.exists.return_value = False
        mock_path.exists.return_value = False
        mock_get.return_value = MagicMock(content=b'test')

        data.download_demo_dataset()

        mock_dir.mkdir.assert_called_once()

    @patch('src.data.requests.get')
    @patch('src.data.RAW_DATA_PATH')
    @patch('src.data.RAW_DATA_DIR')
    def test_download_skips_if_exists(self, mock_dir, mock_path, mock_get):
        """Download should skip if file already exists."""
        mock_dir.mkdir = MagicMock()
        mock_path.exists.return_value = True
        mock_get.return_value = MagicMock(content=b'test')

        data.download_demo_dataset()

        mock_get.assert_not_called()

    @patch('src.data.requests.get')
    @patch('src.data.RAW_DATA_PATH')
    @patch('src.data.RAW_DATA_DIR')
    def test_download_writes_content(self, mock_dir, mock_path, mock_get):
        """Download should write response content to file."""
        mock_dir.mkdir = MagicMock()
        mock_path.exists.return_value = False
        mock_path.write_bytes = MagicMock()
        test_content = b'xlsx_content'
        mock_get.return_value = MagicMock(content=test_content)

        data.download_demo_dataset()

        mock_path.write_bytes.assert_called_once_with(test_content)


class TestNonEssentialColumns:
    """Test non-essential columns definition."""

    def test_non_essential_columns_is_list(self):
        """NON_ESSENTIAL_COLUMNS should be a list."""
        assert isinstance(data.NON_ESSENTIAL_COLUMNS, list)

    def test_non_essential_columns_not_empty(self):
        """NON_ESSENTIAL_COLUMNS should not be empty."""
        assert len(data.NON_ESSENTIAL_COLUMNS) > 0

    def test_essential_columns_excluded(self):
        """Essential columns should not be in NON_ESSENTIAL_COLUMNS."""
        assert 'Order Date' not in data.NON_ESSENTIAL_COLUMNS
        assert 'Sales' not in data.NON_ESSENTIAL_COLUMNS
