"""Tests for src/setup.py"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src import setup


class TestEnsureDirectories:
    """Test directory creation."""

    @patch('src.setup.RAW_DATA_DIR')
    @patch('src.setup.OUTPUT_DIR')
    @patch('src.setup.MLRUNS_DIR')
    def test_ensure_directories_creates_all(self, mock_mlruns, mock_output, mock_raw):
        """Ensure directories should create all required directories."""
        mock_raw.mkdir = MagicMock()
        mock_output.mkdir = MagicMock()
        mock_mlruns.mkdir = MagicMock()

        setup.ensure_directories()

        mock_raw.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_output.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_mlruns.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('src.setup.RAW_DATA_DIR')
    @patch('src.setup.OUTPUT_DIR')
    @patch('src.setup.MLRUNS_DIR')
    def test_ensure_directories_called_with_correct_args(self, mock_mlruns, mock_output, mock_raw):
        """Directories should be created with parents=True, exist_ok=True."""
        setup.ensure_directories()

        for mock_dir in [mock_raw, mock_output, mock_mlruns]:
            mock_dir.mkdir.assert_called_once()
            call_kwargs = mock_dir.mkdir.call_args[1]
            assert call_kwargs['parents'] is True
            assert call_kwargs['exist_ok'] is True


class TestVerifyDataFile:
    """Test data file verification."""

    @patch('src.setup.RAW_DATA_PATH')
    def test_verify_data_file_returns_bool(self, mock_path):
        """Verify data file should return boolean."""
        mock_path.exists.return_value = True

        result = setup.verify_data_file()

        assert isinstance(result, bool)

    @patch('src.setup.RAW_DATA_PATH')
    def test_verify_data_file_true_when_exists(self, mock_path):
        """Should return True when file exists."""
        mock_path.exists.return_value = True

        result = setup.verify_data_file()

        assert result is True

    @patch('src.setup.RAW_DATA_PATH')
    def test_verify_data_file_false_when_not_exists(self, mock_path):
        """Should return False when file doesn't exist."""
        mock_path.exists.return_value = False

        result = setup.verify_data_file()

        assert result is False
