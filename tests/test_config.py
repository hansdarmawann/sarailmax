"""Tests for src/config.py"""

import pytest
from pathlib import Path
from src import config


class TestConfig:
    """Test configuration constants and functions."""

    def test_project_root_exists(self):
        """PROJECT_ROOT should point to the project directory."""
        assert config.PROJECT_ROOT.exists()
        assert config.PROJECT_ROOT.is_dir()

    def test_paths_are_relative_to_root(self):
        """All paths should be relative to PROJECT_ROOT."""
        assert config.RAW_DATA_DIR.parent == config.DATA_DIR
        assert config.DATA_DIR.parent == config.PROJECT_ROOT
        assert config.OUTPUT_DIR.parent == config.PROJECT_ROOT
        assert config.MLRUNS_DIR.parent == config.PROJECT_ROOT

    def test_constants_are_positive(self):
        """Validation and forecast months should be positive."""
        assert config.VALIDATION_MONTHS > 0
        assert config.FORECAST_MONTHS > 0

    def test_experiment_name_is_string(self):
        """Experiment name should be a non-empty string."""
        assert isinstance(config.EXPERIMENT_NAME, str)
        assert len(config.EXPERIMENT_NAME) > 0

    def test_is_custom_data_is_bool(self):
        """IS_CUSTOM_DATA should be a boolean."""
        assert isinstance(config.IS_CUSTOM_DATA, bool)

    def test_demo_data_url_is_valid(self):
        """Demo data URL should start with https."""
        assert config.DEMO_DATA_URL.startswith("https://")

    def test_relative_path_converts_absolute_to_relative(self):
        """relative_path should convert absolute path to relative from PROJECT_ROOT."""
        test_path = config.PROJECT_ROOT / "src" / "data.py"
        result = config.relative_path(test_path)
        assert "src" in result
        assert "data.py" in result

    def test_relative_path_handles_external_path(self):
        """relative_path should return string for paths outside PROJECT_ROOT."""
        external_path = Path("/tmp/external/path")
        result = config.relative_path(external_path)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_raw_data_path_extension(self):
        """Raw data path should point to Excel file."""
        assert config.RAW_DATA_PATH.suffix == ".xlsx"

    def test_output_path_extension(self):
        """Output path should be CSV file."""
        assert config.OUTPUT_PATH.suffix == ".csv"
