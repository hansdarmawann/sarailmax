# Testing Guide

This project uses [pytest](https://pytest.org/) for unit and integration testing.

## Installation

Tests require additional dependencies beyond the base `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` already includes `pytest>=7.0` and `pytest-cov>=4.0`.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html
```

This generates a coverage report in `htmlcov/index.html`.

### Run specific test file
```bash
pytest tests/test_data.py
```

### Run specific test class
```bash
pytest tests/test_pipeline.py::TestPipelineRun
```

### Run specific test function
```bash
pytest tests/test_modeling.py::TestGridSearch::test_grid_search_returns_dataframe
```

### Run tests with verbose output
```bash
pytest -v
```

### Run tests matching a pattern
```bash
pytest -k "forecast" -v
```

## Test Structure

Tests are organized by module:

- `test_config.py` — Configuration constants and path handling
- `test_data.py` — Data loading, preprocessing, and aggregation
- `test_modeling.py` — SARIMAX model selection and training
- `test_evaluation.py` — Forecast generation and accuracy metrics
- `test_export.py` — Output file generation and CSV handling
- `test_setup.py` — Directory and file verification
- `test_pipeline.py` — End-to-end pipeline integration tests

## Test Fixtures

Common test data is defined in `conftest.py`:

- `sample_monthly_series` — 48-month time series with trend and seasonality
- `sample_daily_dataframe` — Daily sales transaction data
- `sample_raw_dataframe` — Raw Superstore-like data
- `tmp_project_root` — Temporary project directory structure

## Coverage

The test suite aims for high coverage of critical paths:

- **Data Pipeline** — Loading, cleaning, aggregating data
- **Model Training** — Grid search, parameter selection, SARIMAX fitting
- **Evaluation** — Forecast generation, MAPE calculation, validation
- **Export** — Output formatting and CSV saving
- **Pipeline** — End-to-end leakage-free workflow

To check coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

## CI/CD Integration

Add to your CI pipeline (GitHub Actions, GitLab CI, etc.):

```bash
pytest --cov=src --cov-report=xml
```

## Notes

- Tests use mocking to avoid dependency on external APIs and data files
- Integration tests use realistic sample data to validate the full pipeline
- No real Superstore.xlsx file is required to run tests
- Tests are designed to be fast (< 2 seconds total)

## Troubleshooting

**Import errors:**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest
```

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

**Test discovery issues:**
```bash
pytest --collect-only
```
