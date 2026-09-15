"""Shared fixtures for tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


@pytest.fixture
def sample_monthly_series():
    """Create a sample monthly time series with trend and seasonality."""
    dates = pd.date_range('2014-01-01', periods=48, freq='MS')

    trend = np.linspace(10000, 15000, 48)
    seasonal = 3000 * np.sin(np.arange(48) * 2 * np.pi / 12)
    noise = np.random.normal(0, 500, 48)

    values = trend + seasonal + noise
    values = np.maximum(values, 1000)

    return pd.Series(values, index=dates, name='Sales')


@pytest.fixture
def sample_daily_dataframe():
    """Create a sample daily sales dataframe."""
    dates = pd.date_range('2014-01-01', periods=365, freq='D')

    np.random.seed(42)
    sales = np.random.uniform(5000, 20000, 365)

    data = {
        'Order Date': dates,
        'Sales': sales,
        'Row ID': range(365),
        'Customer ID': ['CUST' + str(i % 100) for i in range(365)],
        'City': ['New York'] * 365,
        'Region': ['North'] * 365,
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_raw_dataframe():
    """Create a sample raw Superstore-like dataframe."""
    dates = pd.date_range('2014-01-01', periods=100, freq='D')

    np.random.seed(42)
    data = {
        'Row ID': range(100),
        'Order ID': ['ORD-' + str(i) for i in range(100)],
        'Order Date': dates,
        'Ship Date': dates + timedelta(days=3),
        'Ship Mode': ['Standard Class'] * 100,
        'Customer ID': ['CUST-' + str(i % 20) for i in range(100)],
        'Customer Name': ['Customer'] * 100,
        'Segment': ['Consumer'] * 100,
        'Country': ['United States'] * 100,
        'City': ['New York'] * 100,
        'State': ['NY'] * 100,
        'Postal Code': [10001] * 100,
        'Region': ['North'] * 100,
        'Product ID': ['PROD-' + str(i % 10) for i in range(100)],
        'Category': ['Office Supplies'] * 100,
        'Sub-Category': ['Paper'] * 100,
        'Product Name': ['Product'] * 100,
        'Sales': np.random.uniform(100, 5000, 100),
        'Quantity': np.random.randint(1, 10, 100),
        'Discount': np.random.uniform(0, 0.2, 100),
        'Profit': np.random.uniform(-100, 1000, 100),
    }

    return pd.DataFrame(data)


@pytest.fixture
def tmp_project_root(tmp_path):
    """Create a temporary project structure."""
    root = tmp_path / "test_project"
    root.mkdir()

    (root / "data" / "raw").mkdir(parents=True)
    (root / "output").mkdir(parents=True)
    (root / "mlruns").mkdir(parents=True)

    return root
