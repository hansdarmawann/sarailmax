"""Loading and preprocessing the Superstore sales data."""

import requests
import pandas as pd

from src.config import DEMO_DATA_URL, RAW_DATA_DIR, RAW_DATA_PATH

NON_ESSENTIAL_COLUMNS = [
    'Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name',
    'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID',
    'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit'
]


def download_demo_dataset() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DATA_PATH.exists():
        response = requests.get(f"{DEMO_DATA_URL}/Superstore.xlsx", timeout=30)
        RAW_DATA_PATH.write_bytes(response.content)


def load_raw_data() -> pd.DataFrame:
    return pd.read_excel(RAW_DATA_PATH)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=NON_ESSENTIAL_COLUMNS)
    return df.sort_values('Order Date')


def aggregate_monthly(df: pd.DataFrame):
    df = df.groupby('Order Date')['Sales'].sum().reset_index()
    df = df.set_index('Order Date')

    y = df['Sales'].resample('MS').sum()
    y = y.reset_index()
    y['Order Date'] = pd.to_datetime(y['Order Date'])
    y = y.set_index('Order Date')['Sales']

    maximum_date = y.index.max()
    return y, maximum_date
