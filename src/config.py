"""Project-wide constants and paths, resolved relative to this file so they
stay correct regardless of the notebook's working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "Superstore.xlsx"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_PATH = OUTPUT_DIR / "sales_actual_forecast.csv"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"

IS_CUSTOM_DATA = False
EXPERIMENT_NAME = "Superstore_Sales_Forecast"
VALIDATION_MONTHS = 12
FORECAST_MONTHS = 12

DEMO_DATA_URL = "https://synapseaisolutionsa.z13.web.core.windows.net/data/Forecast_Superstore_Sales"


def relative_path(absolute_path: Path) -> str:
    """Convert absolute path to relative path from PROJECT_ROOT for display."""
    try:
        return str(absolute_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(absolute_path)
