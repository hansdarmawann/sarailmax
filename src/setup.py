"""Local environment setup: directories and data file checks."""

from src.config import RAW_DATA_DIR, RAW_DATA_PATH, OUTPUT_DIR, MLRUNS_DIR


def ensure_directories() -> None:
    for directory in (RAW_DATA_DIR, OUTPUT_DIR, MLRUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def verify_data_file() -> bool:
    return RAW_DATA_PATH.exists()
