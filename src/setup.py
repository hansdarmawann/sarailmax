"""Local environment setup: directories and data file checks."""

from src.config import RAW_DATA_DIR, RAW_DATA_PATH, OUTPUT_DIR, MLRUNS_DIR, relative_path


def ensure_directories() -> None:
    for directory in (RAW_DATA_DIR, OUTPUT_DIR, MLRUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ {relative_path(directory)}")


def verify_data_file() -> bool:
    return RAW_DATA_PATH.exists()
