#!/usr/bin/env python3
"""
Setup script for local Superstore forecast execution.
Creates necessary directories and verifies data file exists.
Run this script before opening Superstore_Forecast_Local.ipynb
"""

import os
import sys
from pathlib import Path

def setup_local_environment():
    """Create directories and verify setup for local execution"""

    # Get the script's directory (where the notebook should be)
    script_dir = Path(__file__).parent.resolve()

    print("=" * 60)
    print("Superstore Forecast - Local Setup")
    print("=" * 60)
    print(f"Working directory: {script_dir}\n")

    # Required directories
    required_dirs = {
        'data': 'Input data directory (place Superstore.xlsx here)',
        'output': 'Output directory for results and models',
        'mlruns': 'MLflow experiment tracking directory'
    }

    # Create directories
    all_created = True
    for dir_name, description in required_dirs.items():
        dir_path = script_dir / dir_name
        try:
            dir_path.mkdir(exist_ok=True, parents=True)
            print(f"✓ {dir_name:15} - {description}")
        except Exception as e:
            print(f"✗ {dir_name:15} - Failed to create: {e}")
            all_created = False

    print("\n" + "-" * 60)

    # Verify data file
    data_file = script_dir / 'data' / 'Superstore.xlsx'
    if data_file.exists():
        file_size = data_file.stat().st_size / (1024 * 1024)  # Convert to MB
        print(f"✓ Data file found: Superstore.xlsx ({file_size:.2f} MB)")
        data_ready = True
    else:
        print(f"⚠ Data file missing: {data_file}")
        print(f"  Please place Superstore.xlsx in the ./data/ directory")
        data_ready = False

    print("\n" + "-" * 60)

    # Summary
    if all_created and data_ready:
        print("✓ Setup complete! Ready to run Superstore_Forecast_Local.ipynb")
        return 0
    elif all_created and not data_ready:
        print("✓ Directories created, but data file is missing.")
        print("  Place Superstore.xlsx in ./data/ and run this script again.")
        return 1
    else:
        print("✗ Setup incomplete. Check errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(setup_local_environment())
