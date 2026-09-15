#!/usr/bin/env python3
"""
Transform Fabric-specific Jupyter notebook to local-compatible version.
Reads AIsample - Superstore Forecast v1.2 (2).ipynb and creates Superstore_Forecast_Local.ipynb
"""

import json
import os
import re
from pathlib import Path
from copy import deepcopy

def read_notebook(notebook_path):
    """Read and parse Jupyter notebook JSON"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def transform_cell_source(source):
    """Transform cell source code to remove Fabric-specific references"""
    if isinstance(source, list):
        source = ''.join(source)

    # Replace lakehouse paths (multiple patterns)
    source = source.replace('/lakehouse/default/Files/salesforecast/', './data/')
    source = source.replace('/lakehouse/default/Files/', './data/')
    source = source.replace('/lakehouse/default', '.')
    source = source.replace('/lakehouse', '.')
    source = re.sub(r'["\']/lakehouse[^"\']*["\']\)', r'"./data/")', source)
    source = re.sub(r'["\']/lakehouse[^\'"]+["\']\s*\)', r'"./data/file")', source)
    source = re.sub(r'DATA_ROOT\s*=\s*["\'][^"\']*lakehouse[^"\']*["\']', 'DATA_ROOT = "."', source)

    # Fix nested data paths like ./data/raw/ to ./data/
    source = re.sub(r'./data/raw/', './data/', source)

    # Replace display() calls with print() - more comprehensive patterns
    # Handle display(...) with any content including function calls
    source = re.sub(r'display\s*\(\s*(\w+(?:\.\w+)*(?:\([^)]*\))?)\s*\)', r'print(\1)', source)
    # Handle display() with no arguments
    source = re.sub(r'display\s*\(\s*\)', r'print("Output:")', source)

    # Replace workspace/lakehouse checks with local directory checks
    source = source.replace(
        "workspace_id = spark.conf.get('spark.databricks.clusterUserId')",
        "workspace_id = 'local_user'"
    )

    # Replace Spark writes to parquet/delta with pandas CSV
    # Pattern: df.write.mode("overwrite").parquet(...)
    source = re.sub(
        r'\.write\.mode\(["\']overwrite["\']\)\.(?:parquet|delta)\([\'"]([^\'"]+)["\']\)',
        lambda m: f'.to_csv(\'./output/{Path(m.group(1)).name}.csv\', index=False)',
        source
    )

    # Replace Spark writes mode patterns
    source = re.sub(
        r'\.write\.mode\(["\']overwrite["\']\)',
        '.to_csv(\'./output/temp.csv\', index=False)',
        source
    )

    # Comment out Fabric-specific imports and checks
    fabric_patterns = [
        (r'from notebookutils import mssparkutils', '# from notebookutils import mssparkutils'),
        (r'import mssparkutils', '# import mssparkutils'),
        (r'mssparkutils\.notebook\.run\(', '# mssparkutils.notebook.run('),
        (r'spark\.read\.load\(', 'pd.read_excel('),
    ]

    for pattern, replacement in fabric_patterns:
        source = re.sub(pattern, replacement, source)

    # Add local backend for MLflow
    if 'mlflow.set_experiment' in source:
        # Ensure MLflow uses local backend
        if 'mlflow.set_tracking_uri' not in source:
            source = source.replace(
                'mlflow.set_experiment',
                'mlflow.set_tracking_uri("file:///./mlruns")\nmlflow.set_experiment'
            )

    return source

def transform_notebook(notebook_path, output_path):
    """Transform notebook and save to new file"""
    notebook = read_notebook(notebook_path)
    new_notebook = deepcopy(notebook)

    # Create setup cell as first cell
    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup cell for local execution\n",
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "# Create necessary directories\n",
            "dirs = ['./data', './output', './mlruns']\n",
            "for dir_name in dirs:\n",
            "    Path(dir_name).mkdir(exist_ok=True, parents=True)\n",
            "    print(f'Created/verified directory: {dir_name}')\n",
            "\n",
            "# Verify Superstore.xlsx exists\n",
            "if not os.path.exists('./data/Superstore.xlsx'):\n",
            "    print('WARNING: ./data/Superstore.xlsx not found. Please ensure the data file is in place.')\n",
            "else:\n",
            "    print('Superstore.xlsx found at ./data/Superstore.xlsx')"
        ]
    }

    # Insert setup cell at beginning
    new_cells = [setup_cell]

    # Transform all existing cells
    for idx, cell in enumerate(notebook['cells']):
        new_cell = deepcopy(cell)

        if cell['cell_type'] == 'code':
            # Transform source code
            original_source = cell['source']
            transformed_source = transform_cell_source(original_source)

            # Add comment if cell was modified
            if transformed_source != (''.join(original_source) if isinstance(original_source, list) else original_source):
                if isinstance(transformed_source, str):
                    transformed_source = [
                        "# MODIFIED FOR LOCAL EXECUTION\n",
                        *transformed_source.split('\n')[:-1],
                        transformed_source.split('\n')[-1]
                    ] if transformed_source else transformed_source

            new_cell['source'] = transformed_source if isinstance(transformed_source, list) else transformed_source.split('\n')

            # Clear outputs for code cells
            new_cell['outputs'] = []
            new_cell['execution_count'] = None

        new_cells.append(new_cell)

    new_notebook['cells'] = new_cells

    # Save new notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_notebook, f, indent=1, ensure_ascii=False)

    print(f"✓ Notebook transformed and saved to: {output_path}")
    print(f"✓ Total cells: {len(new_cells)} (1 setup + {len(new_cells)-1} original)")
    return output_path

if __name__ == '__main__':
    base_dir = Path(r'c:\Users\hadarmawan\Documents\sarailmax')
    notebook_path = base_dir / "AIsample - Superstore Forecast v1.2 (2).ipynb"
    output_path = base_dir / "Superstore_Forecast_Local.ipynb"

    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        sys.exit(1)

    transform_notebook(str(notebook_path), str(output_path))
