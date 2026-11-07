"""Workshop folder paths, found automatically from where this file lives."""

import subprocess
from pathlib import Path

# Folder that holds the notebooks (works on 2i2c, Docker, or locally); this file is in its utils/ folder
WORKSHOP_DIR = Path(__file__).resolve().parent.parent

# All NGIAB preprocessing inputs/outputs are stored here, next to the notebooks
DATA_DIR = WORKSHOP_DIR / "ngiab_preprocess_output"

# Saved figures go here
FIGURES_DIR = WORKSHOP_DIR / "figures"

_NGEN_PYTHON = "/ngen/.venv/bin/python"


def set_preprocess_output_dir():
    """Tell ngiab_data_cli to write its outputs into DATA_DIR (avoids the interactive prompt)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (Path.home() / ".ngiab").mkdir(exist_ok=True)  # a new machine does not have it yet; NGIAB saves its settings there
    subprocess.run(
        [
            _NGEN_PYTHON,
            "-c",
            "from data_sources.source_validation import FilePaths; "
            f"FilePaths.set_working_dir({str(DATA_DIR)!r})",
        ],
        check=True,
    )
    print(f"NGIAB outputs folder: {DATA_DIR}")
