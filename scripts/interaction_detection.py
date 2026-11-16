"""Plot cow positioning data during feeding events on top of farm zones."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pyMoove.io as mooveIO
import zone_info

# Ensure scripts directory is in sys.path so zone_info can be imported
# regardless of current working directory
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Resolve data directory (works from project root or scripts directory)
DATA_DIR = (
    Path("data")
    if Path("data").exists()
    else SCRIPTS_DIR.parent / "data"
)

data = mooveIO.load_processed_dataset(DATA_DIR)