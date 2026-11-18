from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pyMoove.io as mooveIO

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

# Parameters
# Inter cow distance
# Feeding bin distance
# Interaction time criterion

def detect_interactions(data, max_cow_distance=3, max_bin_distance=2, min_duration=10):
    # IDEA: Find pairs of cows that fulfill these criterions under a timespan. The positional data for the two cows during
    # this timespan should be returned

    # Preprocess data:
    # - Remove all positional data that is max_cow_distance + max_bin_distance away from the bins
    
    bin_area = {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}

    # Interactions: cow1, cow2, positions1, postions2, times1, times2 ( cow_distance, bin_distance, bin?)
    pass

if __name__=='__main__':
    detect_interactions(data)