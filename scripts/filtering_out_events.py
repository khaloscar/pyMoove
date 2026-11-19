"""This scripts shows how to use pandas DataFrame to filter out specific events from given conditions"""
from pathlib import Path
from pyMoove import io as mooveIO
from pyMoove import statistics as mooveStats
import pandas as pd

data_dir = Path("data")

# Load the dataset
dataset = mooveIO.load_processed_dataset(data_dir)

df_positions = dataset["positions"]
df_replacements: pd.DataFrame = dataset["replacements"]
df_bin_stats = mooveIO.load_bin_stats_parquet(data_dir / "bin_position_stats.parquet")
stats = mooveStats.inspect_replacement_event_statistics(
    df_positions, df_replacements, df_bin_stats)

# Create a boolean mask
# Use .isin(), .between() and boolean operaators |, &, ==

# Select individual bins
selectMask = stats['bin'].isin([10, 15])
# Can also do:
# selectMask = (stats['bin'] == 10) | (stats['bin'] == 15)
selected_bins = stats[selectMask]

print("Selection of bin 10 and 15")
print(selected_bins.head())
print()

# Or maybe remove one or more bins, i.e. filter them out
filtered_bins = stats[~selectMask]

print("\nSelect all bins except 10 and 15")
print(filtered_bins.head())
print()

# Select a range of bins
rangeMask = stats['bin'].between(0, 5, inclusive="both")
range_bins = stats[rangeMask]
print("\nSelect a range of bins 0 to 5")
print(range_bins.head())
print()

# Selection of individual event
eventMask = stats["event_id"] == 1
eventOne = stats[eventMask]
print("\nSelection of event_id 1")
print(eventOne.head())
print()

# Combining critera, 
# Criteria 1 min cow to cow < 2.5 and cow to bin < 2.5 for bins 0 to 20
binMask = stats['bin'].between(0, 10, inclusive="both")
cowTocowMask = stats["cow_min_m"] < 2.5
cowTobinMask = (stats['actor_bin_min_m'] < 2.5) & (stats['reactor_bin_min_m'] < 2.5)
mask = binMask & cowTocowMask & cowTobinMask
selection = stats[mask]
print("\nCombined critera selection")
print(selection.head())
