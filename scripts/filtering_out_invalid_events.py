"""This scripts shows the workflow for filtering out invalid events"""
from pathlib import Path
from pyMoove import io as mooveIO
from pyMoove import statistics as mooveStats
from pyMoove import cleaning as mooveClean
import pandas as pd

data_dir = Path("data")

# Load the dataset
dataset = mooveIO.load_processed_dataset(data_dir)

df_positions = dataset["positions"]
df_replacements: pd.DataFrame = dataset["replacements"]
df_bin_stats = mooveIO.load_bin_stats_parquet(data_dir / "bin_position_stats.parquet")

# Get cow to cow and cow to feeding bin distance information for each event
repl_event_stats = mooveStats.inspect_replacement_event_statistics(
    df_positions, df_replacements, df_bin_stats)

# Summarize the information for the whole set
mooveStats.summarize_event_coverage(repl_event_stats)
# This shows that there might be invalid events, 
# i.e. events in which there is missing positional data for one cow, both cows
# or if the positional data is not synchronized


# We can filter out these events 
df_replacements_clean = mooveClean.filter_replacement_events(
    df_replacements, repl_event_stats)

# Recalculate statistics for cleaned events
repl_event_stats_clean = mooveStats.inspect_replacement_event_statistics(
    df_positions, df_replacements_clean, df_bin_stats
)
# Inspect again to verify
mooveStats.summarize_event_coverage(repl_event_stats_clean)

# And we can work with the valid events only however we like
print(repl_event_stats_clean.head())
