"""This script examplifies a method of extracting boxplot outliers"""
from pathlib import Path
import pyMoove.io as mooveIO
from pyMoove import statistics as mooveStats
import pandas as pd

# Starting point, load the dataset as per usual

data_dir = Path("data")
dataset = mooveIO.load_processed_dataset(data_dir)


# These three are needed to understand cow to cow distances and cow to feeding bin distances
df_positions = dataset["positions"]
df_replacements = dataset["replacements"]
df_bin_stats = mooveIO.load_bin_stats_parquet(data_dir / "bin_position_stats.parquet")

# Compute the cow to cow and cow to feeding bin distances for all events
repl_event_stats = mooveStats.compute_replacement_event_statistics(
    df_positions, df_replacements, df_bin_stats)


# Box-plots, scatter plots and histograms
# --
# --
# --
# --

# Looking at the results we might find outliers and would like to inspect these

# Select the columns you are interested in
columns = [
    "cow_min_m", "cow_mean_m", "cow_max_m",
    "actor_bin_min_m", "actor_bin_mean_m", "actor_bin_max_m",
    "reactor_bin_min_m", "reactor_bin_mean_m", "reactor_bin_max_m"
]

# If you want to select several outliers combine them uniqulie into one dataframe
# Otherwise find_boxplot_outliers() returns all outlier events for one specific column
outliers = []
for column in columns:
    outliers.append(mooveStats.find_boxplot_outliers(repl_event_stats, column))

# All unique event outliers are collected into one dataframe.
unique_outliers = pd.concat(outliers).drop_duplicates(subset="repl_id")


# We can inspect the outliers however we like
print(f"Total replacement events: {len(repl_event_stats)}")
print(f"Unique outlier events: {len(unique_outliers)}")

print(unique_outliers.head())
