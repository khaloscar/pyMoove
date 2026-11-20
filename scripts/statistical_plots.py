from pathlib import Path
from pyMoove import io as mooveIO
from pyMoove import statistics as mooveStats
from pyMoove import cleaning as mooveClean
from pyMoove import calculations as mooveCalc
import matplotlib.pyplot as plt


plt.rcParams.update({
    'figure.titlesize': 18,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 9,
    'legend.title_fontsize': 10
})


data_dir = Path("data")
dataset = mooveIO.load_processed_dataset(data_dir)
repl_stats_dir = Path("Replacement_statistics")
repl_stats_path = data_dir / repl_stats_dir
repl_stats_path.mkdir(parents=True, exist_ok=True)

#NOTE: Workflow for calculating the bin positions and saving the results to a parquet.
bin_stats_filePath = data_dir / "bin_position_stats.parquet"
if False:
    df_bin_stats = mooveCalc.calculate_bin_position_statistics(dataset, list(range(1,59)))
    mooveIO.save_dataframe_to_parquet(df_bin_stats, bin_stats_filePath)

df_bin_stats = mooveIO.load_bin_stats_parquet(bin_stats_filePath)
df_positions = dataset["positions"]
df_replacements = dataset["replacements"]

print(df_positions.head())

# Get the replacement statistics df
time_before = 30.0
time_after = 30.0
df_replacements_stats = mooveStats.compute_replacement_event_statistics(
                        df_positions,
                        df_replacements,
                        df_bin_stats,
                        time_before=time_before,
                        time_after=time_after,
)

# Remove invalid events - i.e. events where there is missing or not synchronized positional data for the cows, also check that it went okay..
mooveStats.summarize_event_coverage(df_replacements_stats)
df_replacements = mooveClean.filter_replacement_events(df_replacements, df_replacements_stats)
# The replacement event statistics does not contain invalid events, do boxplots on this data..
df_replacements_stats = mooveStats.compute_replacement_event_statistics(
                        df_positions,
                        df_replacements,
                        df_bin_stats,
                        time_before=time_before,
                        time_after=time_after,
)
mooveStats.summarize_event_coverage(df_replacements_stats)


# Various plots now!


##############
# BOX PLOTS
##############

if True:
    # Cow to cow distances
    fig, ax = plt.subplots()
    df_replacements_stats[["cow_min_m", "cow_mean_m", "cow_max_m"]].plot.box(
        ax=ax,
        ylabel="Distance [m]",
        title=f"Cow to cow distances during replacement events, Before: {time_before}, After: {time_after}",
    )
    filepath = repl_stats_path / "cow_to_cow_boxplot.png"
    fig.savefig(filepath, dpi=300)

    # Actor to bin distances
    fig, ax = plt.subplots()
    df_replacements_stats[["actor_bin_min_m", "actor_bin_mean_m", "actor_bin_max_m"]].plot.box(
        ax=ax,
        ylabel="Distance [m]",
        title=f"Actor to feed bin distances during replacement events, Before: {time_before}, After: {time_after}",
    )
    filepath = repl_stats_path / "actor_to_bin_boxplot.png"
    fig.savefig(filepath, dpi=300)

    # Reactor to bin distances
    fig, ax = plt.subplots()
    df_replacements_stats[["reactor_bin_min_m", "reactor_bin_mean_m", "reactor_bin_max_m"]].plot.box(
        ax=ax,
        ylabel="Distance [m]",
        title=f"Reactor to feed bin distances during replacement events, Before: {time_before}, After: {time_after}",
    )
    filepath = repl_stats_path / "reactor_to_bin_boxplot.png"
    fig.savefig(filepath, dpi=300)


##############
# HISTOGRAMS
##############

if True:
    # Minimum cow to cow distance
    fig, ax = plt.subplots()
    df_replacements_stats["cow_min_m"].plot.hist(
        ax=ax,
        bins=40,
        xlabel="Minimum cow to cow distance [m]",
        ylabel="Number of replacement events",
        title=f"Minimum separation during replacements, Before: {time_before}, After: {time_after}",
    )
    filepath = repl_stats_path / "cow_to_cow_min_histogram.png"
    fig.savefig(filepath, dpi=300)

    # All cow to cow distances
    fig, ax = plt.subplots()
    df_replacements_stats[["cow_min_m", "cow_mean_m", "cow_max_m"]].plot.hist(
        ax=ax,
        bins=40,
        alpha=0.7,
        xlabel="Cow to cow distances [m]",
        ylabel="Number of replacement events",
        title=f"Cow to cow distance histogram, Before: {time_before}, After: {time_after}",
    )
    filepath = repl_stats_path / "cow_to_cow_all_histogram.png"
    fig.savefig(filepath, dpi=300)

    # Replacement events per feeding bin
    fig, ax = plt.subplots(figsize=(12, 5))
    df_replacements_stats["bin"].value_counts().sort_index().plot.bar(
        ax=ax,
        xlabel="Feeding bin",
        ylabel="Number of replacements",
        title=f"Replacement events per feeding bin, Before: {time_before}, After: {time_after}",
        grid=True,
    )
    filepath = repl_stats_path / "events_per_feeding_bin.png"
    fig.savefig(filepath, dpi=300, bbox_inches="tight")


##############
# SCATTER PLOTS
##############

if True:
    # Minimum distance vs bout interval
    fig, ax = plt.subplots()
    df_replacements_stats.plot.scatter(
        ax=ax,
        x="bout_interval",
        y="cow_min_m",
        alpha=0.4,
        xlabel="Bout interval [s]",
        ylabel="Minimum cow distance [m]",
        title=f"Time between visits vs minimum cow to cow distance, Before: {time_before}, After: {time_after}",
    )
    filepath = repl_stats_path / "cow_to_cow_dist_v_bout_interval_scatter.png"
    fig.savefig(filepath, dpi=300)

    # Actor vs reactor distance to feeding bin
    fig, ax = plt.subplots()
    df_replacements_stats.plot.scatter(
        ax=ax,
        x="actor_bin_mean_m",
        y="reactor_bin_mean_m",
        alpha=0.4,
        xlabel="Actor mean distance to bin [m]",
        ylabel="Reactor mean distance to bin [m]",
        title=f"Actor vs reactor distance to feeding bin, Before: {time_before}, After: {time_after}",
    )
    filepath = repl_stats_path / "actor_v_reactor_distance_to_bin_scatter.png"
    fig.savefig(filepath, dpi=300)


plt.show()
