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

# positions columns: NIDs, IDs, Time, x, y, Inter, velocity, datetime
# replacements columns: reactor_cow, bin, time, actor_cow, bout_interval
# feeding columns: cow, bin, start_time, intake, duration, transponder, end_time, start_weight, end_weight

# IDEA: Plot the position of a cow while it is feeding at a specific bin/gate.
# Step 1: Select a feeding event (a row of data['feeding']).
# Step 2: Find the position data for that cow (feeding["cow"] = positions["IDs"]) in the time interval between start_time and end_time
# Step 3: Plot the positional data on top of the plot created by zone_info.py


def plot_feeding_event(
    data: dict[str, pd.DataFrame],
    zones: pd.DataFrame | str | Path,
    event_index: int = 0,
    tz: str = "Europe/Stockholm",
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a single cow's trajectory during one feeding event on top of farm zones.

    Parameters
    ----------
    data:
        Processed dataset containing 'feeding' and 'positions' DataFrames.
    zones:
        Farm layout zones DataFrame or path to zones CSV.
    event_index:
        Row index of the feeding event to plot (default: 0).
    tz:
        Timezone of the feeder timestamps (default: 'Europe/Stockholm').
    show:
        Whether to display the plot with plt.show().

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes.
    """
    feeding_df = data["feeding"]
    positions_df = data["positions"]

    # Step 1: Select a feeding event (a row of data['feeding'])
    event = feeding_df.iloc[event_index]
    cow_id = str(event["cow"])
    bin_nr = event["bin"]

    print(f"\n--- Selected Feeding Event (row {event_index}) ---")
    print(f"Cow ID:     {cow_id}")
    print(f"Bin/Gate:   {bin_nr}")
    print(f"Start Time: {event['start_time']}")
    print(f"End Time:   {event['end_time']}")
    print(f"Duration:   {event['duration']} s")
    print(f"Intake:     {event['intake']} kg")

    # Step 2: Find position data for that cow in the time interval between start_time and end_time.
    # Note on timestamps:
    # - positions['datetime'] is stored in UTC.
    # - feeding start_time and end_time are recorded in local Swedish time (Europe/Stockholm)
    #   as 'HH:MM:SS' strings.
    # We construct the timestamp using the date from the positions dataset,
    # localize to Europe/Stockholm, and convert to UTC.
    date_str = str(positions_df["datetime"].min().date())

    start_str = str(event["start_time"])
    start_dt = pd.Timestamp(f"{date_str} {start_str}")
    if start_dt.tzinfo is None:
        start_dt = start_dt.tz_localize(tz)
    start_dt_utc = start_dt.tz_convert("UTC")

    # Compute end time in UTC using duration to safely handle visits crossing midnight
    if "duration" in event and pd.notna(event["duration"]):
        end_dt_utc = start_dt_utc + pd.Timedelta(seconds=float(event["duration"]))
    else:
        end_str = str(event["end_time"])
        end_dt = pd.Timestamp(f"{date_str} {end_str}")
        if end_dt.tzinfo is None:
            end_dt = end_dt.tz_localize(tz)
        end_dt_utc = end_dt.tz_convert("UTC")

    # Filter positions for the cow within the feeding interval
    cow_positions = positions_df.loc[
        positions_df["IDs"].astype(str).eq(cow_id)
        & positions_df["datetime"].between(start_dt_utc, end_dt_utc)
    ].sort_values("datetime")

    print(
        f"Position fixes found: {len(cow_positions)} "
        f"(UTC window: {start_dt_utc.strftime('%H:%M:%S')} - {end_dt_utc.strftime('%H:%M:%S')})"
    )

    # Step 3: Plot positional data on top of the plot created by zone_info.py
    fig, ax = zone_info.plot_zones(zones, show=False)

    if not cow_positions.empty:
        # Plot trajectory line and fixes
        ax.plot(
            cow_positions["x"],
            cow_positions["y"],
            "-",
            color="#d9381e",
            linewidth=1.8,
            alpha=0.7,
            zorder=4,
            label=f"Trajectory (Cow {cow_id})",
        )
        ax.scatter(
            cow_positions["x"],
            cow_positions["y"],
            color="#d9381e",
            s=22,
            alpha=0.85,
            zorder=5,
            label="Position fixes",
        )

        # Highlight start and end positions
        ax.scatter(
            cow_positions["x"].iloc[0],
            cow_positions["y"].iloc[0],
            color="#2ca02c",
            s=75,
            marker="^",
            zorder=6,
            label="Start position",
        )
        ax.scatter(
            cow_positions["x"].iloc[-1],
            cow_positions["y"].iloc[-1],
            color="#1f77b4",
            s=75,
            marker="v",
            zorder=6,
            label="End position",
        )
    else:
        ax.text(
            0.5,
            0.5,
            f"No position fixes found for Cow {cow_id} during feeding interval",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    ax.set_title(
        f"Cow {cow_id} feeding at Bin {bin_nr} "
        f"({event['start_time']} - {event['end_time']}, {event['duration']} s, intake: {event['intake']} kg)"
    )
    ax.legend(loc="upper right", frameon=True)

    if show:
        plt.show()

    return fig, ax


def get_bin_feeding_positions(
    data: dict[str, pd.DataFrame],
    bin_id: int,
    tz: str = "Europe/Stockholm",
) -> pd.DataFrame:
    """Extract all positional records across all feeding events at a specific bin.

    Parameters
    ----------
    data:
        Processed dataset containing 'feeding' and 'positions' DataFrames.
    bin_id:
        Bin/gate number (e.g. 21, 24).
    tz:
        Timezone of the feeder timestamps (default: 'Europe/Stockholm').

    Returns
    -------
    pd.DataFrame
        Position fixes across all visits to this bin, with extra columns
        'cow_id', 'bin', and 'visit_idx'.
    """
    feeding_df = data["feeding"]
    positions_df = data["positions"]

    bin_events = feeding_df.loc[feeding_df["bin"] == bin_id].copy()
    if bin_events.empty:
        print(f"Warning: No feeding events found for bin {bin_id}")
        return pd.DataFrame()

    cows_at_bin = set(bin_events["cow"].astype(str).unique())
    # Pre-filter positions by the set of cows who visited this bin for efficiency
    pos_subset = positions_df.loc[positions_df["IDs"].astype(str).isin(cows_at_bin)].copy()
    date_str = str(positions_df["datetime"].min().date())

    collected = []
    for visit_idx, event in bin_events.iterrows():
        cow_id = str(event["cow"])
        start_str = str(event["start_time"])
        start_dt = pd.Timestamp(f"{date_str} {start_str}")
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize(tz)
        start_dt_utc = start_dt.tz_convert("UTC")

        if "duration" in event and pd.notna(event["duration"]):
            end_dt_utc = start_dt_utc + pd.Timedelta(seconds=float(event["duration"]))
        else:
            end_str = str(event["end_time"])
            end_dt = pd.Timestamp(f"{date_str} {end_str}")
            if end_dt.tzinfo is None:
                end_dt = end_dt.tz_localize(tz)
            end_dt_utc = end_dt.tz_convert("UTC")

        fixes = pos_subset.loc[
            pos_subset["IDs"].astype(str).eq(cow_id)
            & pos_subset["datetime"].between(start_dt_utc, end_dt_utc)
        ].copy()

        if not fixes.empty:
            fixes["cow_id"] = cow_id
            fixes["bin"] = bin_id
            fixes["visit_idx"] = visit_idx
            collected.append(fixes)

    if not collected:
        print(f"Warning: No position fixes found during feeding events for bin {bin_id}")
        return pd.DataFrame()

    return pd.concat(collected, ignore_index=True)


def plot_bin_feeding_positions(
    data: dict[str, pd.DataFrame],
    zones: pd.DataFrame | str | Path,
    bin_id: int,
    color_by_cow: bool = True,
    tz: str = "Europe/Stockholm",
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot all positional fixes during all feeding events for a specific bin.

    Parameters
    ----------
    data:
        Processed dataset containing 'feeding' and 'positions' DataFrames.
    zones:
        Farm layout zones DataFrame or path to zones CSV.
    bin_id:
        The bin/gate number to inspect.
    color_by_cow:
        If True, each cow is assigned a distinct color. If False, all points
        are plotted with a single semi-transparent color to show spatial density.
    tz:
        Timezone of the feeder timestamps (default: 'Europe/Stockholm').
    show:
        Whether to display the plot with plt.show().

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes.
    """
    bin_events = data["feeding"].loc[data["feeding"]["bin"] == bin_id]
    total_events = len(bin_events)
    unique_cows = bin_events["cow"].nunique()
    total_duration = bin_events["duration"].sum() if "duration" in bin_events else 0

    print(f"\n--- Feeding Events at Bin {bin_id} ---")
    print(f"Total feeding visits: {total_events}")
    print(f"Unique cows:          {unique_cows}")
    print(f"Total feeding time:   {total_duration:,} s ({total_duration / 60:.1f} min)")

    bin_pos = get_bin_feeding_positions(data, bin_id=bin_id, tz=tz)

    fig, ax = zone_info.plot_zones(zones, show=False)

    if bin_pos.empty:
        ax.text(
            0.5,
            0.5,
            f"No positioning data recorded during feeding visits at Bin {bin_id}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    else:
        print(f"Total position fixes: {len(bin_pos)}")
        mean_x = bin_pos["x"].mean()
        mean_y = bin_pos["y"].mean()
        std_x = bin_pos["x"].std()
        std_y = bin_pos["y"].std()
        print(f"Fixes centroid:       x = {mean_x:.2f} m (std {std_x:.2f}), y = {mean_y:.2f} m (std {std_y:.2f})")

        if color_by_cow:
            cows = bin_pos["cow_id"].unique()
            # Use colormap to cycle through distinct colors for each cow
            cmap = plt.get_cmap("tab20" if len(cows) > 10 else "tab10")
            for i, cow in enumerate(cows):
                subset = bin_pos.loc[bin_pos["cow_id"] == cow]
                color = cmap(i % cmap.N)
                ax.scatter(
                    subset["x"],
                    subset["y"],
                    color=color,
                    s=18,
                    alpha=0.6,
                    zorder=4,
                    label=f"Cow {cow} ({len(subset)} fixes)",
                )
        else:
            ax.scatter(
                bin_pos["x"],
                bin_pos["y"],
                color="#d9381e",
                s=18,
                alpha=0.5,
                zorder=4,
                label=f"Position fixes (N={len(bin_pos)})",
            )

        # Plot estimated centroid marker of the bin
        ax.scatter(
            mean_x,
            mean_y,
            marker="*",
            s=160,
            color="#ffd700",
            edgecolor="#222222",
            linewidth=1.2,
            zorder=6,
            label=f"Bin {bin_id} centroid ({mean_x:.1f}, {mean_y:.1f} m)",
        )

    ax.set_title(
        f"All cow positions during feeding at Bin {bin_id} "
        f"({total_events} visits, {unique_cows} cows, {len(bin_pos):,} fixes)"
    )
    # Position legend cleanly (ncol=2 if many cows)
    legend_cols = 2 if color_by_cow and not bin_pos.empty and bin_pos["cow_id"].nunique() > 6 else 1
    ax.legend(loc="upper right", frameon=True, fontsize=8, ncol=legend_cols)

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    # Load filtered zones for farm layout
    zones_path = DATA_DIR / "zones_filtered.csv"
    if not zones_path.exists():
        zones_path = DATA_DIR / "zone_info.csv"
        zones = zone_info.get_zone_info(zones_path)
    else:
        zones = zones_path

    # Option A: Plot all feeding events for a specific bin (e.g. Bin 21, 24, etc.)
    target_bin = 2
    fig, ax = plot_bin_feeding_positions(data, zones, bin_id=target_bin, color_by_cow=True)

    # Option B: Plot a single feeding event by index (uncomment to use):
    # fig, ax = plot_feeding_event(data, zones, event_index=0)
