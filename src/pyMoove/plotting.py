"""Plot farm zones and cow trajectories around feeding replacements."""

from collections.abc import Mapping
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch, Ellipse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import ast
from pyMoove.calculations import get_bin_feeding_positions, filter_bin_position_outliers

def plot_zones(
    zones: pd.DataFrame | str | Path,
    ax: plt.Axes | None = None,
    show: bool = True,
    legend: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot farm zones from a DataFrame or a CSV exported by
    :func:`get_zone_info`.

    Parameters
    ----------
    zones:
        The filtered DataFrame returned by :func:`get_zone_info`, or
        the path to a CSV written by that function.
    ax:
        Optional existing matplotlib Axes to draw on. If None, a new figure
        and axes are created.
    show:
        Whether to call ``plt.show()`` before returning. Defaults to True.
    legend:
        Whether to add a zone-type legend. Set to False when the caller will
        create a combined legend containing other plotted elements.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes, allowing further customization or saving.
    """
    if isinstance(zones, (str, Path)):
        zones = pd.read_csv(
            zones,
            converters={"coordinates_m": ast.literal_eval},
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)
    else:
        fig = ax.get_figure()

    zone_colors = {
        "WalkingAlley": "#eeeeee",
        "FeedTable": "#91d18b",
        "Cubicles": "#a6cee3",
        "MilkingArea": "#fdbf6f",
        "HoldingArea": "#cab2d6",
        "Other": "#bdb76b",
        "Sort_TreatmentArea": "#fb9a99",
    }

    seen_zone_types = set()
    for _, row in zones.iterrows():
        zone_type = (
            row["zone_type"] if pd.notna(row["zone_type"]) else "Unspecified"
        )
        x, y = zip(*row["coordinates_m"])
        ax.fill(
            x,
            y,
            color=zone_colors.get(zone_type, "#8c8c8c"),
            alpha=0.35,
            edgecolor="#555555",
            linewidth=0.5,
            zorder=0,
            label=zone_type if zone_type not in seen_zone_types else None,
        )
        seen_zone_types.add(zone_type)

        if zone_type == "FeedTable":
            ax.annotate(
                row["zone_name"].removeprefix("rect"),
                xy=(
                    (row["x_min_m"] + row["x_max_m"]) / 2,
                    row["y_max_m"],
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
                color="#174d22",
                bbox={
                    "boxstyle": "round,pad=0.1",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
                zorder=1,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Farm-local X (m)")
    ax.set_ylabel("Farm-local Y (m)")
    ax.set_title("Farm layout by zone type")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    if legend:
        ax.legend(loc="upper right", frameon=True)
    ax.margins(0.02)
    if show:
        plt.show()
    return fig, ax

def plot_replacement_example(
    data: Mapping[str, pd.DataFrame],
    zones: pd.DataFrame | None = None,
    bin_statistics: pd.DataFrame | None = None,
    event_index: int = 0,
    relative_start_time: int = 60,
    animate: bool = False,
    mark_start: bool = True,
    print_start_positions: bool = False,
    animation_interval_ms: int = 250,
) -> (
    tuple[plt.Figure, plt.Axes]
    | tuple[plt.Figure, plt.Axes, FuncAnimation]
):
    """Plot the two cows' positions during a replacement interval.

    The interval starts at the event's ``time`` and ends after its
    ``bout_interval`` (in seconds). If ``zones`` is provided, draw the farm
    layout in the background; otherwise, show only the cows' positions.

    The start marker marks each cow's first available fix in the interval.
    Set ``animate=True`` to also return a ``FuncAnimation`` as the third
    return value; in Jupyter, the animation is displayed automatically.
    ``print_start_positions=True`` prints those first-fix coordinates in
    farm-local metres.
    """
    positions = data["positions"].copy()
    replacements = data["replacements"]
    event = replacements.iloc[event_index]

    event_time = pd.Timestamp(event["time"])
    if event_time.tzinfo is None:
        event_time = event_time.tz_localize("Europe/Stockholm")
    event_time = event_time.tz_convert("UTC")
    event_start = event_time - pd.Timedelta(seconds=relative_start_time) 
    event_end = event_time + pd.Timedelta(seconds=float(event["bout_interval"]))

    cow_ids = {
        "reactor": str(event["reactor_cow"]),
        "actor": str(event["actor_cow"]),
    }
    cow_colors = {"reactor": "#2878B5", "actor": "#E07A1F"}

    positions["IDs"] = positions["IDs"].astype(str)
    local_positions = positions.loc[
        positions["IDs"].isin(cow_ids.values())
        & positions["datetime"].between(event_start, event_end)
    ]

    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)

    zone_handles: dict[str, Patch] = {}
    if zones is not None:
        plot_zones(zones, ax=ax, show=False, legend=False)

    # Plot the feeding bin position and spatial distribution
    if bin_statistics is not None:
        bin_stats = bin_statistics.loc[
            bin_statistics["bin"] == event["bin"]
        ]

        if not bin_stats.empty:
            bin_stats = bin_stats.iloc[0]

            centroid_x = bin_stats["centroid_x"]
            centroid_y = bin_stats["centroid_y"]
            std_x = bin_stats["std_x"]
            std_y = bin_stats["std_y"]

            if pd.notna(centroid_x) and pd.notna(centroid_y):
                # Plot the bin's spatial standard deviation
                if pd.notna(std_x) and pd.notna(std_y):
                    ellipse = Ellipse(
                        xy=(centroid_x, centroid_y),
                        width=2 * std_x,
                        height=2 * std_y,
                        facecolor="#e31a1c",
                        edgecolor="#b30000",
                        alpha=0.20,
                        linewidth=1.5,
                        zorder=2,
                        label=f"Bin {event['bin']} ±1 SD",
                    )
                    ax.add_patch(ellipse)

                # Plot the bin centroid
                ax.scatter(
                    centroid_x,
                    centroid_y,
                    marker="*",
                    s=250,
                    color="#e31a1c",
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=6,
                    label=f"Bin {event['bin']} centroid",
                )


    start_positions = {}
    cow_tracks = {}
    for role, cow_id in cow_ids.items():
        cow_positions = local_positions.loc[
            local_positions["IDs"].eq(cow_id)
        ].sort_values("datetime")
        if cow_positions.empty:
            continue
        start_positions[role] = cow_positions.iloc[0]
        cow_tracks[role] = cow_positions

        (line,) = ax.plot(
            [],
            [],
            "o-",
            color=cow_colors[role],
            markersize=3,
            linewidth=1.5,
            zorder=3,
            label=f"{role.title()} ({cow_id})",
        )
        if animate:
            (current_point,) = ax.plot(
                [],
                [],
                "o",
                color=cow_colors[role],
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=4,
            )
            cow_tracks[role] = (cow_positions, line, current_point)
        else:
            cow_tracks[role] = (cow_positions, line, None)
            line.set_data(cow_positions["x"], cow_positions["y"])

        if mark_start:
            start = start_positions[role]
            ax.scatter(
                start["x"],
                start["y"],
                marker="X",
                s=100,
                color=cow_colors[role],
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
                label=f"{role.title()} start",
            )

    if print_start_positions:
        print("First observed positions in the replacement interval:")
        for role, cow_id in cow_ids.items():
            if role not in start_positions:
                print(f"  {role.title()} ({cow_id}): no position fix")
                continue
            start = start_positions[role]
            print(
                f"  {role.title()} ({cow_id}): "
                f"x={start['x']:.3f} m, y={start['y']:.3f} m"
            )

    if local_positions.empty:
        ax.text(
            0.5,
            0.5,
            "No position fixes during this replacement interval",
            transform=ax.transAxes,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    handles, labels = ax.get_legend_handles_labels()
    if zone_handles or handles:
        ax.legend(
            handles=list(zone_handles.values()) + handles,
            labels=list(zone_handles) + labels,
            fontsize=8,
            loc="upper right",
        )
    if handles:
        ax.legend(handles=handles, labels=labels, fontsize=8, loc="upper right")
    if zones is None and not local_positions.empty:
        x_span = local_positions["x"].max() - local_positions["x"].min()
        y_span = local_positions["y"].max() - local_positions["y"].min()
        x_padding = max(x_span * 0.05, 0.25)
        y_padding = max(y_span * 0.05, 0.25)
        ax.set_xlim(local_positions["x"].min() - x_padding,
                    local_positions["x"].max() + x_padding)
        ax.set_ylim(local_positions["y"].min() - y_padding,
                    local_positions["y"].max() + y_padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Farm-local X (m)")
    ax.set_ylabel("Farm-local Y (m)")
    ax.set_title("Cow positions during replacement interval")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.margins(0.02)
    fig.suptitle(
        f"Bin {event['bin']} · {event['time']} · "
        f"interval {event['bout_interval']} s"
    )

    if not animate:
        return fig, ax
    if local_positions.empty:
        raise ValueError("Cannot animate this event: no cow positions in its interval.")

    frame_times = pd.DatetimeIndex(local_positions["datetime"].drop_duplicates())
    frame_times = frame_times.sort_values()

    def update(frame_index: int):
        frame_time = frame_times[frame_index]
        updated_artists = []
        for role, (cow_positions, line, current_point) in cow_tracks.items():
            visible = cow_positions.loc[cow_positions["datetime"] <= frame_time]
            line.set_data(visible["x"], visible["y"])
            latest = visible.iloc[-1]
            current_point.set_data([latest["x"]], [latest["y"]])
            updated_artists.extend([line, current_point])
        ax.set_title(
            "Cow positions during replacement interval\n"
            f"{frame_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        return updated_artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_times),
        interval=animation_interval_ms,
        repeat=False,
        blit=False,
    )

    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
            display(HTML(animation.to_jshtml()))
            # Prevent Jupyter's end-of-cell hook from also displaying the
            # underlying Matplotlib figure as a separate static image.
            plt.close(fig)
    except ImportError:
        pass

    return fig, ax, animation

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
    fig, ax = plot_zones(zones, show=False)

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

def plot_bin_feeding_positions(
    data: dict[str, pd.DataFrame],
    zones: pd.DataFrame | str | Path,
    bin_id: int,
    color_by_cow: bool = True,
    tz: str = "Europe/Stockholm",
    show: bool = True,
    outlier_confidence: float = 0.999,
    show_outliers: bool = True,
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

    fig, ax = plot_zones(zones, show=False)

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
        inliers, outliers, _, _ = filter_bin_position_outliers(
            bin_pos, confidence=outlier_confidence
        )
        print(f"Total position fixes: {len(bin_pos)}")
        print(
            f"Retained fixes:       {len(inliers)} "
            f"({len(outliers)} flagged as outliers; confidence={outlier_confidence:.1%})"
        )

        if not inliers.empty:
            mean_x = inliers["x"].mean()
            mean_y = inliers["y"].mean()
            std_x = inliers["x"].std()
            std_y = inliers["y"].std()
            print(
                f"Filtered centroid:    x = {mean_x:.2f} m (std {std_x:.2f}), "
                f"y = {mean_y:.2f} m (std {std_y:.2f})"
            )

            if color_by_cow:
                cows = inliers["cow_id"].unique()
                cmap = plt.get_cmap("tab20" if len(cows) > 10 else "tab10")
                for i, cow in enumerate(cows):
                    subset = inliers.loc[inliers["cow_id"] == cow]
                    ax.scatter(
                        subset["x"], subset["y"],
                        color=cmap(i % cmap.N), s=18, alpha=0.6,
                        zorder=4, label=f"Cow {cow} ({len(subset)} inliers)",
                    )
            else:
                ax.scatter(
                    inliers["x"], inliers["y"],
                    color="#d9381e", s=18, alpha=0.5, zorder=4,
                    label=f"Inlier fixes (N={len(inliers)})",
                )

            if show_outliers and not outliers.empty:
                ax.scatter(
                    outliers["x"], outliers["y"],
                    color="#888888", marker="x", s=42, linewidth=1.5,
                    alpha=0.9, zorder=5,
                    label=f"Rejected outliers (N={len(outliers)})",
                )

            # The reported centroid is calculated from retained fixes only.
            ax.scatter(
                mean_x, mean_y,
                marker="*", s=160, color="#ffd700",
                edgecolor="#222222", linewidth=1.2, zorder=6,
                label=f"Filtered bin centroid ({mean_x:.1f}, {mean_y:.1f} m)",
            )
        else:
            ax.text(
                0.5, 0.5, "No valid inlier positions available",
                transform=ax.transAxes, ha="center", va="center",
            )

    ax.set_title(
        f"All cow positions during feeding at Bin {bin_id} "
        f"({total_events} visits, {unique_cows} cows, {len(bin_pos):,} fixes; "
        f"{len(inliers) if not bin_pos.empty else 0:,} retained)"
    )
    # Position legend cleanly (ncol=2 if many cows)
    legend_cols = 2 if color_by_cow and not bin_pos.empty and bin_pos["cow_id"].nunique() > 6 else 1
    ax.legend(loc="upper right", frameon=True, fontsize=8, ncol=legend_cols)

    if show:
        plt.show()

    return fig, ax