"""Plot farm zones and cow trajectories around feeding replacements."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch


_ZONE_COLORS = {
    "WalkingAlley": "#eeeeee",
    "FeedTable": "#91d18b",
    "Cubicles": "#a6cee3",
    "MilkingArea": "#fdbf6f",
    "HoldingArea": "#cab2d6",
    "Other": "#bdb76b",
    "Sort_TreatmentArea": "#fb9a99",
}


def plot_replacement_example(
    data: Mapping[str, pd.DataFrame],
    zones: pd.DataFrame | None = None,
    event_index: int = 0,
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
    event_end = event_time + pd.Timedelta(seconds=float(event["bout_interval"]))

    cow_ids = {
        "reactor": str(event["reactor_cow"]),
        "actor": str(event["actor_cow"]),
    }
    cow_colors = {"reactor": "#2878B5", "actor": "#E07A1F"}

    positions["IDs"] = positions["IDs"].astype(str)
    local_positions = positions.loc[
        positions["IDs"].isin(cow_ids.values())
        & positions["datetime"].between(event_time, event_end)
    ]

    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)

    zone_handles: dict[str, Patch] = {}
    if zones is not None:
        for _, zone in zones.iterrows():
            zone_type = (
                zone["zone_type"] if pd.notna(zone["zone_type"]) else "Unspecified"
            )
            zone_color = _ZONE_COLORS.get(zone_type, "#bdbdbd")
            x_zone, y_zone = zip(*zone["coordinates_m"])
            ax.fill(
                x_zone,
                y_zone,
                color=zone_color,
                alpha=0.35,
                edgecolor="#555555",
                linewidth=0.5,
                zorder=0,
            )

            if zone_type not in zone_handles:
                zone_handles[zone_type] = Patch(
                    facecolor=zone_color,
                    edgecolor="#555555",
                    label=zone_type,
                )

            if zone_type == "FeedTable":
                ax.annotate(
                    zone["zone_name"].removeprefix("rect"),
                    xy=(
                        (zone["x_min_m"] + zone["x_max_m"]) / 2,
                        zone["y_max_m"],
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
