"""Plot farm zones and cow trajectories around feeding replacements."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
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
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the two cows' positions during a replacement interval.

    The interval starts at the event's ``time`` and ends after its
    ``bout_interval`` (in seconds). If ``zones`` is provided, draw the farm
    layout in the background; otherwise, show only the cows' positions.
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

    line_handles = []
    for role, cow_id in cow_ids.items():
        cow_positions = local_positions.loc[
            local_positions["IDs"].eq(cow_id)
        ].sort_values("datetime")
        if cow_positions.empty:
            continue
        (line,) = ax.plot(
            cow_positions["x"],
            cow_positions["y"],
            "o-",
            color=cow_colors[role],
            markersize=3,
            linewidth=1.5,
            zorder=3,
            label=f"{role.title()} ({cow_id})",
        )
        line_handles.append(line)

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

    line_labels = [handle.get_label() for handle in line_handles]
    if zone_handles or line_handles:
        ax.legend(
            handles=list(zone_handles.values()) + line_handles,
            labels=list(zone_handles) + line_labels,
            fontsize=8,
            loc="upper right",
        )
    ax.set_aspect("equal", adjustable="datalim" if zones is None else "box")
    ax.set_xlabel("Farm-local X (m)")
    ax.set_ylabel("Farm-local Y (m)")
    ax.set_title("Cow positions during replacement interval")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.margins(0.02)
    fig.suptitle(
        f"Bin {event['bin']} · {event['time']} · "
        f"interval {event['bout_interval']} s"
    )
    return fig, ax
