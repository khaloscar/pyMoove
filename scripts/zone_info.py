"""Load, filter, inspect, and plot farm zones from ``zone_info.csv``."""

import ast
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


_ZONE_COLUMNS = [
    "zone_name",
    "geometry",
    "properties",
    "orgState",
    "fill_color",
    "scale_x",
    "start_point_y",
    "scale_y",
    "start_point_x",
    "zone_rotation",
    "zone_type",
    "zone_layer",
    "geometry_type",
    "updated",
]


def _load_zone_coordinates(zone_info_path: str | Path) -> pd.DataFrame:
    """Load zones and convert their polygon vertices and bounds to metres."""
    zone_info = pd.read_csv(
        zone_info_path,
        sep=";",
        engine="python",
        quoting=csv.QUOTE_NONE,
    )
    zone_info.columns = _ZONE_COLUMNS

    for column in ["zone_name", "zone_type"]:
        zone_info[column] = zone_info[column].astype("string").str.strip('"')

    zone_info = zone_info.loc[
        zone_info["zone_name"].notna()
        & zone_info["zone_name"].ne("")
        & zone_info["geometry"].notna()
    ].copy()

    def parse_coordinates_m(geometry: str) -> list[tuple[float, float]]:
        polygon = ast.literal_eval(geometry.replace('","', ","))
        return [(float(x) / 1000, float(y) / 1000) for x, y in polygon[0]]

    def get_bounds_m(
        points: list[tuple[float, float]],
    ) -> dict[str, float]:
        x_values, y_values = zip(*points)
        return {
            "x_min_m": min(x_values),
            "x_max_m": max(x_values),
            "y_min_m": min(y_values),
            "y_max_m": max(y_values),
            "width_m": max(x_values) - min(x_values),
            "height_m": max(y_values) - min(y_values),
        }

    zones = zone_info[["zone_name", "zone_type", "geometry"]].copy()
    zones["coordinates_m"] = zones["geometry"].map(parse_coordinates_m)
    zones = zones.drop(columns="geometry")
    bounds = pd.DataFrame.from_records(
        [get_bounds_m(points) for points in zones["coordinates_m"]],
        index=zones.index,
    )
    return pd.concat([zones, bounds], axis=1)


def get_zone_info(
    zone_info_path: str | Path,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and filter farm zones, returning vertices and dimensions in metres.

    Parameters
    ----------
    zone_info_path:
        Path to the semicolon-separated ``zone_info.csv`` file.
    csv_path:
        Optional output path. If provided, save the returned zone information
        as a CSV file. ``coordinates_m`` is serialized as text.

    Returns
    -------
    pandas.DataFrame
        One row per zone. ``coordinates_m`` contains the ordered
        polygon vertices as ``(x, y)`` tuples; bounds and dimensions are
        provided in separate metre-valued columns.

    Examples
    --------
    >>> info = get_zone_info("data/zone_info.csv")
    >>> info = get_zone_info("data/zone_info.csv", "data/zones.csv")
    """
    zones = _load_zone_coordinates(zone_info_path)
    # Keep the existing farm-area bounds filter, applied to every zone type.
    zones_filtered = zones.loc[
        (zones["x_min_m"] > 150) & (zones["y_min_m"] > -10)
    ]
    if csv_path is not None:
        zones_filtered.to_csv(csv_path, index=False)
    return zones_filtered.reset_index(drop=True)


def plot_zones(
    zones: pd.DataFrame | str | Path,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot zones from a DataFrame or a CSV exported by
    :func:`get_zone_info`.

    Parameters
    ----------
    zones:
        The filtered DataFrame returned by :func:`get_zone_info`, or
        the path to a CSV written by that function.

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

    fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)

    zone_colors = {
        "WalkingAlley": "#eeeeee",
        "FeedTable": "#91d18b",
        "Cubicles": "#a6cee3",
        "MilkingArea": "#fdbf6f",
        "HoldingArea": "#cab2d6",
        "Other": "#bdb76b",
        "Sort_TreatmentArea": "#fb9a99",
    }
    zones_to_plot = zones.assign(
        _plot_order=zones["zone_type"].eq("FeedTable").astype(int)
    ).sort_values("_plot_order")
    seen_zone_types = set()

    for _, row in zones_to_plot.iterrows():
        zone_type = (
            row["zone_type"] if pd.notna(row["zone_type"]) else "Unspecified"
        )
        x, y = zip(*row["coordinates_m"])
        ax.fill(
            x,
            y,
            color=zone_colors.get(zone_type, "#8c8c8c"),
            edgecolor="#444444" if zone_type == "WalkingAlley" else "#333333",
            linewidth=1.3 if zone_type == "WalkingAlley" else 0.8,
            alpha=0.65 if zone_type != "WalkingAlley" else 0.5,
            zorder=2 if zone_type == "FeedTable" else 1,
            label=zone_type if zone_type not in seen_zone_types else None,
        )
        seen_zone_types.add(zone_type)

    zones_to_label = zones_to_plot.loc[
        zones_to_plot["zone_type"].eq("FeedTable")
    ]
    for _, row in zones_to_label.iterrows():
        ax.annotate(
            row["zone_name"].removeprefix("rect"),
            xy=(
                (row["x_min_m"] + row["x_max_m"]) / 2,
                row["y_max_m"],
            ),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#174d22",
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
            arrowprops={"arrowstyle": "-", "color": "#267a35", "lw": 0.8},
            zorder=3,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Farm-local X (m)")
    ax.set_ylabel("Farm-local Y (m)")
    ax.set_title("Farm layout by zone type")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.legend(loc="upper right", frameon=True)
    ax.margins(0.02)
    plt.show()
    return fig, ax


if __name__ == "__main__":
    zones = get_zone_info("data/zone_info.csv", "data/zones_filtered.csv")
    plot_zones(zones)
