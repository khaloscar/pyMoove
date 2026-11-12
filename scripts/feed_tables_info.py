"""Load, inspect, and plot feed-table zones from zone_info.csv."""

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


def get_feed_table_info(
    zone_info_path: str | Path,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return feed-table vertices and dimensions in metres.

    Parameters
    ----------
    zone_info_path:
        Path to the semicolon-separated ``zone_info.csv`` file.
    csv_path:
        Optional output path. If provided, save the returned feed-table
        information as a CSV file. ``coordinates_m`` is serialized as text.

    Returns
    -------
    pandas.DataFrame
        One row per FeedTable zone. ``coordinates_m`` contains the ordered
        polygon vertices as ``(x, y)`` tuples; bounds and dimensions are
        provided in separate metre-valued columns.

    Examples
    --------
    >>> info = get_feed_table_info("data/zone_info.csv")
    >>> info = get_feed_table_info("data/zone_info.csv", "data/feed_tables.csv")
    """
    zones = _load_zone_coordinates(zone_info_path)
    feed_tables = zones.loc[zones["zone_type"].eq("FeedTable")].reset_index(
        drop=True
    )

    if csv_path is not None:
        feed_tables.to_csv(csv_path, index=False)

    return feed_tables


def plot_feed_tables(
    zone_info_path: str | Path,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot feed-table polygons over the walking-alley zones.

    Parameters
    ----------
    zone_info_path:
        Path to the semicolon-separated ``zone_info.csv`` file.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes, allowing further customization or saving.
    """
    zones = _load_zone_coordinates(zone_info_path)
    feed_tables = zones.loc[zones["zone_type"].eq("FeedTable")]
    walking_alleys = zones.loc[zones["zone_type"].eq("WalkingAlley")]

    fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)

    for index, row in walking_alleys.iterrows():
        x, y = zip(*row["coordinates_m"])
        ax.fill(
            x,
            y,
            color="#eeeeee",
            edgecolor="#444444",
            linewidth=1.3,
            zorder=1,
            label="Walking-alley zones"
            if index == walking_alleys.index[0]
            else None,
        )

    for index, row in feed_tables.iterrows():
        x, y = zip(*row["coordinates_m"])
        ax.fill(
            x,
            y,
            color="#91d18b",
            edgecolor="#267a35",
            linewidth=1.0,
            zorder=2,
            label="Feed tables" if index == feed_tables.index[0] else None,
        )
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
    ax.set_title("Farm layout: walking alleys and feed tables")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.legend(loc="upper right", frameon=True)
    ax.margins(0.02)
    plt.show()
    return fig, ax
