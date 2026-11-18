"""Plot cow positioning data and calculate bin spatial statistics during feeding events."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
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

#NOTE: Ändrat i scriptet så att jag kan spara datan sen som en parquet och ladda senare
#NOTE: Vore bra om det fanns en funktion att använda där man kan skicka in replacements och positions och 
#NOTE: få ut en ny dataFrame som man sen kan spara som en parquet.
# data = mooveIO.load_processed_dataset(DATA_DIR)

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
    outlier_confidence: float = 0.999,
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



def filter_bin_position_outliers(
    positions: pd.DataFrame,
    confidence: float = 0.999,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None, np.ndarray | None]:
    """Filter 2D positional outliers using robust MCD covariance and Mahalanobis distance.

    Returns (inliers, outliers, robust_center, robust_covariance). If there are
    too few valid points or the covariance is degenerate, all valid points are
    retained rather than risking removal of good observations.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    valid = positions.dropna(subset=["x", "y"]).copy()
    coords = valid[["x", "y"]].to_numpy(dtype=float)
    finite = np.isfinite(coords).all(axis=1)
    valid = valid.loc[finite].copy()
    coords = coords[finite]

    if len(coords) < 5 or np.unique(coords, axis=0).shape[0] < 3:
        valid["mahalanobis_sq"] = np.nan
        valid["is_inlier"] = True
        return valid, valid.iloc[0:0].copy(), None, None

    try:
        estimator = MinCovDet(random_state=42).fit(coords)
        distances_sq = estimator.mahalanobis(coords)
        threshold = float(chi2.ppf(confidence, df=2))
        is_inlier = distances_sq <= threshold

        valid["mahalanobis_sq"] = distances_sq
        valid["is_inlier"] = is_inlier
        return (
            valid.loc[is_inlier].copy(),
            valid.loc[~is_inlier].copy(),
            estimator.location_.copy(),
            estimator.covariance_.copy(),
        )
    except (ValueError, np.linalg.LinAlgError):
        valid["mahalanobis_sq"] = np.nan
        valid["is_inlier"] = True
        return valid, valid.iloc[0:0].copy(), None, None


def calculate_bin_position_statistics(
    data: dict[str, pd.DataFrame],
    bins: list[int] | int | range,
    tz: str = "Europe/Stockholm",
    outlier_confidence: float = 0.9999,
) -> pd.DataFrame:
    """Calculate the spatial centroid and standard deviation for each bin during feeding.

    For each specified bin, finds all feeding visits, retrieves the cow's
    position fixes during each visit, and computes the spatial centroid
    (mean x, mean y) and standard deviation (std x, std y).

    Parameters
    ----------
    data:
        Processed dataset containing 'feeding' and 'positions' DataFrames.
    bins:
        A bin number, list of bin numbers, or range of bin numbers.
    tz:
        Timezone of the feeder timestamps (default: 'Europe/Stockholm').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'bin': Bin number
        - 'centroid_x': Mean X coordinate in meters
        - 'centroid_y': Mean Y coordinate in meters
        - 'std_x': Standard deviation in X (meters)
        - 'std_y': Standard deviation in Y (meters)
        - 'num_fixes': Total number of position fixes during visits to this bin
        - 'num_visits': Total feeding visits recorded at this bin
        - 'num_cows': Number of unique cows that visited this bin
    """
    if isinstance(bins, (int, str)):
        bin_list = [int(bins)]
    else:
        bin_list = [int(b) for b in bins]

    feeding_df = data["feeding"]
    positions_df = data["positions"]

    # Filter feeding events for the requested bins
    target_events = feeding_df.loc[feeding_df["bin"].isin(bin_list)].copy()
    if target_events.empty:
        return pd.DataFrame(
            columns=[
                "bin",
                "centroid_x",
                "centroid_y",
                "std_x",
                "std_y",
                "num_fixes",
                "num_visits",
                "num_cows",
                "num_inliers",
                "num_outliers",
            ]
        )

    # Pre-filter positions by the cows that visited these bins for high query speed
    cows_at_bins = set(target_events["cow"].astype(str).unique())
    pos_subset = positions_df.loc[
        positions_df["IDs"].astype(str).isin(cows_at_bins)
    ].copy()
    date_str = str(positions_df["datetime"].min().date())

    # Group positions by cow ID for fast lookups
    pos_by_cow = dict(tuple(pos_subset.groupby(pos_subset["IDs"].astype(str))))

    results = []
    for bin_id in bin_list:
        b_events = target_events.loc[target_events["bin"] == bin_id]
        if b_events.empty:
            results.append({
                "bin": bin_id,
                "centroid_x": None,
                "centroid_y": None,
                "std_x": None,
                "std_y": None,
                "num_fixes": 0,
                "num_visits": 0,
                "num_cows": 0,
                "num_inliers": 0,
                "num_outliers": 0,
            })
            continue

        b_fixes = []
        for _, event in b_events.iterrows():
            cow_id = str(event["cow"])
            if cow_id not in pos_by_cow:
                continue
            cow_pos = pos_by_cow[cow_id]

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

            pts = cow_pos.loc[cow_pos["datetime"].between(start_dt_utc, end_dt_utc)]
            if not pts.empty:
                b_fixes.append(pts[["x", "y"]])

        if b_fixes:
            all_b = pd.concat(b_fixes, ignore_index=True)
            inliers, outliers, _, _ = filter_bin_position_outliers(
                all_b, confidence=outlier_confidence
            )
            results.append({
                "bin": bin_id,
                "centroid_x": float(inliers["x"].mean()) if not inliers.empty else None,
                "centroid_y": float(inliers["y"].mean()) if not inliers.empty else None,
                "std_x": float(inliers["x"].std()) if len(inliers) > 1 else None,
                "std_y": float(inliers["y"].std()) if len(inliers) > 1 else None,
                "num_fixes": len(all_b),
                "num_visits": len(b_events),
                "num_cows": b_events["cow"].nunique(),
                "num_inliers": len(inliers),
                "num_outliers": len(outliers),
            })
        else:
            results.append({
                "bin": bin_id,
                "centroid_x": None,
                "centroid_y": None,
                "std_x": None,
                "std_y": None,
                "num_fixes": 0,
                "num_visits": len(b_events),
                "num_cows": b_events["cow"].nunique(),
                "num_inliers": 0,
                "num_outliers": 0,
            })

    return pd.DataFrame(results)


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


#NOTE: Ändrat i scriptet så att jag kan spara datan sen som en parquet och ladda senare
#NOTE: Vore bra om det fanns en funktion att använda där man kan skicka in replacements och positions och 
#NOTE: få ut en ny dataFrame som man sen kan spara som en parquet.
# if __name__ == "__main__":
#     # Load filtered zones for farm layout
#     zones_path = DATA_DIR / "zones_filtered.csv"
#     if not zones_path.exists():
#         zones_path = DATA_DIR / "zone_info.csv"
#         zones = zone_info.get_zone_info(zones_path)
#     else:
#         zones = zones_path
#
#     # Example 1: Calculate centroids and standard deviations for a list of bins
#     bins_to_analyze = range(1, 21)
#     bin_stats = calculate_bin_position_statistics(data, bins=bins_to_analyze)
#     print("\n--- Bin Position Statistics ---")
#     print(bin_stats.to_string(index=False))
#
#     # Example 2: Plot all feeding events for a specific bin
#     # target_bin = 2
#     # fig, ax = plot_bin_feeding_positions(data, zones, bin_id=target_bin, color_by_cow=True)
#
#     # Example 3: Plot a single feeding event by index (uncomment to use):
#     # fig, ax = plot_feeding_event(data, zones, event_index=0)

#NOTE: Detta är den nya __main__ jag använder // Oscar
if __name__ == "__main__":
    data = mooveIO.load_processed_dataset(DATA_DIR)

    # Calculate statistics for all feeding bins
    bins = sorted(data["feeding"]["bin"].dropna().unique())

    bin_stats = calculate_bin_position_statistics(data, bins)

    # Save results to Parquet
    output = DATA_DIR / "bin_position_stats.parquet"
    bin_stats.to_parquet(output, index=False)

    print(bin_stats.to_string(index=False))
    print(f"\nSaved bin statistics to {output}")
