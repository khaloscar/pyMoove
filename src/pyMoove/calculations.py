import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

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

def get_bin_feeding_positions(
    data: dict[str, pd.DataFrame],
    bin_id: int,
    tz: str = "Europe/Stockholm",
    outlier_confidence: float = 0.9999,
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
    confidence: float = 0.9999,
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