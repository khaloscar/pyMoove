"""Build gos event records from replacement and position data."""

from pathlib import Path
import numpy as np
import pandas as pd
from pyMoove.io import load_positions_parquet, load_replacements_parquet
import pyMoove.validation as mooveValidate


_GOS_EVENT_COLUMNS = [
    "cow_1_id",
    "cow_2_id",
    "bin",
    "timestamps",
    "positions",
    "velocities",
]

#######################################################
# Assigning ID's to Replacement events and Gos-events
#######################################################


def assign_replacement_id(replacements: pd.DataFrame)->pd.DataFrame:
    """
    Assign unique ID's to replacement events
    Structure of ID:
        'REPL-YYYYMMDD-XXXXX'
        Where XXXXX is the chronological event number for that day
    """


    sortedTime = replacements.sort_values("time", kind="stable")

    #NOTE: We should create consistent naming for the timestamp/datetime/time columns whenever possible
    dates = sortedTime["time"].dt.strftime("%Y%m%d")
    event_numbers = sortedTime.groupby(dates).cumcount()
    event_numbers = event_numbers.astype(str).str.zfill(5)

    sortedTime.insert(0, "repl_id", "REPL-" + dates + "-" + event_numbers)

    return sortedTime.reset_index(drop=True)


def assign_gos_id(gos_events: pd.DataFrame)->pd.DataFrame:
    """
    Assign unique ID's to GOS events
    Structure of ID:
        'GOS-YYYYMMDD-XXXXX'
        Where XXXXX is the event in chronological order for that day
    """

    sortedTime = gos_events.sort_values("time", kind="stable")

    #NOTE: We should create consistent naming for the timestamp/datetime/time columns whenever possible.
    dates = sortedTime["timestamps"].dt.strftime("%Y%m%d")
    event_numbers = sortedTime.groupby(dates).cumcount()
    event_numbers = event_numbers.astype(str).str.zfill(5)

    sortedTime.insert(0, "gos_id", "GOS-" + dates + "-" + event_numbers)

    return sortedTime.reset_index(drop=True)


#######################################################
# Creation of gos-events
#######################################################


def build_gos_event_table(
    replacements: pd.DataFrame | str | Path,
    positions: pd.DataFrame | str | Path,
    *,
    save_to_parquet: bool = False,
    output_path: str | Path | None = None,
    print_skipped: bool = False,
) -> pd.DataFrame:

    """Create one GOS-style event record per replacement.

    The actor is stored as cow 1 and the reactor as cow 2. Positions are
    restricted to the inclusive interval from ``time`` through
    ``time + bout_interval``. Only timestamps present for both cows are kept,
    so each cow's positions and velocities align with ``timestamps``. Events
    without synchronized positions are skipped. Set ``print_skipped=True`` to
    print a summary of skipped events.
    Either input may be a DataFrame or a path to its processed Parquet file.

    If ``save_to_parquet`` is true, ``output_path`` specifies where to save
    the resulting table. The table is returned whether or not it is saved.

    Parameters
    ----------
    print_skipped:
        Whether to print a summary when replacement events are skipped.

    Raises
    ------
    ValueError
        If the input columns are missing, position timestamps are duplicated
        for a cow, or ``save_to_parquet`` and ``output_path`` are inconsistent.
    """

    #checking if outputpath is set if one wants to save
    if save_to_parquet and output_path is None:
        raise ValueError("output_path is required when save_to_parquet=True")
    if not save_to_parquet and output_path is not None:
        raise ValueError("set save_to_parquet=True to write output_path")

    #if replacements/positions is not set to dataframe we load it from parquet
    if not isinstance(replacements, pd.DataFrame):
        replacements = load_replacements_parquet(replacements)
    if not isinstance(positions, pd.DataFrame):
        positions = load_positions_parquet(positions)

    #validate replacements and positions 
    mooveValidate.check_replacement_dataframe_structure(replacements)
    mooveValidate.check_positions_dataframe_structure(positions)

    #grab the data and make sure it is in right format
    position_data = positions[["IDs", "datetime", "x", "y", "velocity"]].copy()
    position_data["IDs"] = position_data["IDs"].astype("string")
    position_data["datetime"] = pd.to_datetime(position_data["datetime"], utc=True, errors="raise")
    for column in ("x", "y", "velocity"):
        position_data[column] = pd.to_numeric(
            position_data[column], errors="raise"
        )

    #checking no duplicates
    if position_data.duplicated(["IDs", "datetime"]).any():
        raise ValueError("positions must have at most one row per cow and timestamp")

    #checking that all positions are finite
    position_data = position_data.loc[
        np.isfinite(position_data[["x", "y"]]).all(axis=1)
    ]
    cow_tracks = {
        str(cow_id): group.set_index("datetime")[["x", "y", "velocity"]].sort_index()
        for cow_id, group in position_data.groupby("IDs", sort=False)
    }
    empty_track = pd.DataFrame(
        columns=["x", "y", "velocity"],
        index=pd.DatetimeIndex([], tz="UTC", name="datetime"),
    )

    records: list[dict[str, object]] = []
    skipped_count = 0
    zero_fix_counts: dict[str, int] = {}
    unsynchronized_count = 0
    for row_index, event in replacements.iterrows():
        actor_id = str(event["actor_cow"])
        reactor_id = str(event["reactor_cow"])
        start = pd.to_datetime(event["time"], utc=True, errors="raise")
        bout_interval = float(event["bout_interval"])
        if not np.isfinite(bout_interval) or bout_interval < 0:
            raise ValueError(
                f"replacement row {row_index} has an invalid bout_interval"
            )
        end = start + pd.to_timedelta(bout_interval, unit="s")

        actor_track = cow_tracks.get(actor_id, empty_track).loc[start:end]
        reactor_track = cow_tracks.get(reactor_id, empty_track).loc[start:end]
        synchronized = actor_track.join(
            reactor_track,
            how="inner",
            lsuffix="_actor",
            rsuffix="_reactor",
        )
        if synchronized.empty:
            skipped_count += 1
            if actor_track.empty:
                zero_fix_counts[actor_id] = zero_fix_counts.get(actor_id, 0) + 1
            if reactor_track.empty:
                zero_fix_counts[reactor_id] = zero_fix_counts.get(reactor_id, 0) + 1
            if not actor_track.empty and not reactor_track.empty:
                unsynchronized_count += 1
            continue

        records.append(
            {
                "cow_1_id": actor_id,
                "cow_2_id": reactor_id,
                "bin": event["bin"],
                "timestamps": synchronized.index.to_list(),
                "positions": [
                    synchronized[["x_actor", "y_actor"]].to_numpy().tolist(),
                    synchronized[["x_reactor", "y_reactor"]].to_numpy().tolist(),
                ],
                "velocities": [
                    synchronized["velocity_actor"].to_numpy(dtype=float).tolist(),
                    synchronized["velocity_reactor"].to_numpy(dtype=float).tolist(),
                ],
            }
        )

    if skipped_count > 0 and print_skipped:
        zero_fix_summary = ", ".join(
            f"{cow_id} ({count} event windows)"
            for cow_id, count in sorted(zero_fix_counts.items())
        )
        print(
            f"Skipped {skipped_count} replacement event(s) with no "
            "synchronized positions. Cows with zero valid fixes in their "
            f"event windows: {zero_fix_summary or 'none'}. "
            f"Events where both cows had fixes but no timestamps matched: "
            f"{unsynchronized_count}."
        )

    event_table = pd.DataFrame.from_records(records, columns=_GOS_EVENT_COLUMNS)
    if output_path is not None:
        event_table.to_parquet(output_path, index=False)
    return event_table
