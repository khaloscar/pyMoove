"""Build gos event records from replacement and position data."""

import numpy as np
import pandas as pd
import pyMoove.validation as mooveValidate


#TODO: GOS event ID generation function should be included for consistency, 
#TODO: Replacements dataframe is guaranteed to contain 'repl_id' column, 
# therefore gos-events which is also replacements can be easily flagged or kept track of
# or someting?
_GOS_EVENT_COLUMNS = [
    "cow_1_id",
    "cow_2_id",
    "bin",
    "timestamps",
    "positions",
    "velocities",
]

#######################################################
# Creation of gos-events
#######################################################


#NOTE: This function for creating gos-events can be improved.
def build_gos_event_table(
    replacements: pd.DataFrame,
    positions: pd.DataFrame,
    print_skipped: bool = False,
    relative_start_time: int = 30,
    relative_end_time: int = 30
) -> pd.DataFrame:
    """Construct GOS event records from replacement and position DataFrames.

    For each replacement event, extract the actor's and reactor's
    position trajectories over the interval [time, time + bout_interval].

    Only observations with matching timestamps for both cows are retained.
    Events without synchronized observations are skipped.

    The actor is stored as cow 1 and the reactor as cow 2.

    Parameters
    ----------
    replacements : pd.DataFrame
        Replacement events containing cow IDs, event times, feeding-bin IDs,
        and bout intervals.
    positions : pd.DataFrame
        Position observations with cow IDs, timestamps, coordinates,
        and velocities.
    print_skipped : bool, default False
        Print a summary of events skipped due to missing synchronized data.

    Returns
    -------
    pd.DataFrame
        One row per successfully constructed GOS event.
    """

    #validate replacements and positions 
    mooveValidate.check_replacement_dataframe_structure(replacements)
    mooveValidate.check_positions_dataframe_structure(positions)

    #NOTE: How does this differ from the above, and why is this needed?
    #grab the data and make sure it is in right format
    position_data = positions[["IDs", "datetime", "x", "y", "velocity"]].copy()
    position_data["IDs"] = position_data["IDs"].astype("string")
    position_data["datetime"] = pd.to_datetime(position_data["datetime"], utc=True, errors="raise")
    for column in ("x", "y", "velocity"):
        position_data[column] = pd.to_numeric(
            position_data[column], errors="raise"
        )

    #NOTE: Why is this useful ?
    #checking no duplicates
    if position_data.duplicated(["IDs", "datetime"]).any():
        raise ValueError("positions must have at most one row per cow and timestamp")

    #NOTE: Useful why?
    #checking that all positions are finite
    position_data = position_data.loc[
        np.isfinite(position_data[["x", "y"]]).all(axis=1)
    ]

    #NOTE: Purpose here?
    cow_tracks = {
        str(cow_id): group.set_index("datetime")[["x", "y", "velocity"]].sort_index()
        for cow_id, group in position_data.groupby("IDs", sort=False)
    }

    #NOTE: Why the utc thing what happens here?
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
        start = pd.to_datetime(event["time"], utc=True, errors="raise") - pd.Timedelta(seconds=relative_start_time)
        bout_interval = float(event["bout_interval"])
        if not np.isfinite(bout_interval) or bout_interval < 0:
            raise ValueError(
                f"replacement row {row_index} has an invalid bout_interval"
            )
        end = start + pd.to_timedelta(bout_interval, unit="s") + pd.Timedelta(seconds=relative_end_time)

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

        #NOTE: Why 'timestamps' and not 'time'? What is the difference really ? Is this not or is it
        # consistent w/ how 'time'-column is defined inside replacements dataframe?
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
    return event_table
