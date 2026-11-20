import pandas as pd

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

    if "repl_id" in replacements.columns:
        raise ValueError(
            "Replacement events already have column 'repl_id' "
            "Existing ID's shold be preserved and not re-created."
        )


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

    if "gos_id" in gos_events.columns:
        raise ValueError(
            "Gos-events already have column 'gos_id' "
            "Existing ID's shold be preserved and not re-created."
        )

    sortedTime = gos_events.sort_values("time", kind="stable")

    #TODO: We should create consistent naming for the timestamp/datetime/time columns whenever possible.
    dates = sortedTime["timestamps"].dt.strftime("%Y%m%d")
    event_numbers = sortedTime.groupby(dates).cumcount()
    event_numbers = event_numbers.astype(str).str.zfill(5)

    sortedTime.insert(0, "gos_id", "GOS-" + dates + "-" + event_numbers)

    return sortedTime.reset_index(drop=True)


