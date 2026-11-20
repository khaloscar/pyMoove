import pandas as pd
import pyMoove.validation as mooveValidate


def filter_replacement_events(replacements: pd.DataFrame, stats: pd.DataFrame)->pd.DataFrame:
    """Replacement events are assoicated to an actor cow and a reactor cow. For some events positional data may only be available for 
    one of the cows. This function removes these from the dataset.

    Inputs:
        replacements: pd.DataFrame
            Replacement events and their information
        stats: pd.DataFrame
            Statistics derived from replacement events, positions and feeding bin information using 
                statistics.compute_replacement_event_statistics()
    Outputs:
        replacements_cleaned: pd.DataFrame
        The same replacements df with invalid events filtered out.
    """

    #NOTE: This function requires the statistics derived earlier
    #NOTE: Maybe one should only cosnider the replacements and the positions,
    #TODO: Maybe it is better to keep events in which all positions have matching timestamps for both cows,
    #TODO: I.e. it is not enough to onyl consider events where atleast one of the position timestamps are synchronized.
    #NOTE: Related function: summarize_event_coverage()

    mooveValidate.check_replacement_dataframe_structure(replacements)
    mooveValidate.check_replacement_statistics_dataframe_structure(stats)
    mooveValidate.check_replacement_ids(replacements)
    mooveValidate.check_replacement_ids(stats)
    if set(replacements["repl_id"]) != set(stats["repl_id"]):
        raise ValueError(
            "Replacement events and statistics have different repl_id values."
        )

    events_before = len(replacements)
    print(f"Total events before cleaning: {events_before}")

    valid_ids_mask = (stats["n_matches"] > 0)
    valid_ids = stats[valid_ids_mask]["repl_id"]

    replacements_cleaned_mask = replacements["repl_id"].isin(valid_ids)
    replacements_cleaned = replacements[replacements_cleaned_mask].copy()

    events_after = len(replacements_cleaned)
    n_removed = events_before - events_after

    print(f"Events retained: {events_after}")
    print(f"Events removed: {n_removed}")

    return replacements_cleaned
