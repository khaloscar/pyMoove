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
                statistics.inspect.replacement_event_statistics()
    Outputs:
        replacements_cleaned: pd.DataFrame
        The same replacements df with invalid events filtered out.
    """
    #NOTE: This function requires the statistics derived earlier
    #NOTE: Maybe one should only cosnider the replacements and the positions,
    #TODO: Maybe it is better to keep events in which all positions have matching timestamps for both cows,
    #TODO: I.e. it is not enough to onyl consider events where atleast one of the position timestamps are synchronized.
    #NOTE: Related function: summarize_event_coverage()

    if len(replacements) != len(stats):
        raise ValueError(
            f"Event count mismatch: replacementss containts {len(replacements)} events "
            f"but stats contains {len(stats)} events"
        )
    mooveValidate.check_replacement_dataframe_structure(replacements)
    mooveValidate.check_replacement_statistics_dataframe_structure(stats)

    events_before = replacements.shape[0]
    print(f"Total event before cleaning: {events_before}")


    event_columns = ["actor_cow", "reactor_cow", "bin", "time", "bout_interval"]

    valid_events = stats.loc[
        stats["n_matches"] > 0,
        event_columns,
    ]

    replacements_cleaned = replacements.merge(
        valid_events,
        on=event_columns,
        how="inner",
        validate="one_to_one"
    )

    events_after = len(replacements_cleaned)
    print(f"Events retained: {events_after}")

    nRemoved = events_before - events_after
    nRemoved_expected = (stats["n_matches"] == 0).sum()
    print(f"Expected removal: {nRemoved_expected}")
    print(f"Events removed: {nRemoved}")

    return replacements_cleaned
