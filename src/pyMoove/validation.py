"""Checks for the various dataframes used, can be used to ensure that the passed dataframes has the expected structure."""
import pandas as pd

#NOTE: Currently, columns may be defined at different places, 
#NOTE: perhaps one central definition should be used and modified?
#TODO: Look into this
#NOTE: Can be added here for convenience.. ?

def _check_columns(df: pd.DataFrame, expected_columns: list[str], name: str)->None:
    """
    Verify that a given DataFrame contains at least the expected columns.
    I.e. additional columns are permitted

    Parameters
    ----------
        df: pd.DataFrame
        expected_columns: list[str]
        name: str

    Raises
    -------
        ValueError: The df does not contain all of the expected columns
        Nothing: The df contains at least all of the expected columns
    """

    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Invalid {name} DataFrame.\n"
                f"Missing columns: {sorted(missing_columns)}\n"
                f"Expected columns: {expected_columns}\n"
                f"Actual columns: {df.columns.to_list()}")


def check_replacement_dataframe_structure(df: pd.DataFrame)->None:
    """Verify that a DataFrame has the expected replacement event structure"""

    expected_columns = [
        "reactor_cow", 
        "bin",
        "time",
        "actor_cow",
        "bout_interval"
    ]

    _check_columns(df, expected_columns, "Replacements")


def check_gos_event_dataframe_structure(df: pd.DataFrame)->None:
    """Verify that a DataFrame has the expected replacement event structure"""

    expected_columns = [
        "cow_1_id",
        "cow_2_id",
        "bin",
        "timestamps",
        "positions",
        "velocities",
    ]

    _check_columns(df, expected_columns, "Gos-events")


def check_positions_dataframe_structure(df: pd.DataFrame)->None:
    """Verify that a DataFrame has the expected positions structure"""

    expected_columns = [
        "NIDs",
        "IDs",
        "Time",
        "x",
        "y",
        "Inter",
        "velocity",
        "datetime",
    ]

    _check_columns(df, expected_columns, "Positions")


def check_feeding_dataframe_structure(df: pd.DataFrame) -> None:
    """Verify that a DataFrame has the expected feeding-data columns."""

    expected_columns = [
        "cow",
        "bin",
        "start_time",
        "end_time",
        "duration",
        "intake",
        "transponder",
        "start_weight",
        "end_weight",
    ]

    _check_columns(df, expected_columns, "Feeding")


def check_bin_stats_dataframe_structure(df: pd.DataFrame) -> None:
    """Verify that a DataFrame has the expected feeding-bin statistics columns."""

    expected_columns = [
        "bin",
        "centroid_x",
        "centroid_y",
    ]

    _check_columns(df, expected_columns, "Bin statistics")


def check_replacement_statistics_dataframe_structure(df: pd.DataFrame) -> None:
    """Verify that a DataFrame has the expected replacement-statistics columns."""

    expected_columns = [
        "event_id",
        "actor_cow",
        "reactor_cow",
        "bin",
        "time",
        "bout_interval",
        "n_actor",
        "n_reactor",
        "n_matches",
        "cow_min_m",
        "cow_mean_m",
        "cow_max_m",
        "actor_bin_min_m",
        "actor_bin_mean_m",
        "actor_bin_max_m",
        "reactor_bin_min_m",
        "reactor_bin_mean_m",
        "reactor_bin_max_m",
    ]

    _check_columns(df, expected_columns, "Replacement statistics")


def check_replacement_ids(replacements: pd.DataFrame)->None:
    """Verify that replacement ID's exists, that they are not null and that they are unique"""

    column = "repl_id"

    if column not in replacements.columns:
        raise ValueError(f"The {column} column is missing in dataset")

    if replacements[column].isna().any():
        raise ValueError("There are missing replacement ID's in dataset")

    if not replacements[column].is_unique:
        raise ValueError("There are non unique replacement ID's in dataset")
