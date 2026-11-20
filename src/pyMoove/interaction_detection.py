from collections.abc import Mapping
import numpy as np
import pandas as pd

#Column information for data['positions']:
#                        dtype
#NIDs                   string
#IDs                    string      This is the cow_id we are interested in
#Time                  float64
#x                     float64
#y                     float64
#Inter                    bool
#velocity              float64
#datetime  datetime64[us, UTC]

# Parameters
# Inter cow distance
# Feeding bin distance
# Interaction time criterion


def detect_bin_interactions_by_bin(positions: pd.DataFrame, 
                                   bin_statistics: pd.DataFrame, 
                                   bin: int, max_bin_distance: float=1.0, 
                                   bout_interval: int=20, 
                                   cooldown: int=30) -> pd.DataFrame:
    """Detect feeding-bin displacement (replacement) events from raw position data.

    For a given feeding bin, scans all position fixes near the bin and identifies
    moments where one cow is immediately followed by a different cow — a proxy
    for a social displacement / replacement event.

    Parameters
    ----------
    data:
        Processed dataset; must contain a ``'positions'`` DataFrame with columns
        ``IDs``, ``Time``, ``x``, ``y``, and ``datetime``.
    bin_statistics:
        DataFrame produced by ``calculate_bin_position_statistics`` that contains
        at least ``bin``, ``centroid_x``, and ``centroid_y`` columns.
    bin:
        The bin number to analyse.
    max_bin_distance:
        Maximum Euclidean distance (metres) from the bin centroid for a position
        fix to be considered "at the bin".  Default is 1.0 m.
    replacement_criterion:
        Maximum number of seconds allowed between two consecutive fixes by
        *different* cows for the transition to count as a replacement event.
        Default is 26 s.

    Returns
    -------
    pd.DataFrame
        One row per detected replacement event with columns matching
        ``data['replacements']``:

        - ``reactor_cow``  – cow that was present first (displaced)
        - ``bin``          – bin number (int32)
        - ``time``         – datetime of the first fix (datetime64[us, UTC])
        - ``actor_cow``    – cow that arrived next (displacer)
        - ``bout_interval``– elapsed seconds between the two fixes (float64)
    """

    # --- 1. Get bin centroid and filter to positions within max_bin_distance ---
    bin_row = bin_statistics.loc[bin_statistics['bin'] == bin]
    if bin_row.empty or pd.isna(bin_row.iloc[0]['centroid_x']):
        return pd.DataFrame(columns=['reactor_cow', 'bin', 'time', 'actor_cow', 'bout_interval'])

    cx = float(bin_row.iloc[0]['centroid_x'])
    cy = float(bin_row.iloc[0]['centroid_y'])

    dist = np.sqrt((positions['x'] - cx) ** 2 + (positions['y'] - cy) ** 2)
    nearby = positions.loc[dist <= max_bin_distance].copy()

    if nearby.empty:
        return pd.DataFrame(columns=['reactor_cow', 'bin', 'time', 'actor_cow', 'bout_interval'])

    # --- 2. Sort by time ---
    nearby = nearby.sort_values('Time').reset_index(drop=True)
    times = sorted(positions['Time'].unique())

    # --- 3-5. Scan for replacement events with pair-specific cooldown ---
    bin_interactions = []
    cooldown_until = {}  # Tracks cooldown separately for each cow pair
    i = 0

    while i < len(nearby) - 1:
        time = float(nearby.at[i, 'Time'])
        datetime = nearby.at[i, 'datetime']

        #If there is only one cow close to the bin at that timepoint, skip to the next one
        if time != float(nearby.at[i + 1, 'Time']):
            i += 1
            continue

        cows = []
        while (i < len(nearby) and float(nearby.at[i, 'Time']) == time):
            cow_i = str(nearby.at[i, 'IDs'])
            cows.append(cow_i)
            i+=1

        #Iterate through every pair of cows
        for u in range(len(cows)):
            for v in range(u+1, len(cows)):
                cow_u = cows[u]
                cow_v = cows[v]
                cow_pair = tuple(sorted((cow_u, cow_v)))

                # skip if this pair is still within its cooldown
                if time < cooldown_until.get(cow_pair, -np.inf):
                    continue

                bin_interactions.append({
                    'reactor_cow': cow_u,
                    'bin': bin,
                    'time': datetime,
                    'actor_cow': cow_v,
                    'bout_interval': bout_interval,
                })

                # Apply 60-second cooldown only to this cow pair
                cooldown_until[cow_pair] = time + cooldown

    if not bin_interactions:
        return pd.DataFrame(columns=['reactor_cow', 'bin', 'time', 'actor_cow', 'bout_interval'])

    result = pd.DataFrame(bin_interactions)
    result['bin'] = result['bin'].astype('int32')
    result['bout_interval'] = result['bout_interval'].astype('float64')
    result['time'] = pd.to_datetime(result['time'], utc=True).astype('datetime64[us, UTC]')
    result['reactor_cow'] = result['reactor_cow'].astype('string')
    result['actor_cow'] = result['actor_cow'].astype('string')

    return result

def detect_bin_interactions(positions: pd.DataFrame, bin_statistics: pd.DataFrame, bins: list[int] | range, max_bin_distance: float=1.0, bout_interval: int=20) -> pd.DataFrame:
    """Detect feeding-bin displacement (replacement) events from raw position data.

    For a given feeding bin, scans all position fixes near the bin and identifies
    moments where one cow is immediately followed by a different cow — a proxy
    for a social displacement / replacement event.

    Parameters
    ----------
    data:
        Processed dataset; must contain a ``'positions'`` DataFrame with columns
        ``IDs``, ``Time``, ``x``, ``y``, and ``datetime``.
    bin_statistics:
        DataFrame produced by ``calculate_bin_position_statistics`` that contains
        at least ``bin``, ``centroid_x``, and ``centroid_y`` columns.
    bin:
        The bin number to analyse.
    max_bin_distance:
        Maximum Euclidean distance (metres) from the bin centroid for a position
        fix to be considered "at the bin".  Default is 1.0 m.
    replacement_criterion:
        Maximum number of seconds allowed between two consecutive fixes by
        *different* cows for the transition to count as a replacement event.
        Default is 26 s.

    Returns
    -------
    pd.DataFrame
        One row per detected replacement event with columns matching
        ``data['replacements']``:

        - ``reactor_cow``  – cow that was present first (displaced)
        - ``bin``          – bin number (int32)
        - ``time``         – datetime of the first fix (datetime64[us, UTC])
        - ``actor_cow``    – cow that arrived next (displacer)
        - ``bout_interval``– elapsed seconds between the two fixes (float64)
    """

    bin_interactions = []
    for bin in bins:
        bin_interactions.append(detect_bin_interactions_by_bin(positions, bin_statistics, bin, max_bin_distance, bout_interval))    

    output = pd.concat(bin_interactions)
    return output
    

def validate_bin_interactions(
    replacements: pd.DataFrame,
    bin_interactions: pd.DataFrame,
    acceptance_interval: int = 30,
    cow_role_matters: bool=False,
) -> dict[str, pd.DataFrame]:
    """Calculate the recall of detected replacement events.

    An actual replacement is considered detected if there is at least
    one detected event with the same reactor and actor cows and a
    timestamp within +/- acceptance_interval seconds.

    Returns
    -------
    float
        Recall between 0.0 and 1.0.
    """

    if replacements.empty:
        print("Replacements dataframe is empty")
        return

    if bin_interactions.empty:
        print("Bin_interactions dataframe is empty")
        return

    interaction_by_bin = {
        bin_value: group
        for bin_value, group in bin_interactions.groupby('bin')
    }

    validated_indices = []

    for _, replacement in replacements.iterrows():

        bin_events = interaction_by_bin.get(
            replacement['bin'],
            bin_interactions.iloc[0:0]
        )

        # Then find detections involving the same two cows
        if cow_role_matters:
            cow_match = (
                (bin_events['reactor_cow'] == replacement['reactor_cow']) &
                (bin_events['actor_cow'] == replacement['actor_cow'])
            )
        else:
            cow_match = (
                (
                    (bin_events['reactor_cow'] == replacement['reactor_cow']) &
                    (bin_events['actor_cow'] == replacement['actor_cow'])
                ) |
                (
                    (bin_events['reactor_cow'] == replacement['actor_cow']) &
                    (bin_events['actor_cow'] == replacement['reactor_cow'])
                )
            )

        matching_events = bin_events[cow_match]

        # Check time difference
        time_difference = (
            matching_events['time'] - replacement['time']
        ).dt.total_seconds().abs()

        validated_events = matching_events[
            time_difference <= acceptance_interval
        ]

        if not validated_events.empty:
            validated_indices.extend(validated_events.index)

    label_mask = bin_interactions.index.isin(validated_indices)
    true_bin_interactions = label_mask.sum()
    false_bin_interactions = len(bin_interactions) - true_bin_interactions

    print(f"Detected replacements: {len(bin_interactions)}")
    print(f"Actual replacements: {len(replacements)}")
    print(f"Detected replacements labeled as real: {true_bin_interactions}")
    print(f"Detected replacements labeled as not real: {false_bin_interactions}")
    print()

    return label_mask

def check_bin_interactions(
    replacements: pd.DataFrame,
    bin_interactions: pd.DataFrame,
    acceptance_interval: int = 30,
    cow_role_matters: bool=False,
    show_detections: bool = False
) -> float:
    """Calculate the recall of detected replacement events.

    An actual replacement is considered detected if there is at least
    one detected event with the same reactor and actor cows and a
    timestamp within +/- acceptance_interval seconds.

    Returns
    -------
    float
        Recall between 0.0 and 1.0.
    """
    if replacements.empty:
        print("Replacements dataframe is empty")
        return

    if bin_interactions.empty:
        print("Bin_interactions dataframe is empty")
        return

    interaction_by_bin = {
            bin_value: group
            for bin_value, group in bin_interactions.groupby('bin')
        }

    detected_count = 0

    for _, replacement in replacements.iterrows():
        # Find detections involving the same two cows
        bin_events = interaction_by_bin.get(
            replacement['bin'],
            replacements.iloc[0:0]
        )

        # Then find detections involving the same two cows
        if cow_role_matters:
            cow_match = (
                (bin_events['reactor_cow'] == replacement['reactor_cow']) &
                (bin_events['actor_cow'] == replacement['actor_cow'])
            )
        else:
            cow_match = (
                (
                    (bin_events['reactor_cow'] == replacement['reactor_cow']) &
                    (bin_events['actor_cow'] == replacement['actor_cow'])
                ) |
                (
                    (bin_events['reactor_cow'] == replacement['actor_cow']) &
                    (bin_events['actor_cow'] == replacement['reactor_cow'])
                )
            )

        matching_events = bin_events[cow_match]

        # Check time difference
        time_difference = (
            matching_events['time'] - replacement['time']
        ).dt.total_seconds().abs()

        if (time_difference <= acceptance_interval).any():
            detected_count += 1
            if show_detections:
                print(f'Detected replacement between {replacement['actor_cow']} and {replacement['reactor_cow']} at {replacement['time']}')

    print(f"Actual replacements: {len(replacements)}")
    print(f"Bin interactions: {len(bin_interactions)}")
    print(f"{detected_count} / {len(replacements)} replacement events were detected")
    print()

    return detected_count / len(replacements)
