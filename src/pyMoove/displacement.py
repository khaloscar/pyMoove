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

def detect_replacements_by_bin_replacement(positions: pd.DataFrame, bin_statistics: pd.DataFrame, bin: int, max_bin_distance: float=1.0, replacement_criterion:int=26) -> pd.DataFrame:
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

    # --- 3-5. Scan for replacement events with pair-specific cooldown ---
    events = []
    cooldown_until = {}  # Tracks cooldown separately for each cow pair

    for i in range(len(nearby) - 1):
        t_i = float(nearby.at[i, 'Time'])

        cow_i = str(nearby.at[i, 'IDs'])
        t_j = float(nearby.at[i + 1, 'Time'])
        cow_j = str(nearby.at[i + 1, 'IDs'])

        interval = t_j - t_i

        # Identify the cow pair (order-independent)
        cow_pair = tuple(sorted((cow_i, cow_j)))

        # Skip if this pair is still within its cooldown
        if t_i < cooldown_until.get(cow_pair, -np.inf):
            continue

        if cow_i != cow_j and 0 <= interval <= replacement_criterion:
            events.append({
                'reactor_cow': cow_i,
                'bin': bin,
                'time': nearby.at[i, 'datetime'],
                'actor_cow': cow_j,
                'bout_interval': float(interval),
            })

            # Apply 60-second cooldown only to this cow pair
            cooldown_until[cow_pair] = t_i + 60.0

    if not events:
        return pd.DataFrame(columns=['reactor_cow', 'bin', 'time', 'actor_cow', 'bout_interval'])

    result = pd.DataFrame(events)
    result['bin'] = result['bin'].astype('int32')
    result['bout_interval'] = result['bout_interval'].astype('float64')
    result['time'] = pd.to_datetime(result['time'], utc=True).astype('datetime64[us, UTC]')
    result['reactor_cow'] = result['reactor_cow'].astype('string')
    result['actor_cow'] = result['actor_cow'].astype('string')

    return result

def detect_replacements_by_bin_hug(positions: pd.DataFrame, bin_statistics: pd.DataFrame, bin: int, max_bin_distance: float=1.0) -> pd.DataFrame:
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
    events = []
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

                events.append({
                    'reactor_cow': cow_u,
                    'bin': bin,
                    'time': datetime,
                    'actor_cow': cow_v,
                    'bout_interval': 0.0,
                })

                # Apply 60-second cooldown only to this cow pair
                cooldown_until[cow_pair] = time + 60.0

    if not events:
        return pd.DataFrame(columns=['reactor_cow', 'bin', 'time', 'actor_cow', 'bout_interval'])

    result = pd.DataFrame(events)
    result['bin'] = result['bin'].astype('int32')
    result['bout_interval'] = result['bout_interval'].astype('float64')
    result['time'] = pd.to_datetime(result['time'], utc=True).astype('datetime64[us, UTC]')
    result['reactor_cow'] = result['reactor_cow'].astype('string')
    result['actor_cow'] = result['actor_cow'].astype('string')

    return result

def detect():

    pass

def check_replacement_recall(
    replacements: pd.DataFrame,
    detected_replacements: pd.DataFrame,
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
        return 0.0

    if detected_replacements.empty:
        return 0.0

    # Ensure timestamps are datetime objects
    actual = replacements.copy()
    detected = detected_replacements.copy()

    actual['time'] = pd.to_datetime(actual['time'], utc=True)
    detected['time'] = pd.to_datetime(detected['time'], utc=True)

    detected_count = 0

    for _, replacement in actual.iterrows():
        # Find detections involving the same two cows
        if cow_role_matters:
            cow_match = (
                (detected['reactor_cow'] == replacement['reactor_cow']) &
                (detected['actor_cow'] == replacement['actor_cow'])
            )
        else:
            cow_match = (
                (
                    (detected['reactor_cow'] == replacement['reactor_cow']) &
                    (detected['actor_cow'] == replacement['actor_cow'])
                ) |
                (
                    (detected['reactor_cow'] == replacement['actor_cow']) &
                    (detected['actor_cow'] == replacement['reactor_cow'])
                )
            )

        matching_events = detected[
            cow_match &
            (detected['bin'] == replacement['bin'])
        ]

        # Check whether any detection is within the acceptance interval
        time_difference = (
            matching_events ['time'] - replacement['time']
        ).dt.total_seconds().abs()

        if (time_difference <= acceptance_interval).any():
            detected_count += 1
            if show_detections:
                print(f'Detected replacement between {replacement['actor_cow']} and {replacement['reactor_cow']} at {replacement['time']}')

    print(f"Actual replacements: {len(replacements)}")
    print(f"Detected replacements: {len(detected_replacements)}")
    print(f"{detected_count} / {len(actual)} replacement events were detected")

    return detected_count / len(actual)
