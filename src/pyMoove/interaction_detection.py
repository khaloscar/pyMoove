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


def _exchange_roles(
    cow_tracks: dict[str, tuple[np.ndarray, np.ndarray]],
    cow_1: str,
    cow_2: str,
    time: float,
    bout_interval: float,
    exchange_window: float,
    minimum_distance_change: float,
    minimum_track_fixes: int,
) -> tuple[str, str] | None:
    """Identify the departing and approaching cow in a candidate event."""
    before_start = time - exchange_window
    before_end = time
    after_start = time + bout_interval
    after_end = time + bout_interval + exchange_window

    distance_changes: dict[str, float] = {}
    before_distances: dict[str, float] = {}

    for cow_id in (cow_1, cow_2):
        track = cow_tracks.get(cow_id)
        if track is None:
            return None

        track_times, distance_prefix = track

        before_left = int(np.searchsorted(track_times, before_start, side='left'))
        before_right = int(np.searchsorted(track_times, before_end, side='right'))
        after_left = int(np.searchsorted(track_times, after_start, side='left'))
        after_right = int(np.searchsorted(track_times, after_end, side='right')) 

        before_count = before_right - before_left
        after_count = after_right - after_left
        if (
            before_count < minimum_track_fixes
            or after_count < minimum_track_fixes
        ):
            return None

        before_sum = distance_prefix[before_right] - distance_prefix[before_left]
        after_sum = distance_prefix[after_right] - distance_prefix[after_left]
        before_mean = float(before_sum / before_count)
        after_mean = float(after_sum / after_count)
        distance_changes[cow_id] = after_mean - before_mean
        before_distances[cow_id] = before_mean

    cow_1_before = before_distances[cow_1]
    cow_2_before = before_distances[cow_2]

    if cow_1_before <= cow_2_before:
        reactor = cow_1
        actor = cow_2
    else:
        reactor = cow_2
        actor = cow_1

    reactor_change = distance_changes[reactor]
    actor_change = distance_changes[actor]

    if (
        reactor_change >= minimum_distance_change
        and actor_change <= -minimum_distance_change
    ):
        return reactor, actor

    return None


def detect_bin_interactions_by_bin(positions: pd.DataFrame, 
                                   bin_statistics: pd.DataFrame, 
                                   bin: int, max_bin_distance: float=1.0, 
                                   bout_interval: int=20, 
                                   cooldown: int=30,
                                   require_exchange: bool=True,
                                   exchange_window: float=23.0,
                                   minimum_distance_change: float=0.5,
                                   minimum_track_fixes: int=2) -> pd.DataFrame:
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
    require_exchange:
        Require the candidate cows to exchange their position relative to the
        bin: one cow becomes farther away while the other becomes closer.
    exchange_window:
        Number of seconds before and after the candidate timestamp used to
        compare each cow's distance to the bin.
    minimum_distance_change:
        Minimum absolute change in mean bin distance on either side of the
        candidate timestamp.
    minimum_track_fixes:
        Minimum number of position fixes for each cow before and after the
        candidate timestamp.

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
    bin_interactions = []
    cooldown_until = {}  # Tracks cooldown separately for each cow pair
    position_data = positions.copy()
    position_data['IDs'] = position_data['IDs'].astype(str)
    position_data['Time'] = pd.to_numeric(position_data['Time'], errors='coerce')
    position_data['datetime'] = pd.to_datetime(
        position_data['datetime'],
        utc=True,
        errors='coerce',
    )
    position_data = position_data.dropna(
        subset=['IDs', 'Time', 'datetime', 'x', 'y']
    )

    position_data['bin_distance'] = np.sqrt(
        (position_data['x'] - cx) ** 2
        + (position_data['y'] - cy) ** 2
    )

    cow_tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if require_exchange:
        for cow_id, cow_positions in position_data.groupby('IDs', sort=False):
            cow_positions = cow_positions.sort_values('Time')
            track_times = cow_positions['Time'].to_numpy(dtype=float)
            track_distances = cow_positions['bin_distance'].to_numpy(dtype=float)
            prefix = np.empty(len(track_distances) + 1, dtype=float)
            prefix[0] = 0.0
            np.cumsum(track_distances, out=prefix[1:])
            cow_tracks[str(cow_id)] = (track_times, prefix)

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

                if require_exchange:
                    roles = _exchange_roles(
                        cow_tracks=cow_tracks,
                        cow_1=cow_u,
                        cow_2=cow_v,
                        time=time,
                        bout_interval=bout_interval,
                        exchange_window=exchange_window,
                        minimum_distance_change=minimum_distance_change,
                        minimum_track_fixes=minimum_track_fixes,
                    )
                    if roles is None:
                        continue
                    reactor, actor = roles
                else:
                    reactor = cow_u
                    actor = cow_v

                bin_interactions.append({
                    'repl_id': '',
                    'reactor_cow': reactor,
                    'bin': bin,
                    'time': datetime,
                    'actor_cow': actor,
                    'bout_interval': bout_interval,
                })

                # Apply 60-second cooldown only to this cow pair
                cooldown_until[cow_pair] = time + cooldown

    if not bin_interactions:
        return pd.DataFrame(columns=['repl_id', 'reactor_cow', 'bin', 'time', 'actor_cow', 'bout_interval'])

    result = pd.DataFrame(bin_interactions)
    result['repl_id'] = result['repl_id'].astype('string')
    result['bin'] = result['bin'].astype('int32')
    result['bout_interval'] = result['bout_interval'].astype('float64')
    result['time'] = pd.to_datetime(result['time'], utc=True).astype('datetime64[us, UTC]')
    result['reactor_cow'] = result['reactor_cow'].astype('string')
    result['actor_cow'] = result['actor_cow'].astype('string')

    return result

def detect_bin_interactions(
    positions: pd.DataFrame,
    bin_statistics: pd.DataFrame,
    bins: list[int] | range,
    max_bin_distance: float = 1.0,
    bout_interval: int = 20,
    cooldown: int = 30,
    require_exchange: bool = True,
    exchange_window: float = 23.0,
    minimum_distance_change: float = 0.5,
    minimum_track_fixes: int = 2,
) -> pd.DataFrame:
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
    require_exchange:
        Require one cow to move farther from the bin and the other cow closer
        over the exchange window.
    exchange_window:
        Number of seconds before and after a candidate used to compare bin
        distances.
    minimum_distance_change:
        Minimum absolute mean-distance change required for both cows.
    minimum_track_fixes:
        Minimum number of position fixes before and after the candidate time
        required for each cow.

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
        bin_interactions.append(
            detect_bin_interactions_by_bin(
                positions,
                bin_statistics,
                bin,
                max_bin_distance,
                bout_interval,
                cooldown,
                require_exchange,
                exchange_window,
                minimum_distance_change,
                minimum_track_fixes,
            )
        )

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

    #validated_bin_interactions = bin_interactions.copy()

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

        for i in validated_events.index:
            bin_interactions.loc[i, 'repl_id'] = replacement['repl_id']

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
