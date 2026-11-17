"""Feature-design skeleton for event-level displacement classification.

The functions below document the planned inputs and outputs. Their feature
calculations will be implemented after the gos-event schema is agreed.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd


#PLANNED: 
# Cow-to-cow distance: minimum, mean, maximum, start, end, and standard deviation.
# Each cow’s distance to the feeding bin: minimum, mean, maximum, start, end, and standard deviation.
# Maybe movement features: each cow’s mean and maximum speed, path length, and net displacement.

def summarize_distance_series(
    distances: Sequence[float],
    feature_name: str,
) -> dict[str, float]:
    """Summarize one time-ordered distance sequence.

    Parameters
    ----------
    distances:
        One numeric distance in metres per event timestamp, in time order.
        Missing values, if present, need a consistent handling policy.
    feature_name:
        Prefix for the returned columns, e.g. ``cow_distance``.

    Returns
    -------
    dict[str, float]
        Planned keys: ``{feature_name}_min_m``, ``{feature_name}_mean_m``,
        ``{feature_name}_max_m``, ``{feature_name}_start_m``,
        ``{feature_name}_end_m``, and ``{feature_name}_std_m``. The start and
        end values refer to the first and last valid observations under the
        eventual missing-value policy.
    """
    raise NotImplementedError("Distance-series summaries are not implemented yet.")


def cow_to_cow_features(event: Mapping[str, Any]) -> dict[str, float]:
    """Calculate event-level features describing separation of the focal pair.

    Parameters
    ----------
    event:
        The ``gos_event`` mapping with aligned ``actor_pos`` and
        ``reactor_pos`` arrays, each shaped ``(n_times, 2)`` in metres. Each
        position is an ``(x, y)`` pair in the same farm-local coordinate frame.

    Returns
    -------
    dict[str, float]
        Planned keys ``cow_distance_min_m``, ``cow_distance_mean_m``,
        ``cow_distance_max_m``, ``cow_distance_start_m``,
        ``cow_distance_end_m``, and ``cow_distance_std_m``. Each value
        summarizes the pair's distance over the whole event, in metres.
    """
    raise NotImplementedError("Cow-to-cow features are not implemented yet.")


def cow_to_bin_features(sample: Mapping[str, Any]) -> dict[str, float]:
    """Calculate event-level distances from each focal cow to its feeding bin.

    Parameters
    ----------
    sample:
        Full training sample containing ``dist_bin.actor`` and
        ``dist_bin.reactor`` sequences aligned with ``gos_event.times``. These
        sequences are assumed to be in metres and use the same bin-distance
        definition for all events (for example, zero inside the bin polygon,
        otherwise distance to its boundary). If multiple bins are possible,
        the sample must already identify which bin is associated with the
        event, or provide distances according to a consistent selection rule.

    Returns
    -------
    dict[str, float]
        Planned ``actor_bin_distance_*`` and ``reactor_bin_distance_*`` keys
        for minimum, mean, maximum, start, end, and standard deviation, all in
        metres. The precise key suffixes follow ``summarize_distance_series``.
    """
    raise NotImplementedError("Cow-to-bin features are not implemented yet.")


def movement_features(event: Mapping[str, Any]) -> dict[str, float]:
    """Calculate optional movement summaries for each focal cow.

    Parameters
    ----------
    event:
        The ``gos_event`` mapping with positions and aligned timestamps.
        Required fields are ``times``, ``actor_pos``, and ``reactor_pos``.
        Positions are farm-local coordinates in metres. Timestamps must be
        convertible to elapsed seconds, and sequence lengths must match.

    Returns
    -------
    dict[str, float]
        Optional later-version features, with one named numeric value per
        summary and cow: mean/maximum speed (m/s), path length (m), and net
        displacement (m). This family can return an empty mapping for the
        initial distance-only baseline.
    """
    raise NotImplementedError("Movement features are not implemented yet.")


def gos_event_to_features(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Combine one labeled event's feature families into one flat record.

    Parameters
    ----------
    sample:
        One complete sample following the module-level schema, including the
        event, aligned position and bin-distance sequences, and binary label.

    Returns
    -------
    dict[str, Any]
        One flat record: fixed numeric summary features, separate ``event_id``,
        ``actor_id``, and ``reactor_id`` metadata, and the binary ``label``.
        Metadata and label must not be included in the predictor matrix.
    """
    raise NotImplementedError("Gos-event feature assembly is not implemented yet.")


def build_feature_table(
    events: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build the event-level table consumed by the modeling step.

    Parameters
    ----------
    events:
        Iterable of labeled samples following the module-level schema. Each
        item represents one candidate event, and each item becomes exactly one
        table row. An empty iterable should produce an empty table with the
        documented columns.

    Returns
    -------
    pandas.DataFrame
        One row per event; fixed numeric feature columns; separate ``event_id``,
        ``actor_id``, and ``reactor_id`` metadata columns; and a binary
        ``label`` target column. Feature columns should have the same names and
        order for every event.
    """
    raise NotImplementedError("Feature-table construction is not implemented yet.")

