"""Feature extraction for event-level displacement classification."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


_FEATURE_COLUMNS = [
    "cow_to_cow_distance_min_m",
    "cow_to_cow_distance_mean_m",
    "cow_to_cow_distance_max_m",
    "cow_to_cow_distance_start_m",
    "cow_to_cow_distance_end_m",
    "cow_to_cow_distance_std_m",
    "cow_1_to_bin_distance_min_m",
    "cow_1_to_bin_distance_mean_m",
    "cow_1_to_bin_distance_max_m",
    "cow_1_to_bin_distance_start_m",
    "cow_1_to_bin_distance_end_m",
    "cow_1_to_bin_distance_std_m",
    "cow_2_to_bin_distance_min_m",
    "cow_2_to_bin_distance_mean_m",
    "cow_2_to_bin_distance_max_m",
    "cow_2_to_bin_distance_start_m",
    "cow_2_to_bin_distance_end_m",
    "cow_2_to_bin_distance_std_m",
    "cow_1_speed_mean_m_s",
    "cow_1_speed_max_m_s",
    "cow_1_path_length_m",
    "cow_1_net_displacement_m",
    "cow_2_speed_mean_m_s",
    "cow_2_speed_max_m_s",
    "cow_2_path_length_m",
    "cow_2_net_displacement_m",
]


def _event_positions(event: Mapping[str, Any]) -> np.ndarray:
    """Return event positions as an array shaped (2, n_times, 2)."""
    if "positions" in event:
        tracks = event["positions"]
        if len(tracks) != 2:
            raise ValueError("positions must contain one track for each cow")
        positions = np.stack(
            [
                np.stack(
                    [np.asarray(point, dtype=float) for point in track],
                    axis=0,
                )
                for track in tracks
            ],
            axis=0,
        )
    else:
        positions = np.stack(
            [
                np.asarray(event["actor_pos"], dtype=float),
                np.asarray(event["reactor_pos"], dtype=float),
            ],
            axis=0,
        )

    if (
        positions.ndim != 3
        or positions.shape[0] != 2
        or positions.shape[2] != 2
        or positions.shape[1] == 0
    ):
        raise ValueError("cow positions must have shape (2, n_times, 2)")
    if not np.isfinite(positions).all():
        raise ValueError("cow positions must contain finite coordinates")
    return positions


def _event_times(event: Mapping[str, Any], expected_length: int) -> pd.DatetimeIndex:
    times = event.get("timestamps", event.get("times"))
    if times is None:
        raise ValueError("event must contain timestamps")
    parsed_times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    if len(parsed_times) != expected_length:
        raise ValueError("timestamps and positions must have matching lengths")
    return parsed_times


def _event_velocities(
    event: Mapping[str, Any],
    expected_length: int,
) -> list[np.ndarray] | None:
    velocities = event.get("velocities")
    if velocities is None:
        if "actor_velocity" not in event or "reactor_velocity" not in event:
            return None
        velocities = [event["actor_velocity"], event["reactor_velocity"]]
    if len(velocities) != 2:
        raise ValueError("velocities must contain one sequence for each cow")

    result = [np.asarray(speed, dtype=float).reshape(-1) for speed in velocities]
    if any(len(speed) != expected_length for speed in result):
        raise ValueError("velocity and position sequences must have matching lengths")
    return result


def summarize_distance_series(
    distances: Sequence[float],
    feature_name: str,
) -> dict[str, float]:
    """Summarize finite distance observations in time order."""
    values = np.asarray(list(distances), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"{feature_name} has no finite observations")

    return {
        f"{feature_name}_min_m": float(np.min(values)),
        f"{feature_name}_mean_m": float(np.mean(values)),
        f"{feature_name}_max_m": float(np.max(values)),
        f"{feature_name}_start_m": float(values[0]),
        f"{feature_name}_end_m": float(values[-1]),
        f"{feature_name}_std_m": float(np.std(values)),
    }


def cow_to_cow_features(event: Mapping[str, Any]) -> dict[str, float]:
    """Summarize the distance between the two cows at each event timestamp."""
    positions = _event_positions(event)
    distances = np.linalg.norm(positions[0] - positions[1], axis=1)
    return summarize_distance_series(distances, "cow_to_cow_distance")


def cow_to_bin_features(
    event: Mapping[str, Any],
    bin_locations: pd.DataFrame,
) -> dict[str, float]:
    """Summarize each cow's distance to the centroid of the event's bin.

    ``event`` must contain a ``bin`` value. ``bin_locations`` must contain
    ``bin``, ``centroid_x``, and ``centroid_y`` columns, as returned by
    ``calculate_bin_position_statistics``.
    """
    required_columns = {"bin", "centroid_x", "centroid_y"}
    missing_columns = required_columns.difference(bin_locations.columns)
    if missing_columns:
        raise ValueError(
            f"bin_locations is missing columns: {sorted(missing_columns)}"
        )
    if "bin" not in event:
        raise ValueError("event is missing the 'bin' value")

    event_bin = event["bin"]
    matching_bin = bin_locations.loc[
        bin_locations["bin"].eq(event_bin),
        ["centroid_x", "centroid_y"],
    ]
    if matching_bin.empty:
        raise ValueError(f"bin_locations contains no centroid for bin {event_bin!r}")
    if len(matching_bin) > 1:
        raise ValueError(f"bin_locations contains multiple centroids for bin {event_bin!r}")

    centroid = matching_bin.iloc[0].to_numpy(dtype=float)
    if not np.isfinite(centroid).all():
        raise ValueError(f"bin {event_bin!r} has no finite centroid coordinates")

    positions = _event_positions(event)
    features: dict[str, float] = {}
    for cow_index, cow_name in enumerate(("cow_1", "cow_2")):
        distances = np.linalg.norm(positions[cow_index] - centroid, axis=1)
        features.update(
            summarize_distance_series(distances, f"{cow_name}_to_bin_distance")
        )
    return features


def movement_features(event: Mapping[str, Any]) -> dict[str, float]:
    """Summarize speed, path length, and net displacement for each cow."""
    positions = _event_positions(event)
    times = _event_times(event, positions.shape[1])
    velocities = _event_velocities(event, positions.shape[1])
    time_deltas = np.diff(times.asi8).astype(float) / 1_000_000_000
    features: dict[str, float] = {}

    for cow_index, cow_name in enumerate(("cow_1", "cow_2")):
        track = positions[cow_index]
        segment_distances = np.linalg.norm(np.diff(track, axis=0), axis=1)
        valid_intervals = time_deltas > 0
        derived_speeds = segment_distances[valid_intervals] / time_deltas[
            valid_intervals
        ]

        if velocities is not None:
            observed_speeds = velocities[cow_index]
            observed_speeds = observed_speeds[np.isfinite(observed_speeds)]
        else:
            observed_speeds = np.array([], dtype=float)
        speeds = observed_speeds if len(observed_speeds) else derived_speeds

        features[f"{cow_name}_speed_mean_m_s"] = (
            float(np.mean(speeds)) if len(speeds) else float("nan")
        )
        features[f"{cow_name}_speed_max_m_s"] = (
            float(np.max(speeds)) if len(speeds) else float("nan")
        )
        features[f"{cow_name}_path_length_m"] = float(np.sum(segment_distances))
        features[f"{cow_name}_net_displacement_m"] = float(
            np.linalg.norm(track[-1] - track[0])
        )

    return features


def gos_event_to_features(
    sample: Mapping[str, Any],
    bin_locations: pd.DataFrame,
) -> dict[str, Any]:
    """Combine spatial and movement features for one event."""
    features: dict[str, Any] = {}
    features.update(cow_to_cow_features(sample))
    features.update(cow_to_bin_features(sample, bin_locations))
    features.update(movement_features(sample))
    if "label" in sample:
        features["label"] = sample["label"]
    return features


def build_feature_table(
    events: Iterable[Mapping[str, Any]] | pd.DataFrame,
    bin_locations: pd.DataFrame,
) -> pd.DataFrame:
    """Build one feature row per event, retaining labels when present."""
    if isinstance(events, pd.DataFrame):
        event_records: list[Mapping[str, Any]] = [
            {str(key): value for key, value in record.items()}
            for record in events.to_dict(orient="records")
        ]
    else:
        event_records = list(events)

    records = [
        gos_event_to_features(event, bin_locations)
        for event in event_records
    ]

    columns = list(_FEATURE_COLUMNS)
    if any("label" in record for record in records):
        columns.append("label")
    return pd.DataFrame.from_records(records, columns=columns)
