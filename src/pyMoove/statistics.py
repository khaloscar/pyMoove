"""Get some useful information about different datasets."""
import pandas as pd
import numpy as np

#TODO: 

def inspect_dataframe(df: pd.DataFrame, name: str = "Dataset")->None:

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"\n{'='*60}")

    rows, cols = df.shape
    memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Rows: {rows}")
    print(f"Columns: {cols}")
    print(f"Memory: {memory:.2f} MiB")

    non_null = df.count()

    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": non_null,
        "missing": rows - non_null,
        "missing_%": (1 - non_null/rows)*100
        if rows else 0.0,
    })

    print("\n Column information:")
    print(summary.to_string())

    print("\nPreview 5 rows")
    print(df.head(n=5))
    print(f"\n{'='*60}\n")


def inspect_positions(df: pd.Dataframe, name: str = "Positions data")->None:
    """
    Columns for the final positions frame, after AniSpace interpolation and cleaning
    Missing values:
    NIDs          0
    IDs           0
    Time          0
    x             0
    y             0
    Inter         0
    velocity    226
    datetime      0
    dtype: int64
    """


    #TODO: ??
    print(f"TODO: inspect_positions()")


    return;


def inspect_cow_statistict(ID: str, positions: pd.DataFrame, replacements: pd.DataFrame)->None:
    """Inspect individual cow, how many times actor, which repl. events are it associated with? And so on"""
    """positions: ['NIDs', 'IDs', 'Time', 'x', 'y', 'Inter', 'velocity', 'datetime']"""
    """replacements: ['reactor_cow', 'bin', 'time', 'actor_cow', 'bout_interval']"""

    mask_positions = positions["IDs"] == ID
    cow_positions = positions[mask_positions]

    actor_mask = replacements["actor_cow"] == ID
    reactor_mask = replacements["reactor_cow"] == ID
    cow_events = replacements[actor_mask | reactor_mask]

    # Basic information about the cow, how many times is reactor and actor
    # How many total events?
    print(f"\nCow ID: {ID}")
    print(f"Position observations: {len(cow_positions):,}")

    print(f"Replacements as actor: {actor_mask.sum()}")
    print(f"Replacements as reactor: {reactor_mask.sum()}")
    print(f"Total replacement events: {len(cow_events)}")

    # All cows displaced by this cow as an actor
    reactors = replacements.loc[actor_mask, "reactor_cow"]

    # actor cows that replaced this cow
    actors = replacements.loc[reactor_mask, "actor_cow"]

    # Most frequently replaced cows
    reactor_counts = reactors.value_counts().head(5)

    print("\nMost frequently replaced cows:")
    print(f"{'Reactor ID':<12} {'Count':>6}")
    for cow, count in reactor_counts.items():
        print(f"{cow:<12} {count:>6}")

    # Most frequent actors replacing this cow
    actor_counts = actors.value_counts().head(5)

    print("\nMost frequent actors replacing this cow:")
    print(f"{'Actor ID':<12} {'Count':>6}")
    for cow, count in actor_counts.items():
        print(f"{cow:<12} {count:>6}")

    # Replacements per feeding bin
    bin_counts = cow_events["bin"].value_counts()

    print("\nReplacements per feeding bin:")
    print(f"{'Bin':<12} {'Count':>6}")
    for bin_id in sorted(bin_counts.index):
        print(f"{bin_id:<12} {bin_counts[bin_id]:>6}")

    # How much of the position data has been interpolated for this cow?
    if not cow_positions.empty:
        print("\nPositioning statistics:")
        print(f"First observation: {cow_positions['datetime'].min()}")
        print(f"Last observation:  {cow_positions['datetime'].max()}")

        print(f"Interpolated observations: {cow_positions['Inter'].mean():.1%}")
        print(f"Median velocity: {cow_positions['velocity'].median():.3f} m/s")
        print(f"Maximum velocity: {cow_positions['velocity'].max():.3f} m/s")


def inspect_replacement_event_statistics(
        positions: pd.DataFrame,
        replacements: pd.DataFrame,
        bin_stats: pd.DataFrame,
        time_before: float = 0.0,
        time_after: float = 0.0,
)->pd.DataFrame:
    #TODO: time_start -> kolla före repl. event börjar, ist. för extra padding runtomkring ett event.

    event_statistics = []

    # Associate each replacement with its feeding-bin position
    repl = replacements.merge(
        bin_stats[["bin", "centroid_x", "centroid_y"]],
        on="bin",
        how="left",
        validate="many_to_one",
    )

    # Group positions by cow once for faster access
    cow_positions = {
        str(ID): group.set_index("datetime")[["x", "y"]].sort_index()
        for ID, group in positions.groupby("IDs")
    }

    empty = pd.DataFrame(
        columns=["x", "y"],
        index=pd.DatetimeIndex([], tz="UTC", name="datetime"),
    )

    for event_id, event in repl.iterrows():

        actor_id = str(event["actor_cow"])
        reactor_id = str(event["reactor_cow"])

        t = event["time"]
        dt = pd.Timedelta(seconds=event["bout_interval"])

        # Look at a replacement event, 
        # Can look at a some time before the event and after the event
        start = t - pd.Timedelta(seconds=time_before)
        end = t + dt + pd.Timedelta(seconds=time_after)

        # Extract both trajectories
        actor_pos = cow_positions.get(actor_id, empty).loc[start:end].dropna()
        reactor_pos = cow_positions.get(reactor_id, empty).loc[start:end].dropna()

        # Match positions at identical timestamps
        matched = actor_pos.join(
            reactor_pos,
            how="inner",
            lsuffix="_actor",
            rsuffix="_reactor",
        )

        # Cow-to-cow distances
        d_cows = np.hypot(
            matched["x_actor"] - matched["x_reactor"],
            matched["y_actor"] - matched["y_reactor"],
        )

        # Cow-to-bin distances
        bx, by = event["centroid_x"], event["centroid_y"]

        d_actor_bin = pd.Series(dtype=float)
        d_reactor_bin = pd.Series(dtype=float)

        if pd.notna(bx) and pd.notna(by):
            d_actor_bin = np.hypot(actor_pos["x"] - bx, actor_pos["y"] - by)
            d_reactor_bin = np.hypot(reactor_pos["x"] - bx, reactor_pos["y"] - by)

        # Store statistics for this event
        event_statistics.append({
            "event_id": event_id,
            "actor_cow": actor_id,
            "reactor_cow": reactor_id,
            "bin": event["bin"],
            "time": t,
            "bout_interval": event["bout_interval"],

            "n_actor": len(actor_pos),
            "n_reactor": len(reactor_pos),
            "n_matches": len(matched),

            "cow_min_m": d_cows.min(),
            "cow_mean_m": d_cows.mean(),
            "cow_max_m": d_cows.max(),

            "actor_bin_min_m": d_actor_bin.min(),
            "actor_bin_mean_m": d_actor_bin.mean(),
            "actor_bin_max_m": d_actor_bin.max(),

            "reactor_bin_min_m": d_reactor_bin.min(),
            "reactor_bin_mean_m": d_reactor_bin.mean(),
            "reactor_bin_max_m": d_reactor_bin.max(),
        })

    return pd.DataFrame(event_statistics)

def summarize_event_coverage(stats: pd.DataFrame)->None:
    """
    Summarize and print positional data availability for replacement events.

    Requires the columns n_actor, n_reactor and n_matches from 
    inspect_replacement_event_statistics()

    Events are defined as invalid when no synchronized positional data
    are available for both the reactor and the actor cow.
    """

    total_events = stats.shape[0]
    matching = stats["n_matches"] > 0
    both_valid = (stats["n_actor"] > 0) & (stats["n_reactor"] > 0)
    only_actor = (stats["n_actor"] > 0) & (stats["n_reactor"] == 0)
    only_reactor = (stats["n_actor"] == 0) & (stats["n_reactor"] > 0)
    neither = (stats["n_actor"] == 0) & (stats["n_reactor"] == 0)
    no_sync = both_valid & (stats["n_matches"] == 0)
    invalid = only_actor | only_reactor | neither | no_sync

    print("\n\n")
    print("Event position summary:")
    print(f"Total events: {total_events}")
    print(f"Events with synchronized positions: {matching.sum()}")
    print(f"Both cows have at least one valid position: {both_valid.sum()}")
    print(f"Both present, but no matching timestamps: {no_sync.sum()}")
    print(f"Only actor has positions: {only_actor.sum()}")
    print(f"Only reactor has positions: {only_reactor.sum()}")
    print(f"Neither cow has positions: {neither.sum()}")
    print(f"Amount of invalid events: {invalid.sum()}")
    print("\n\n")

