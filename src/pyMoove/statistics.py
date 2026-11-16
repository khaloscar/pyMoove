"""Get some useful information about different datasets."""
import pandas as pd

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

    # Figure out number of cows in this set
    # 

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

    print("\nMost frequently replaced cows:")
    print(reactors.value_counts().head(5).rename_axis("Reactor ID").reset_index(name="Count").to_string(index=False))

    print("\nMost frequent actors replacing this cow:")
    print(actors.value_counts().head(5).rename_axis("Actor ID").reset_index(name="Count").to_string(index=False))

    print("\nReplacements per feeding bin:")
    print(cow_events["bin"].value_counts().sort_index().rename_axis("Bin").reset_index(name="Count").to_string(index=False))

    # How much of the position data has been interpolated for this cow?
    if not cow_positions.empty:
        print("\nPositioning statistics:")
        print(f"First observation: {cow_positions['datetime'].min()}")
        print(f"Last observation:  {cow_positions['datetime'].max()}")

        print(f"Interpolated observations: {cow_positions['Inter'].mean():.1%}")
        print(f"Median velocity: {cow_positions['velocity'].median():.3f} m/s")
        print(f"Maximum velocity: {cow_positions['velocity'].max():.3f} m/s")
