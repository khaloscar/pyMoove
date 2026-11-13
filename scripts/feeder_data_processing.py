import os
from pathlib import Path
import pandas as pd
import re

# Define paths relative to PyMoove root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Update these paths to match your files
feeding_file = DATA_DIR / "Utfodring 20251010.xlsx"
mapping_file = DATA_DIR / "Delpro taggar 20251010.xlsx"

feeding_df = pd.read_excel(feeding_file)
mapping_df = pd.read_excel(mapping_file)

# Make sure the column matches for the animal IDs in feeding data and tranponsder data
mapping_df = mapping_df.rename(columns={
    "Djurnr": "cow",
    "Örontransponder för beteendeanalys": "transponder"
})
feeding_df = feeding_df.rename(columns={
    "Delpro Id": "cow",
    "Roughage consumed": "intake",
    "Duration sec": "duration",
    "Equipment Name": "bin",
    "Time": "start_time"
})

# Ensure Delpro Id has matching type (string) in both DataFrames
feeding_df["cow"] = feeding_df["cow"].astype(str)
mapping_df["cow"] = mapping_df["cow"].astype(str)

feeding_df = feeding_df.drop(columns=["Date"])
feeding_df = feeding_df.drop(columns=["Group"])

# Merge using Delpro Id
df = pd.merge(
    feeding_df,
    mapping_df,
    on="cow",
    how="left"
)

# Match Gate followed by a number (e.g. Gate 1, Gate 2, Gate 10)
df = df[
    df["bin"].astype(str).str.match(
        r"^Gate \d+$",
        na=False
    )
].copy()


# Replace "Gate x" with just the number x
df["bin"] = (
    df["bin"]
    .str.extract(r"Gate (\d+)", expand=False)
    .astype(int)
)

# Calculate end_time using Duration
# Convert start_time to datetime if necessary
df["start_time"] = pd.to_datetime(df["start_time"])

# Convert Duration sec to numeric
df["duration"] = pd.to_numeric(
    df["duration"],
    errors="coerce"
)

# Calculate end_time
df["end_time"] = (
    df["start_time"] +
    pd.to_timedelta(df["duration"], unit="s")
)

# Convert datetime to time only (HH:MM:SS)
df["start_time"] = pd.to_datetime(df["start_time"]).dt.strftime("%H:%M:%S")
df["end_time"] = pd.to_datetime(df["end_time"]).dt.strftime("%H:%M:%S")

# Add start and end weight for moo4feed
# Calculate end_time
df["start_weight"] = (
    df["intake"] + 10
)
df["end_weight"] = (
    10
)

#Save to Parquet
out_path_parquet = DATA_DIR / "Feed_251010.parquet"
df.to_parquet(out_path_parquet, engine='pyarrow', compression='snappy', index=False)
out_path_csv = DATA_DIR / "Feed_251010.csv"
df.to_csv(out_path_csv, index=False)
print(f"Saved {len(df):,} rows to {out_path_parquet} and {out_path_csv}")
