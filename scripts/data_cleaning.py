import gzip
import json
import os
from pathlib import Path
import pandas as pd

# Define paths relative to PyMoove root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Load Tag Mapping
map_file = DATA_DIR / "Delpro taggar 20251010.xlsx"
df_map = pd.read_excel(map_file)
df_map = df_map.dropna(subset=['Örontransponder för beteendeanalys'])
tag_to_cow = dict(zip(
    df_map['Örontransponder för beteendeanalys'].astype(int).astype(str),
    df_map['Djurnr'].astype(str)
))

#Find RTLS files
rtls_files = sorted(DATA_DIR.glob('RTLS 20251010/pozyx-tags-*.json.gz'))
print(f"Processing {len(rtls_files)} RTLS files...")

records = []
total_parsed_entries = 0

for filepath in rtls_files:
    fname = filepath.name
    file_records = 0
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
            entries = data if isinstance(data, list) else [data]
            
            for entry in entries:
                total_parsed_entries += 1
                try:
                    tag_id = str(entry.get('tagId'))
                    cow_id = tag_to_cow.get(tag_id)
                    coords = entry.get('coordinates')
                    
                    if not isinstance(coords, dict):
                        continue
                    
                    x_val = coords.get('x')
                    y_val = coords.get('y')
                    
                    if x_val is None or y_val is None:
                        continue

                    records.append({
                        'tag_id': tag_id,
                        'cow_id': cow_id,
                        'timestamp': entry.get('timestamp'),
                        'x': float(x_val) / 1000.0,  # mm -> meters
                        'y': float(y_val) / 1000.0,
                    })
                    file_records += 1
                except (TypeError, ValueError):
                    continue
                    
        print(f"  Processed {fname}: {len(entries):,} entries standard -> {file_records:,} valid positions")
    except Exception as e:
        print(f"  Error opening {fname}: {e}")

df_all = pd.DataFrame(records)

# Filter for mapped cow IDs
df_clean = df_all.dropna(subset=['cow_id']).copy()

print(f"\n--- Extraction Summary ---")
print(f"Total raw JSON entries scanned: {total_parsed_entries:,}")
print(f"Valid position rows extracted: {len(df_all):,}")
print(f"Mapped position rows extracted: {len(df_clean):,}")

#Save to Parquet
out_path = DATA_DIR / "rtls_raw_combined.parquet"
df_clean.to_parquet(out_path, engine='pyarrow', compression='snappy', index=False)
print(f"Saved {len(df_clean):,} rows to {out_path}")
