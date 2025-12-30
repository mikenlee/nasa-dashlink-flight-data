#Find and remove the bad files:
"""
Usage:
    python scripts/03_remove_incomplete_parquet.py
    run before running 02_rebuild_flight_metadata.py
"""
import polars as pl
from pathlib import Path
from tqdm import tqdm

telemetry_path = Path('data/processed/telemetry')
parquet_files = list(telemetry_path.glob('**/*.parquet'))

print(f"Checking {len(parquet_files)} files...")

bad_files = []
for pq_file in tqdm(parquet_files, desc="Checking files"):
    try:
        # Just read schema - fast way to check if file is valid
        pl.read_parquet_schema(pq_file)
    except Exception as e:
        print(f"✗ Corrupted: {pq_file}")
        bad_files.append(pq_file)

print(f"\nFound {len(bad_files)} corrupted files")

# Delete them
for f in tqdm(bad_files, desc="Deleting files"):
    print(f"Deleting: {f}")
    f.unlink()

print("Done. Re-run rebuild_flight_metadata.py")

# ✗ Corrupted: data/processed/telemetry/tail_number=671/batch=8/flight_156.parquet
# ✗ Corrupted: data/processed/telemetry/tail_number=671/batch=9/flight_436.parquet
# ✗ Corrupted: data/processed/telemetry/tail_number=672/batch=2/flight_090.parquet
# ✗ Corrupted: data/processed/telemetry/tail_number=672/batch=1/flight_493.parquet