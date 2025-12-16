"""
Rebuild flight_metadata.parquet from existing parquet files.
- no need to reprocess the MAT files.

Usage:
    python scripts/02_rebuild_flight_metadata.py

"""

import polars as pl
from pathlib import Path
import re
from tqdm import tqdm

def rebuild_flight_metadata(telemetry_dir: str = 'data/processed/telemetry'):
    """Rebuild flight_metadata.parquet from existing parquet files."""

    telemetry_path = Path(telemetry_dir)
    parquet_files = list(telemetry_path.glob('**/*.parquet'))

    print(f"Found {len(parquet_files)} parquet files")

    records = []
    for pq_file in tqdm(parquet_files, desc="Processing"):
        # Parse path: tail_number=652/batch=1/flight_001.parquet
        parts = pq_file.relative_to(telemetry_path).parts

        tail_match = re.search(r'tail_number=(\d+)', parts[0]) if len(parts) > 0 else None
        batch_match = re.search(r'batch=(\d+)', parts[1]) if len(parts) > 1 else None
        flight_match = re.search(r'flight_(\d+)', pq_file.stem)

        tail_number = tail_match.group(1) if tail_match else None
        batch = int(batch_match.group(1)) if batch_match else None
        flight_number = int(flight_match.group(1)) if flight_match else None

        # Read parquet to get stats (schema only, fast)
        df = pl.read_parquet(pq_file)

        records.append({
            'tail_number': tail_number,
            'batch': batch,
            'flight_number': flight_number,
            'num_samples': len(df),
            'num_signals': len(df.columns) - 1,  # exclude sample_index
            'file_path': str(pq_file),
            'size_mb': pq_file.stat().st_size / (1024 * 1024),
        })

    metadata_df = pl.DataFrame(records)

    # Add tail summary
    tail_summary = metadata_df.group_by('tail_number').agg([
        pl.count().alias('num_flights'),
        pl.n_unique('batch').alias('num_batches'),
    ])

    metadata_df = metadata_df.join(tail_summary, on='tail_number', how='left')

    # Save
    output_path = Path(telemetry_dir).parent / 'flight_metadata.parquet'
    metadata_df.write_parquet(output_path)
    print(f"Saved to {output_path}")
    print(f"Shape: {metadata_df.shape}")
    print(metadata_df.head())

    # Also save tail summary
    tail_summary_path = Path(telemetry_dir).parent / 'tail_summary.parquet'
    tail_summary.write_parquet(tail_summary_path)
    print(f"\nTail summary saved to {tail_summary_path}")
    print(tail_summary.sort('tail_number'))

    return metadata_df

# Run it
def main():
    metadata = rebuild_flight_metadata()
    return metadata

if __name__ == "__main__":
    main()