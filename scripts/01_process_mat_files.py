#!/usr/bin/env python3
"""
Process all DASHlink MAT files into partitioned Parquet format.

Usage:
    python scripts/01_process_mat_files.py
    python scripts/01_process_mat_files.py --config config/config.yaml
"""

import polars as pl
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
import argparse
import re
from typing import Dict, List
from collections import defaultdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import process_mat_file

# Setup logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_source_folder(folder_name: str) -> tuple:
    """
    Parse source folder name like 'Tail_652_1' into (tail_number, batch).
    
    Returns:
        Tuple of (tail_number, batch_number) e.g., ('652', 1)
    """
    match = re.match(r'Tail_(\d+)_(\d+)', folder_name)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def process_all_mat_files(
    input_dir: Path,
    output_dir: Path,
    batch_size: int = 10,
    config_path: str = 'config/config.yaml'
) -> None:
    """
    Process all MAT files in batches.
    
    Args:
        input_dir: Directory containing MAT files
        output_dir: Directory for output Parquet files
        batch_size: Number of files to process before checkpointing
        config_path: Path to config file
    """
    # Find all MAT files
    mat_files = list(input_dir.glob('**/*.mat'))
    logger.info(f"Found {len(mat_files)} MAT files")

    if not mat_files:
        logger.error("No MAT files found!")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group files by tail number and batch
    # Structure: {tail_number: {batch: [files]}}
    files_by_tail_batch = defaultdict(lambda: defaultdict(list))

    for mat_file in mat_files:
        # Get tail and batch from parent folder name (e.g., Tail_652_1)
        folder_name = mat_file.parent.name
        tail_number, batch_num = parse_source_folder(folder_name)

        if tail_number is None:
            # Fallback: extract from filename
            tail_number = mat_file.stem[:3]
            batch_num = 1
            logger.warning(f"Could not parse folder '{folder_name}', using tail={tail_number}, batch={batch_num}")

        files_by_tail_batch[tail_number][batch_num].append(mat_file)

    # Sort files within each batch by filename
    for tail_number in files_by_tail_batch:
        for batch_num in files_by_tail_batch[tail_number]:
            files_by_tail_batch[tail_number][batch_num].sort(key=lambda f: f.stem)

    # Log structure
    logger.info(f"Found {len(files_by_tail_batch)} unique tail numbers:")
    for tail, batches in sorted(files_by_tail_batch.items()):
        batch_info = ", ".join([f"batch {b}: {len(files)} files" for b, files in
sorted(batches.items())])
        logger.info(f"  Tail {tail}: {batch_info}")

    # Track progress
    flight_metadata = []
    failed_files = []
    flights_per_tail = defaultdict(int)
    processed_count = 0
    total_files = len(mat_files)

    # Process by tail number and batch
    for tail_number in tqdm(sorted(files_by_tail_batch.keys()), desc="Processing tails"):
        batches = files_by_tail_batch[tail_number]

        for batch_num in sorted(batches.keys()):
            batch_files = batches[batch_num]

            for flight_num, mat_file in enumerate(batch_files, start=1):
                try:
                    df, metadata = process_mat_file(mat_file, config_path=config_path)

                    if df is None:
                        failed_files.append(str(mat_file))
                        continue

                    # Add batch and flight number to dataframe
                    df = df.with_columns([
                        pl.lit(batch_num).alias('batch'),
                        pl.lit(flight_num).alias('flight_number')
                    ])

                    # Save to partitioned parquet
                    # Structure: tail_number=XXX/batch=Y/flight_ZZZ.parquet
                    output_path = (
                        output_dir
                        / f"tail_number={tail_number}"
                        / f"batch={batch_num}"
                        / f"flight_{flight_num:03d}.parquet"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    df.write_parquet(output_path, compression='snappy')

                    # Add to metadata
                    metadata['batch'] = batch_num
                    metadata['flight_number'] = flight_num
                    metadata['source_folder'] = mat_file.parent.name
                    flight_metadata.append(metadata)
                    flights_per_tail[tail_number] += 1

                    processed_count += 1

                    # Checkpoint
                    if processed_count % batch_size == 0:
                        logger.info(f"Checkpoint: Processed {processed_count}/{total_files} files")
                        pl.DataFrame(flight_metadata).write_parquet(output_dir /
"checkpoint.parquet")

                except Exception as e:
                    logger.error(f"Failed to process {mat_file.name}: {e}")
                    failed_files.append(str(mat_file))

    # Create tail summary
    tail_summary = [
        {'tail_number': tail, 'num_flights': count, 'num_batches': len(files_by_tail_batch[tail])}
        for tail, count in sorted(flights_per_tail.items())
    ]
    tail_summary_df = pl.DataFrame(tail_summary)

    # Save final metadata
    if flight_metadata:
        metadata_df = pl.DataFrame(flight_metadata)

        # Join with tail summary
        metadata_df = metadata_df.join(
            tail_summary_df,
            on='tail_number',
            how='left'
        )

        metadata_path = output_dir.parent / "flight_metadata.parquet"
        metadata_df.write_parquet(metadata_path)
        logger.info(f"Flight metadata saved to {metadata_path}")

        # Save tail summary
        tail_summary_path = output_dir.parent / "tail_summary.parquet"
        tail_summary_df.write_parquet(tail_summary_path)
        logger.info(f"Tail summary saved to {tail_summary_path}")

    # Clean up checkpoint
    checkpoint_path = output_dir / "checkpoint.parquet"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Final summary
    logger.info(f"✓ Processed {len(flight_metadata)} files")
    logger.info(f"✗ Failed {len(failed_files)} files")
    logger.info("Structure created:")
    for tail in sorted(files_by_tail_batch.keys()):
        batches = files_by_tail_batch[tail]
        for batch_num in sorted(batches.keys()):
            num_flights = len(batches[batch_num])
            logger.info(f"  tail_number={tail}/batch={batch_num}/ -> {num_flights} flights")

    if failed_files:
        with open('logs/failed_files.txt', 'w') as f:
            f.write('\n'.join(failed_files))


def main():
    parser = argparse.ArgumentParser(description='Process DASHlink MAT files')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--input', help='Input directory (overrides config)')
    parser.add_argument('--output', help='Output directory (overrides config)')
    args = parser.parse_args()

    config = load_config(args.config)

    input_dir = Path(args.input if args.input else config['data']['raw_dir'])
    output_dir = Path(args.output if args.output else config['data']['processed_dir'])

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    process_all_mat_files(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=config.get('processing', {}).get('batch_size', 10),
        config_path=args.config
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()