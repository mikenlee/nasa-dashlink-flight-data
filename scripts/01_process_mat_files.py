"""
Process all DASHlink MAT files into partitioned Parquet format.

Usage:
    python scripts/01_process_mat_files.py
    python scripts/01_process_mat_files.py --config 
config/config.yaml
"""

import polars as pl
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
import argparse
import re
import json
from typing import Dict
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
    """
    match = re.search(r'Tail_\s*(\d+)\s*_\s*(\d+)', folder_name, flags=re.IGNORECASE)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def extract_tail_batch_from_path(mat_file: Path) -> tuple:
    """
    Extract (tail_number, batch_num) from any ancestor directory name.
    """
    for parent in mat_file.parents:
        tail_number, batch_num = parse_source_folder(parent.name)
        if tail_number is not None:
            return tail_number, batch_num
    return None, None


def process_all_mat_files(
    input_dir: Path,
    output_dir: Path,
    batch_size: int = 10,
    config_path: str = 'config/config.yaml'
) -> None:
    """
    Process all MAT files in batches.
    """
    mat_files = list(input_dir.glob('**/*.mat'))
    logger.info(f"Found {len(mat_files)} MAT files")

    if not mat_files:
        logger.error("No MAT files found!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group files by tail number and batch: {tail: {batch: [files]}}
    files_by_tail_batch = defaultdict(lambda: defaultdict(list))

    for mat_file in mat_files:
        tail_number, batch_num = extract_tail_batch_from_path(mat_file)

        if tail_number is None:
            tail_number = mat_file.stem[:3]
            batch_num = 1
            logger.warning(f"Could not parse path for '{mat_file}', using tail={tail_number}, batch={batch_num}")


        files_by_tail_batch[tail_number][batch_num].append(mat_file)

    # Sort files within each batch
    for tail_number in files_by_tail_batch:
        for batch_num in files_by_tail_batch[tail_number]:
            files_by_tail_batch[tail_number][batch_num].sort(key=lambda f:f.name)

    # Log structure
    logger.info(f"Found {len(files_by_tail_batch)} unique tail numbers")
    for tail, batches in sorted(files_by_tail_batch.items()):
        total = sum(len(files) for files in batches.values())
        logger.info(f"  Tail {tail}: {len(batches)} batches, {total} flights")

    # Process
    flight_metadata = []
    failed_files = []
    processed_count = 0

    for tail_number in tqdm(sorted(files_by_tail_batch.keys()), desc="Processing tails"):
        for batch_num in sorted(files_by_tail_batch[tail_number].keys()):
            batch_files = files_by_tail_batch[tail_number][batch_num]

            for flight_num, mat_file in enumerate(batch_files, start=1):
                try:
                    df, metadata = process_mat_file(mat_file, config_path=config_path)

                    if df is None:
                        failed_files.append(str(mat_file))
                        continue

                    output_path = (
                        output_dir
                        / f"tail_number={tail_number}"
                        / f"batch={batch_num}"
                        / f"flight_{flight_num:03d}.parquet"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Use zstd compression for better compression ratio
                    df.write_parquet(output_path, compression='zstd', compression_level=3)

                    # Prepare metadata for storage (convert signal_metadata to JSON string)
                    flat_metadata = {k: v for k, v in metadata.items() if k != 'signal_metadata'}
                    flat_metadata['batch'] = batch_num
                    flat_metadata['flight_number'] = flight_num
                    flat_metadata['source_folder'] = mat_file.parent.name
                    flat_metadata['signal_metadata_json'] = json.dumps(metadata.get('signal_metadata', {}))

                    flight_metadata.append(flat_metadata)
                    processed_count += 1

                    if processed_count % batch_size == 0:
                        logger.info(f"Checkpoint: {processed_count}/{len(mat_files)} files")
                        
                        pl.DataFrame(flight_metadata).write_parquet(output_dir / "checkpoint.parquet")

                except Exception as e:
                    logger.error(f"Failed to process {mat_file.name}: {e}")
                    failed_files.append(str(mat_file))

    # Build tail summary
    tail_summary = pl.DataFrame([
        {
            'tail_number': tail,
            'num_batches': len(batches),
            'num_flights': sum(len(files) for files in batches.values())
        }
        for tail, batches in sorted(files_by_tail_batch.items())
    ])

    # Save metadata
    if flight_metadata:
        metadata_df = pl.DataFrame(flight_metadata).join(tail_summary, on='tail_number', how='left')
        metadata_df.write_parquet(output_dir.parent / "flight_metadata.parquet")
        tail_summary.write_parquet(output_dir.parent / "tail_summary.parquet")
        logger.info(f"Saved flight_metadata.parquet and tail_summary.parquet")

    # Cleanup checkpoint
    checkpoint_path = output_dir / "checkpoint.parquet"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    logger.info(f"✓ Processed {len(flight_metadata)} files, ✗ Failed {len(failed_files)} files")

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

