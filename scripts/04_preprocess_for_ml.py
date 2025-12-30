"""
Preprocess flight data for ML anomaly detection.

Resamples all signals to a common rate (4 Hz) using proper time-based alignment,
selects relevant features, and saves to parquet.

Usage:
    # Process first 2 tails (for testing)
    python scripts/04_preprocess_for_ml.py --max-tails 2

    # Process all tails
    python scripts/04_preprocess_for_ml.py

    # Process specific tails
    python scripts/04_preprocess_for_ml.py --tails 652 653 654
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import polars as pl
import scipy.io
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_path

# Setup logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

TARGET_HZ = 4.0  # Target sampling rate

# Signals to extract for anomaly detection
SELECTED_SIGNALS = [
    # Engine (8 signals)
    'N1_1', 'N1_2', 'N2_1', 'N2_2',
    'EGT_1', 'EGT_2', 'FF_1', 'FF_2',
    # Flight dynamics (6 signals)
    'PTCH', 'ROLL',
    'VRTG', 'FPAC', 'BLAC', 'CTAC',
    # Altitude/vertical (3 signals)
    'ALT', 'RALT', 'IVV',
    # Speed (4 signals)
    'CAS', 'TAS', 'MACH', 'GS',
    # Controls (3 signals)
    'AIL_1', 'RUDD', 'FLAP',
]

# Sync words to filter out
SYNC_WORDS = {'VAR_1107', 'VAR_2670', 'VAR_5107', 'VAR_6670'}


# =============================================================================
# Core Functions
# =============================================================================

def downsample_average(data: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by averaging consecutive samples."""
    n = len(data) // factor * factor
    if n == 0:
        return data[:1] if len(data) > 0 else np.array([np.nan])
    reshaped = data[:n].reshape(-1, factor)
    return reshaped.mean(axis=1).astype(np.float32)


def upsample_interpolate(data: np.ndarray, factor: int) -> np.ndarray:
    """Upsample by linear interpolation."""
    original_indices = np.arange(len(data))
    target_indices = np.linspace(0, len(data) - 1, len(data) * factor)
    return np.interp(target_indices, original_indices, data).astype(np.float32)


def resample_signal(data: np.ndarray, original_hz: float, target_hz: float) -> np.ndarray:
    """
    Resample signal to target rate.

    - If original_hz > target_hz: downsample by averaging
    - If original_hz < target_hz: upsample by interpolation
    - If equal: return as-is
    """
    if original_hz == target_hz:
        return data.astype(np.float32)

    ratio = original_hz / target_hz

    if ratio > 1:
        # Downsample
        factor = int(round(ratio))
        return downsample_average(data, factor)
    else:
        # Upsample
        factor = int(round(1 / ratio))
        return upsample_interpolate(data, factor)


def process_mat_file_for_ml(
    mat_file: Path,
    target_hz: float = TARGET_HZ,
    selected_signals: List[str] = SELECTED_SIGNALS
) -> Tuple[Optional[pl.DataFrame], Dict]:
    """
    Process a single MAT file with proper time-based resampling.

    Args:
        mat_file: Path to MAT file
        target_hz: Target sampling rate
        selected_signals: List of signals to extract

    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    try:
        mat_data = scipy.io.loadmat(
            str(mat_file),
            squeeze_me=True,
            struct_as_record=False
        )
    except Exception as e:
        logger.error(f"Failed to load {mat_file}: {e}")
        return None, {'error': str(e)}

    # First pass: find total duration from highest-rate signal
    max_duration_sec = 0
    signal_info = {}

    for signal_name, value in mat_data.items():
        if signal_name.startswith('__') or signal_name in SYNC_WORDS:
            continue

        if not hasattr(value, '__dict__') or not hasattr(value, 'data'):
            continue

        try:
            data = value.data
            if not isinstance(data, np.ndarray) or data.ndim != 1:
                continue

            rate = getattr(value, 'Rate', 1.0)
            if hasattr(rate, 'item'):
                rate = rate.item()
            rate = float(rate) if rate else 1.0

            duration_sec = len(data) / rate
            max_duration_sec = max(max_duration_sec, duration_sec)

            signal_info[signal_name] = {
                'rate': rate,
                'samples': len(data),
                'duration_sec': duration_sec
            }
        except Exception:
            continue

    if max_duration_sec == 0:
        logger.warning(f"No valid signals in {mat_file}")
        return None, {'error': 'No valid signals'}

    # Calculate target number of samples
    target_samples = int(max_duration_sec * target_hz)

    # Second pass: resample selected signals
    resampled_data = {}
    signals_found = []
    signals_missing = []

    for signal_name in selected_signals:
        if signal_name not in mat_data:
            signals_missing.append(signal_name)
            continue

        value = mat_data[signal_name]
        if not hasattr(value, 'data'):
            signals_missing.append(signal_name)
            continue

        try:
            data = value.data
            rate = signal_info.get(signal_name, {}).get('rate', 1.0)

            # Resample to target rate
            resampled = resample_signal(data, rate, target_hz)

            # Ensure consistent length (trim or pad if needed due to rounding)
            if len(resampled) > target_samples:
                resampled = resampled[:target_samples]
            elif len(resampled) < target_samples:
                pad_length = target_samples - len(resampled)
                resampled = np.pad(resampled, (0, pad_length), constant_values=np.nan)

            resampled_data[signal_name] = resampled
            signals_found.append(signal_name)

        except Exception as e:
            logger.warning(f"Failed to resample {signal_name} in {mat_file}: {e}")
            signals_missing.append(signal_name)

    if not resampled_data:
        logger.warning(f"No selected signals found in {mat_file}")
        return None, {'error': 'No selected signals found'}

    # Create DataFrame
    df = pl.DataFrame(resampled_data)

    # Add time column
    time_sec = np.arange(target_samples) / target_hz
    df = df.with_columns([
        pl.Series('time_sec', time_sec.astype(np.float32)),
    ])

    # Reorder columns: time first, then signals alphabetically
    signal_cols = sorted([c for c in df.columns if c != 'time_sec'])
    df = df.select(['time_sec'] + signal_cols)

    # Metadata
    metadata = {
        'source_file': mat_file.name,
        'duration_sec': max_duration_sec,
        'target_hz': target_hz,
        'target_samples': target_samples,
        'signals_found': signals_found,
        'signals_missing': signals_missing,
        'num_signals': len(signals_found),
    }

    return df, metadata


def get_mat_files_by_tail(raw_dir: Path) -> Dict[str, List[Path]]:
    """Group MAT files by tail number."""
    from collections import defaultdict
    import re

    files_by_tail = defaultdict(list)

    for mat_file in raw_dir.glob('**/*.mat'):
        # Try to extract tail from directory name (Tail_XXX_Y format)
        for parent in mat_file.parents:
            match = re.search(r'Tail_\s*(\d+)\s*_\s*(\d+)', parent.name, flags=re.IGNORECASE)
            if match:
                tail = match.group(1)
                files_by_tail[tail].append(mat_file)
                break
        else:
            # Fallback: use first 3 characters of filename
            tail = mat_file.stem[:3]
            files_by_tail[tail].append(mat_file)

    # Sort files within each tail
    for tail in files_by_tail:
        files_by_tail[tail].sort(key=lambda f: f.name)

    return dict(files_by_tail)


# =============================================================================
# Main Processing
# =============================================================================

def process_tails(
    raw_dir: Path,
    output_dir: Path,
    tails_to_process: Optional[List[str]] = None,
    max_tails: Optional[int] = None,
) -> None:
    """
    Process MAT files for specified tails.

    Args:
        raw_dir: Directory containing raw MAT files
        output_dir: Output directory for processed parquet files
        tails_to_process: List of specific tail numbers to process (None = all)
        max_tails: Maximum number of tails to process (None = all)
    """
    logger.info(f"Raw directory: {raw_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target rate: {TARGET_HZ} Hz")
    logger.info(f"Selected signals: {len(SELECTED_SIGNALS)}")

    # Get all MAT files grouped by tail
    files_by_tail = get_mat_files_by_tail(raw_dir)
    available_tails = sorted(files_by_tail.keys())

    logger.info(f"Found {len(available_tails)} tails: {available_tails}")

    # Filter tails
    if tails_to_process:
        tails = [t for t in tails_to_process if t in available_tails]
        missing = set(tails_to_process) - set(available_tails)
        if missing:
            logger.warning(f"Tails not found: {missing}")
    else:
        tails = available_tails

    if max_tails:
        tails = tails[:max_tails]

    logger.info(f"Processing {len(tails)} tails: {tails}")

    # Process each tail
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = []
    total_processed = 0
    total_failed = 0

    for tail in tqdm(tails, desc="Processing tails"):
        mat_files = files_by_tail[tail]
        logger.info(f"Tail {tail}: {len(mat_files)} flights")

        tail_output_dir = output_dir / f"tail_number={tail}"
        tail_output_dir.mkdir(parents=True, exist_ok=True)

        for flight_num, mat_file in enumerate(tqdm(mat_files, desc=f"Tail {tail}", leave=False), start=1):
            df, metadata = process_mat_file_for_ml(mat_file)

            if df is None:
                total_failed += 1
                continue

            # Save parquet
            output_path = tail_output_dir / f"flight_{flight_num:03d}.parquet"
            df.write_parquet(output_path, compression='zstd', compression_level=3)

            # Add metadata
            metadata['tail'] = tail
            metadata['flight_num'] = flight_num
            metadata['output_path'] = str(output_path)
            all_metadata.append(metadata)

            total_processed += 1

    # Save metadata
    if all_metadata:
        metadata_df = pl.DataFrame(all_metadata)
        metadata_path = output_dir.parent / "ml_preprocessing_metadata.parquet"
        metadata_df.write_parquet(metadata_path)
        logger.info(f"Saved metadata to {metadata_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed: {total_processed} flights")
    logger.info(f"Failed: {total_failed} flights")
    logger.info(f"Output: {output_dir}")

    # Check signal coverage
    if all_metadata:
        signals_found_counts = {}
        for m in all_metadata:
            for s in m.get('signals_found', []):
                signals_found_counts[s] = signals_found_counts.get(s, 0) + 1

        logger.info(f"\nSignal coverage (out of {total_processed} flights):")
        for signal in SELECTED_SIGNALS:
            count = signals_found_counts.get(signal, 0)
            pct = count / total_processed * 100 if total_processed > 0 else 0
            status = "OK" if pct > 95 else "LOW" if pct > 50 else "MISSING"
            logger.info(f"  {signal}: {count} ({pct:.1f}%) [{status}]")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess flight data for ML anomaly detection'
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default=None,
        help='Raw data directory (default: from config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/processed/ml_ready)'
    )
    parser.add_argument(
        '--tails',
        nargs='+',
        type=str,
        default=None,
        help='Specific tail numbers to process'
    )
    parser.add_argument(
        '--max-tails',
        type=int,
        default=None,
        help='Maximum number of tails to process (for testing)'
    )

    args = parser.parse_args()

    # Set directories
    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
    else:
        raw_dir = get_path('raw')

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_path('processed') / 'ml_ready'

    if not raw_dir.exists():
        logger.error(f"Raw directory does not exist: {raw_dir}")
        sys.exit(1)

    process_tails(
        raw_dir=raw_dir,
        output_dir=output_dir,
        tails_to_process=args.tails,
        max_tails=args.max_tails,
    )


if __name__ == "__main__":
    main()
