"""
Reusable functions for loading and processing flight data.
"""

import polars as pl
import scipy.io
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import logging
import yaml

logger = logging.getLogger(__name__)

# Sync words to filter out (frame synchronization markers, not actual telemetry)
SYNC_WORDS = {'VAR_1107', 'VAR_2670', 'VAR_5107', 'VAR_6670'}

# Module-level cache for signal categories
_SIGNAL_CATEGORIES = None


def load_signal_categories(config_path: str = 
'config/config.yaml') -> Dict:
    """
    Load signal categories from config file.
    """
    global _SIGNAL_CATEGORIES

    if _SIGNAL_CATEGORIES is not None:
        return _SIGNAL_CATEGORIES

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        _SIGNAL_CATEGORIES = config.get('signals', {})
        logger.debug(f"Loaded {len(_SIGNAL_CATEGORIES)} signal categories from {config_path}")
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}. Using empty categories.")
        _SIGNAL_CATEGORIES = {}

    return _SIGNAL_CATEGORIES


def categorize_signal(signal_name: str, config_path: str = 'config/config.yaml') -> Tuple[str, int, str]:
    """
    Return category, priority, and description for a signal name.
    """
    categories = load_signal_categories(config_path)
    clean_name = signal_name.replace('.data', '').replace('.','_').upper()

    for category, info in categories.items():
        signals = info.get('signals', [])
        priority = info.get('priority', 99)
        description = info.get('description', '')

        for sig in signals:
            if clean_name == sig.upper() or clean_name.startswith(sig.upper()):
                return category, priority, description

    return 'uncategorized', 99, 'Not yet classified'


def extract_flight_info(filename: str) -> Dict[str, str]:
    """
    Extract flight metadata from filename.
    
    Example: 687200403251602.mat
    -> tail: 687, date: 2004-03-25, time: 16:02
    """
    stem = Path(filename).stem

    year = '20' + stem[3:5]
    month = stem[5:7]
    day = stem[7:9]
    hour = stem[9:11] if len(stem) >= 11 else '00'
    minute = stem[11:13] if len(stem) >= 13 else '00'

    return {
        'tail_number': stem[:3],
        'year': year,
        'month': month,
        'day': day,
        'hour': hour,
        'minute': minute,
        'date': f"{year}-{month}-{day}",
        'time': f"{hour}:{minute}",
        'flight_id': stem
    }


def process_mat_file(
    mat_file: Path,
    config_path: str = 'config/config.yaml'
) -> Tuple[pl.DataFrame, Dict]:
    """
    Process a single MAT file into wide-format Polars DataFrame.
    
    Each signal becomes a column, rows are time samples.
    This is much more space-efficient than long format.
    
    Args:
        mat_file: Path to MAT file
        config_path: Path to config file for signal categories
        
    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    logger.debug(f"Processing {mat_file.name}")

    mat_data = scipy.io.loadmat(
        str(mat_file),
        squeeze_me=True,
        struct_as_record=False
    )

    flight_info = extract_flight_info(mat_file.name)
    tail_number = flight_info['tail_number']

    # Collect signals as columns (wide format)
    signal_data = {}
    signal_metadata = {}  # Store per-signal metadata separately
    max_length = 0

    for signal_name, value in mat_data.items():
        if signal_name.startswith('__') or signal_name in SYNC_WORDS:
            continue

        try:
            if hasattr(value, '__dict__'):
                struct_data = value.__dict__
                if 'data' not in struct_data:
                    continue

                data = struct_data['data']

                # Extract signal metadata
                rate = struct_data.get('Rate', struct_data.get('rate', 1.0))
                if hasattr(rate, 'item'):
                    rate = rate.item()
                rate = float(rate) if rate else 1.0

                units = struct_data.get('Units', struct_data.get('units', ''))
                if hasattr(units, 'item'):
                    units = str(units)
                units = str(units) if units else ''

                description = struct_data.get('Description', struct_data.get('description', ''))
                if hasattr(description, 'item'):
                    description = str(description)
                description = str(description) if description else ''

                category, priority, _ = categorize_signal(signal_name, config_path)

                signal_metadata[signal_name] = {
                    'rate_hz': rate,
                    'units': units,
                    'description': description,
                    'category': category,
                    'priority': priority,
                }

            elif isinstance(value, np.ndarray):
                data = value
                signal_metadata[signal_name] = {
                    'rate_hz': 1.0,
                    'units': '',
                    'description': '',
                    'category': 'uncategorized',
                    'priority': 99,
                }
            else:
                continue

            if not isinstance(data, np.ndarray) or data.ndim !=1:
                continue

            try:
                signal_data[signal_name] = data.astype(np.float32)  # float32 saves space
                max_length = max(max_length, len(data))
            except (ValueError, TypeError):
                continue

        except Exception as e:
            logger.warning(f"Could not extract {signal_name} from {mat_file.name}: {e}")

    if not signal_data:
        logger.warning(f"No valid signals extracted from {mat_file.name}")
        return None, flight_info

    # Pad shorter signals to max_length (handles different sample rates)
    for name, data in signal_data.items():
        if len(data) < max_length:
            signal_data[name] = np.pad(data, (0, max_length - len(data)), constant_values=np.nan)

    # Create wide DataFrame
    df = pl.DataFrame(signal_data)

    # Add sample index
    df = df.with_columns([
        pl.int_range(pl.len()).cast(pl.Int32).alias('sample_index'),
    ])

    # Reorder columns: sample_index first, then signals alphabetically
    signal_cols = sorted([c for c in df.columns if c !='sample_index'])
    df = df.select(['sample_index'] + signal_cols)

    # Prepare metadata
    metadata = {
        **flight_info,
        'num_samples': len(df),
        'num_signals': len(signal_data),
        'file_path': str(mat_file),
        'signal_metadata': signal_metadata,  # Nested dict with per-signal info
    }

    return df, metadata


def get_signal_info(metadata: Dict) -> pl.DataFrame:
    """
    Extract signal metadata from flight metadata as a DataFrame.
    
    Useful for looking up signal descriptions, units, rates, etc.
    
    Args:
        metadata: Metadata dict from process_mat_file
        
    Returns:
        DataFrame with signal_name, rate_hz, units, description, category, priority
    """
    signal_metadata = metadata.get('signal_metadata', {})

    records = [
        {
            'signal_name': name,
            **info
        }
        for name, info in signal_metadata.items()
    ]

    return pl.DataFrame(records).sort('priority', 'signal_name')


def validate_data(df: pl.DataFrame) -> Dict[str, any]:
    """
    Validate processed data and return quality metrics.
    """
    # Count signal columns (exclude sample_index)
    signal_cols = [c for c in df.columns if c != 'sample_index']

    # Calculate null percentage across all signal columns
    total_cells = len(df) * len(signal_cols)
    null_count = sum(df.select(c).null_count().item() for c in signal_cols)

    metrics = {
        'num_rows': len(df),
        'num_signals': len(signal_cols),
        'null_pct': (null_count / total_cells * 100) if total_cells > 0 else 0,
        'memory_mb': df.estimated_size('mb')
    }

    # Check for anomalies in EGT if present
    if 'EGT_1' in df.columns:
        egt = df.select('EGT_1').to_series()
        metrics['egt_min'] = egt.min()
        metrics['egt_max'] = egt.max()
        metrics['egt_outliers'] = egt.filter((egt < 0) | (egt > 1000)).len()

    return metrics