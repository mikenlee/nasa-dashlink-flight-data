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


def load_signal_categories(config_path: str = 'config/config.yaml') -> Dict:
    """
    Load signal categories from config file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Dictionary of signal categories
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


def categorize_signal(signal_name: str, config_path: str = 'config/config.yaml') -> Tuple[str, int, 
str]:
    """
    Return category, priority, and description for a signal name.
    
    Args:
        signal_name: Name of the signal
        config_path: Path to config file
        
    Returns:
        Tuple of (category, priority, category_description)
    """
    categories = load_signal_categories(config_path)
    clean_name = signal_name.replace('.data', '').replace('.', '_').upper()

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
    year = stem[3:7]
    month = stem[7:9]
    day = stem[9:11]
    hour = stem[10:13] if len(stem) >= 11 else '00'
    minute = stem[12:14] if len(stem) >= 13 else '00'

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
      Process a single MAT file into Polars DataFrame.
      
      Args:
          mat_file: Path to MAT file
          config_path: Path to config file for signal categories
          
      Returns:
          Tuple of (DataFrame, metadata_dict)
      """
      logger.debug(f"Processing {mat_file.name}")

      # Load MAT file
      mat_data = scipy.io.loadmat(
          str(mat_file),
          squeeze_me=True,
          struct_as_record=False
      )

      # Extract flight info from filename
      flight_info = extract_flight_info(mat_file.name)
      tail_number = flight_info['tail_number']

      # Storage for processed data
      signal_dfs = []

      for signal_name, value in mat_data.items():
          # Skip metadata keys and sync words
          if signal_name.startswith('__') or signal_name in SYNC_WORDS:
              continue

          # Get category info
          category, priority, _ = categorize_signal(signal_name, config_path)

          try:
              # Handle structured data (MATLAB structs)
              if hasattr(value, '__dict__'):
                  struct_data = value.__dict__

                  if 'data' not in struct_data:
                      continue

                  signal_data = struct_data['data']

                  # Get sampling rate
                  rate = struct_data.get('Rate', struct_data.get('rate', 1.0))
                  if hasattr(rate, 'item'):
                      rate = rate.item()
                  rate = float(rate) if rate else 1.0

                  # Get units
                  units = struct_data.get('Units', struct_data.get('units', ''))
                  if hasattr(units, 'item'):
                      units = str(units)
                  units = str(units) if units else 'unknown'

                  # Get description
                  description = struct_data.get('Description', struct_data.get('description', ''))
                  if hasattr(description, 'item'):
                      description = str(description)
                  description = str(description) if description else ''

              # Handle raw arrays
              elif isinstance(value, np.ndarray):
                  signal_data = value
                  rate = 1.0
                  units = 'unknown'
                  description = ''
              else:
                  continue

              # Process only 1D arrays
              if not isinstance(signal_data, np.ndarray) or signal_data.ndim != 1:
                  continue

              # Convert to float64
              try:
                  float_values = signal_data.astype(np.float64)
              except (ValueError, TypeError):
                  continue

              n_samples = len(signal_data)
              indices = np.arange(n_samples, dtype=np.int64)
              time_seconds = indices / rate

              # Build DataFrame for this signal
              df_dict = {
                  'tail_number': tail_number,
                  'flight_id': flight_info['flight_id'],
                  'signal_name': signal_name,
                  'description': description,
                  'units': units,
                  'category': category,
                  'priority': priority,
                  'value': float_values,
                  'sample_index': indices,
                  'time_seconds': time_seconds,
                  'rate_hz': float(rate),
              }

              signal_dfs.append(pl.DataFrame(df_dict))

          except Exception as e:
              logger.warning(f"Could not extract {signal_name} from {mat_file.name}: {e}")

      if not signal_dfs:
          logger.warning(f"No valid signals extracted from {mat_file.name}")
          return None, flight_info

      # Combine all signals
      combined_df = pl.concat(signal_dfs, how="vertical_relaxed")

      # Add flight date
      combined_df = combined_df.with_columns([
          pl.lit(flight_info['date']).alias('date'),
      ])

      # Prepare metadata
      metadata = {
          **flight_info,
          'tail_number': tail_number,
          'num_samples': len(combined_df),
          'num_signals': combined_df.select('signal_name').n_unique(),
          'file_path': str(mat_file),
      }

      return combined_df, metadata


def validate_data(df: pl.DataFrame) -> Dict[str, any]:
    """
    Validate processed data and return quality metrics.
    
    Args:
        df: Processed flight data
        
    Returns:
        Dictionary of validation metrics
    """
    metrics = {
        'num_rows': len(df),
        'num_cols': len(df.columns),
        'num_signals': df.select('signal_name').n_unique() if 'signal_name' in df.columns else 0,
        'missing_pct': (df.null_count().sum_horizontal()[0] / (len(df) * len(df.columns)) * 100),
        'memory_mb': df.estimated_size('mb')
    }

    # Check for anomalies in EGT if present
    if 'signal_name' in df.columns and 'value' in df.columns:
        egt_data = df.filter(pl.col('signal_name') == 'EGT_1')
        if egt_data.height > 0:
            metrics['egt_min'] = egt_data.select('value').min().item()
            metrics['egt_max'] = egt_data.select('value').max().item()
            metrics['egt_outliers'] = egt_data.filter(
                (pl.col('value') < 0) | (pl.col('value') > 1000)
            ).height

    return metrics