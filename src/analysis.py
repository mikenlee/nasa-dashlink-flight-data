import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from src.config import get_path, PATHS

def plot_altitude_profile(df: pl.DataFrame, title: str = None):
    alt = df.select(['sample_index', 'ALT']).to_pandas()
    plt.figure(figsize=(12, 4))
    plt.plot(alt['sample_index'] / 4 / 60, alt['ALT'])  # 4 Hz, convert to minutes
    plt.xlabel('Time (minutes)')
    plt.ylabel('Altitude (ft)')
    if title:
        plt.title(title)
    return plt.gcf()

"""
Analysis functions for flight telemetry data.
"""


# =============================================================================
# Data Loading
# =============================================================================

def load_flight(tail: str, batch: int, flight: int) -> pl.DataFrame:
    """
    Load a single flight's parquet file.
    
    Args:
        tail: Tail number (e.g., '652')
        batch: Batch number (e.g., 1)
        flight: Flight number (e.g., 1)
        
    Returns:
        Polars DataFrame with all signals
    """
    path = get_path('telemetry') / f"tail_number={tail}" / f"batch={batch}" / f"flight_{flight:03d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Flight not found: {path}")
    return pl.read_parquet(path)


def load_flights_for_tail(tail: str) -> List[pl.DataFrame]:
    """Load all flights for a given tail number."""
    tail_path = get_path('telemetry') / f"tail_number={tail}"
    if not tail_path.exists():
        raise FileNotFoundError(f"Tail not found: {tail_path}")

    flights = []
    for pq_file in sorted(tail_path.glob('**/*.parquet')):
        flights.append(pl.read_parquet(pq_file))
    return flights


def load_metadata() -> pl.DataFrame:
    """Load flight metadata."""
    return pl.read_parquet(get_path('metadata'))


def get_available_tails() -> List[str]:
    """List all available tail numbers."""
    tails = []
    for p in get_path('telemetry').glob('tail_number=*'):
        tail = p.name.replace('tail_number=', '')
        tails.append(tail)
    return sorted(tails)


# =============================================================================
# Signal Extraction
# =============================================================================

def get_signal(df: pl.DataFrame, signal: str, rate_hz: float = 4.0) -> pl.DataFrame:
    """
    Extract a signal with time in seconds and minutes.
    
    Args:
        df: Flight DataFrame
        signal: Signal name (e.g., 'ALT')
        rate_hz: Sampling rate in Hz (default 4.0 for most signals)
        
    Returns:
        DataFrame with sample_index, time_seconds, time_minutes, and signal value
    """
    if signal not in df.columns:
        raise ValueError(f"Signal '{signal}' not found. Available: {df.columns[:10]}...")

    return df.select([
        'sample_index',
        (pl.col('sample_index') / rate_hz).alias('time_seconds'),
        (pl.col('sample_index') / rate_hz / 60).alias('time_minutes'),
        pl.col(signal)
    ])


def get_signals(df: pl.DataFrame, signals: List[str], rate_hz: float = 4.0) -> pl.DataFrame:
    """Extract multiple signals with time columns."""
    missing = [s for s in signals if s not in df.columns]
    if missing:
        raise ValueError(f"Signals not found: {missing}")

    return df.select([
        'sample_index',
        (pl.col('sample_index') / rate_hz).alias('time_seconds'),
        (pl.col('sample_index') / rate_hz / 60).alias('time_minutes'),
        *signals
    ])


# =============================================================================
# Altitude Analysis
# =============================================================================

def get_flight_phases(df: pl.DataFrame, alt_col: str = 'ALT', 
                    ground_threshold: float = 500,
                    climb_threshold: float = 100,
                    rate_hz: float = 4.0) -> pl.DataFrame:
    """
    Detect flight phases based on altitude.
    
    Phases:
        - ground: altitude < ground_threshold
        - climb: altitude increasing > climb_threshold ft/min
        - cruise: altitude stable
        - descent: altitude decreasing > climb_threshold ft/min
    
    Returns:
        DataFrame with phase column added
    """
    # Calculate altitude rate of change (ft/min)
    window_size = int(rate_hz * 30)  # 30-second window

    result = df.with_columns([
        (pl.col('sample_index') / rate_hz / 60).alias('time_minutes'),
        pl.col(alt_col).rolling_mean(window_size=window_size).alias('alt_smooth'),
    ]).with_columns([
        (pl.col('alt_smooth').diff() * rate_hz * 60).alias('alt_rate'),  # ft/min
    ]).with_columns([
        pl.when(pl.col('alt_smooth') < ground_threshold)
        .then(pl.lit('ground'))
        .when(pl.col('alt_rate') > climb_threshold)
        .then(pl.lit('climb'))
        .when(pl.col('alt_rate') < -climb_threshold)
        .then(pl.lit('descent'))
        .otherwise(pl.lit('cruise'))
        .alias('phase')
    ])

    return result


def get_flight_stats(df: pl.DataFrame, alt_col: str = 'ALT', 
                    rate_hz: float = 4.0) -> dict:
    """
    Calculate summary statistics for a flight.
    
    Returns:
        Dictionary with flight statistics
    """
    alt = df.select(alt_col).to_series()

    duration_seconds = len(df) / rate_hz
    duration_minutes = duration_seconds / 60

    return {
        'duration_minutes': round(duration_minutes, 1),
        'max_altitude_ft': round(alt.max(), 0),
        'min_altitude_ft': round(alt.min(), 0),
        'mean_altitude_ft': round(alt.mean(), 0),
        'alt_std_ft': round(alt.std(), 0),
        'num_samples': len(df),
    }


def compare_flights_stats(flights: List[Tuple[str, pl.DataFrame]]) -> pl.DataFrame:
    """
    Compare statistics across multiple flights.
    
    Args:
        flights: List of (label, DataFrame) tuples
        
    Returns:
        DataFrame with stats for each flight
    """
    records = []
    for label, df in flights:
        stats = get_flight_stats(df)
        stats['flight'] = label
        records.append(stats)

    return pl.DataFrame(records).select(['flight'] + [c for c in records[0].keys() if c != 'flight'])


# =============================================================================
# Plotting
# =============================================================================

def plot_altitude_profile(df: pl.DataFrame, alt_col: str = 'ALT',
                        rate_hz: float = 4.0, title: str = None,
                        figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """
    Plot altitude vs time for a single flight.
    """
    signal_df = get_signal(df, alt_col, rate_hz)
    data = signal_df.to_pandas()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data['time_minutes'], data[alt_col], linewidth=0.5)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Altitude (ft)')
    ax.set_title(title or 'Altitude Profile')
    ax.grid(True, alpha=0.3)

    # Add stats annotation
    stats = get_flight_stats(df, alt_col, rate_hz)
    stats_text = f"Duration: {stats['duration_minutes']} min\nMax Alt: {stats['max_altitude_ft']:,.0f} ft"
    ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_altitude_comparison(flights: List[Tuple[str, pl.DataFrame]], 
                            alt_col: str = 'ALT',
                            rate_hz: float = 4.0,
                            figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Compare altitude profiles across multiple flights.
    
    Args:
        flights: List of (label, DataFrame) tuples
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, df in flights:
        signal_df = get_signal(df, alt_col, rate_hz)
        data = signal_df.to_pandas()
        ax.plot(data['time_minutes'], data[alt_col], linewidth=0.5, label=label, alpha=0.7)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Altitude (ft)')
    ax.set_title('Altitude Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_altitude_with_phases(df: pl.DataFrame, alt_col: str = 'ALT',
                                rate_hz: float = 4.0, title: str = None,
                                figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """
    Plot altitude profile with flight phases color-coded.
    """
    phased_df = get_flight_phases(df, alt_col, rate_hz=rate_hz)
    data = phased_df.to_pandas()

    fig, ax = plt.subplots(figsize=figsize)

    # Color map for phases
    colors = {'ground': 'brown', 'climb': 'green', 'cruise': 'blue', 'descent': 'red'}

    # Plot each phase with different color
    for phase, color in colors.items():
        mask = data['phase'] == phase
        ax.scatter(data.loc[mask, 'time_minutes'], data.loc[mask, alt_col],
                    c=color, s=1, label=phase, alpha=0.5)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Altitude (ft)')
    ax.set_title(title or 'Altitude Profile with Flight Phases')
    ax.legend(loc='upper right', markerscale=5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_altitude_distribution(flights: List[Tuple[str, pl.DataFrame]],
                                alt_col: str = 'ALT',
                                figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot altitude distribution (histogram) across flights.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1 = axes[0]
    for label, df in flights:
        alt = df.select(alt_col).to_series().to_numpy()
        ax1.hist(alt, bins=50, alpha=0.5, label=label)
    ax1.set_xlabel('Altitude (ft)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Altitude Distribution')
    ax1.legend()

    # Box plot
    ax2 = axes[1]
    data = [df.select(alt_col).to_series().to_numpy() for _, df in flights]
    labels = [label for label, _ in flights]
    ax2.boxplot(data, labels=labels)
    ax2.set_ylabel('Altitude (ft)')
    ax2.set_title('Altitude Box Plot')

    plt.tight_layout()
    return fig


# =============================================================================
# Quick Analysis
# =============================================================================

def quick_flight_summary(tail: str, batch: int, flight: int) -> None:
    """Print a quick summary of a flight."""
    df = load_flight(tail, batch, flight)
    stats = get_flight_stats(df)

    print(f"Flight: Tail {tail}, Batch {batch}, Flight {flight}")
    print(f"  Duration: {stats['duration_minutes']} minutes")
    print(f"  Max Altitude: {stats['max_altitude_ft']:,.0f} ft")
    print(f"  Samples: {stats['num_samples']:,}")
    print(f"  Signals: {len(df.columns) - 1}")
    print(f"  Columns: {df.columns[:10]}...")


def analyze_tail(tail: str, max_flights: int = 5) -> None:
    """Quick analysis of flights for a tail number."""
    print(f"Analyzing Tail {tail}")
    print("=" * 50)

    flights = []
    base_path = Path('data/processed/telemetry') / f"tail_number={tail}"

    pq_files = sorted(base_path.glob('**/*.parquet'))[:max_flights]

    for pq_file in pq_files:
        df = pl.read_parquet(pq_file)
        label = pq_file.stem
        flights.append((label, df))

        stats = get_flight_stats(df)
        print(f"\n{label}:")
        print(f"  Duration: {stats['duration_minutes']} min, Max Alt: {stats['max_altitude_ft']:,.0f} ft")

    # Plot comparison
    if flights:
        fig = plot_altitude_comparison(flights)
        plt.show()

    return flights



