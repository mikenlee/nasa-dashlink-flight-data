"""
Project configuration and path resolution.
"""

from pathlib import Path

# Resolve project root (works from any subdirectory)
def get_project_root() -> Path:
    """Find project root by looking for known markers."""
    current = Path(__file__).resolve().parent  # src/

    # Go up until we find project markers
    for parent in [current] + list(current.parents):
        if (parent / 'config').exists() or (parent / 'data').exists():
            return parent

    # Fallback to current working directory
    return Path.cwd()


PROJECT_ROOT = get_project_root()

# Standard paths
PATHS = {
    'data': PROJECT_ROOT / 'data',
    'raw': PROJECT_ROOT / 'data' / 'raw',
    'processed': PROJECT_ROOT / 'data' / 'processed',
    'telemetry': PROJECT_ROOT / 'data' / 'processed' / 'telemetry',
    'metadata': PROJECT_ROOT / 'data' / 'processed' / 'flight_metadata.parquet',
    'config': PROJECT_ROOT / 'config' / 'config.yaml',
}


def get_path(name: str) -> Path:
    """Get a standard project path by name."""
    if name not in PATHS:
        raise ValueError(f"Unknown path: {name}. Available: {list(PATHS.keys())}")
    return PATHS[name]