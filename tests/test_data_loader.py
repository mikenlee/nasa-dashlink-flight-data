#!/usr/bin/env python3
"""
Pytest tests for data_loader.py functions.

Usage:
    pytest tests/test_data_loader.py -v
    pytest tests/test_data_loader.py -v -k "test_categorize"  # run specific test
    
    # Run with output shown
    pytest tests/test_data_loader.py -v -s

    # Run specific test class
    pytest tests/test_data_loader.py::TestCategorizeSignal -v

    # Run specific test
    pytest tests/test_data_loader.py::TestCategorizeSignal::test_engine_health_signals -v

    # Run tests matching pattern
    pytest tests/test_data_loader.py -v -k "engine"
"""

import sys
from pathlib import Path
import pytest
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_signal_categories,
    categorize_signal,
    extract_flight_info,
    process_mat_file,
    build_signal_dictionary,
    validate_data,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def mat_files():
    """Find all MAT files once for the entire test session."""
    possible_paths = [
        Path('data/raw'),
        Path('data'),
    ]

    for path in possible_paths:
        if path.exists():
            files = list(path.glob('**/*.mat'))
            if files:
                return files

    return []


@pytest.fixture(scope="session")
def sample_mat_file(mat_files):
    """Get a single MAT file for testing."""
    if not mat_files:
        pytest.skip("No MAT files found")
    return Path(mat_files[0])


@pytest.fixture(scope="session")
def processed_df(sample_mat_file):
    """Process a MAT file once for reuse in multiple tests."""
    df, metadata, _ = process_mat_file(sample_mat_file, extract_signal_metadata=False)
    if df is None:
        pytest.skip("Failed to process MAT file")
    return df


@pytest.fixture(scope="session")
def signal_categories():
    """Load signal categories once."""
    return load_signal_categories()


# =============================================================================
# Tests: load_signal_categories
# =============================================================================

class TestLoadSignalCategories:

    def test_loads_categories(self, signal_categories):
        """Should load categories from config."""
        assert len(signal_categories) > 0

    def test_has_expected_categories(self, signal_categories):
        """Should have core categories."""
        expected = ['safety_critical', 'engine_health', 'flight_performance']
        for cat in expected:
            assert cat in signal_categories, f"Missing category: {cat}"

    def test_category_structure(self, signal_categories):
        """Each category should have priority, description, and signals."""
        for cat_name, cat_info in signal_categories.items():
            assert 'priority' in cat_info, f"{cat_name} missing priority"
            assert 'signals' in cat_info, f"{cat_name} missing signals"
            assert isinstance(cat_info['signals'], list), f"{cat_name} signals not a list"


# =============================================================================
# Tests: categorize_signal
# =============================================================================

class TestCategorizeSignal:

    @pytest.mark.parametrize("signal_name,expected_category,expected_priority", [
        ('EGT_1', 'engine_health', 2),
        ('EGT_2', 'engine_health', 2),
        ('N1_1', 'engine_health', 2),
        ('N2_3', 'engine_health', 2),
        ('FF_4', 'engine_health', 2),
    ])
    def test_engine_health_signals(self, signal_name, expected_category, expected_priority):
        """Engine signals should be categorized correctly."""
        category, priority, _ = categorize_signal(signal_name)
        assert category == expected_category
        assert priority == expected_priority

    @pytest.mark.parametrize("signal_name,expected_category,expected_priority", [
        ('ALT', 'safety_critical', 1),
        ('RALT', 'safety_critical', 1),
        ('AOA1', 'safety_critical', 1),
        ('GPWS', 'safety_critical', 1),
        ('FIRE_1', 'safety_critical', 1),
    ])
    def test_safety_critical_signals(self, signal_name, expected_category, expected_priority):
        """Safety critical signals should be categorized correctly."""
        category, priority, _ = categorize_signal(signal_name)
        assert category == expected_category
        assert priority == expected_priority

    @pytest.mark.parametrize("signal_name,expected_category,expected_priority", [
        ('PTCH', 'flight_performance', 3),
        ('ROLL', 'flight_performance', 3),
        ('GS', 'flight_performance', 3),
    ])
    def test_flight_performance_signals(self, signal_name, expected_category, expected_priority):
        """Flight performance signals should be categorized correctly."""
        category, priority, _ = categorize_signal(signal_name)
        assert category == expected_category
        assert priority == expected_priority

    def test_unknown_signal(self):
        """Unknown signals should be uncategorized."""
        category, priority, _ = categorize_signal('UNKNOWN_XYZ_123')
        assert category == 'uncategorized'
        assert priority == 99

    def test_returns_description(self):
        """Should return category description."""
        _, _, description = categorize_signal('EGT_1')
        assert isinstance(description, str)
        assert len(description) > 0


# =============================================================================
# Tests: extract_flight_info
# =============================================================================

class TestExtractFlightInfo:

    @pytest.mark.parametrize("filename,expected_tail,expected_date", [
        ('687200403251602.mat', '687', '2004-03-25'),
        ('677200202040430.mat', '677', '2002-02-04'),
        ('654200101011200.mat', '654', '2001-01-01'),
    ])
    def test_extracts_tail_and_date(self, filename, expected_tail, expected_date):
        """Should extract tail number and date from filename."""
        info = extract_flight_info(filename)
        assert info['tail_number'] == expected_tail
        assert info['date'] == expected_date

    def test_extracts_flight_id(self):
        """Flight ID should be the filename stem."""
        info = extract_flight_info('687200403251602.mat')
        assert info['flight_id'] == '687200403251602'

    def test_extracts_all_fields(self):
        """Should extract all expected fields."""
        info = extract_flight_info('687200403251602.mat')
        expected_keys = ['tail_number', 'year', 'month', 'day', 'hour', 'minute', 'date',
'flight_id']
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"


# =============================================================================
# Tests: process_mat_file
# =============================================================================

class TestProcessMatFile:

    def test_returns_dataframe(self, sample_mat_file):
        """Should return a Polars DataFrame."""
        df, _, _ = process_mat_file(sample_mat_file)
        assert df is not None
        assert isinstance(df, pl.DataFrame)

    def test_dataframe_has_required_columns(self, processed_df):
        """DataFrame should have all required columns."""
        required = ['tail_number', 'flight_id', 'signal_name', 'category',
                    'priority', 'value', 'sample_index', 'time_seconds', 'rate_hz']
        for col in required:
            assert col in processed_df.columns, f"Missing column: {col}"

    def test_returns_metadata(self, sample_mat_file):
        """Should return metadata dict."""
        _, metadata, _ = process_mat_file(sample_mat_file)
        assert isinstance(metadata, dict)
        assert 'tail_number' in metadata
        assert 'flight_id' in metadata
        assert 'num_signals' in metadata

    def test_extracts_signal_metadata_when_requested(self, sample_mat_file):
        """Should return signal dict when extract_signal_metadata=True."""
        _, _, signal_dict = process_mat_file(sample_mat_file, extract_signal_metadata=True)
        assert signal_dict is not None
        assert isinstance(signal_dict, pl.DataFrame)
        assert 'description' in signal_dict.columns
        assert 'units' in signal_dict.columns

    def test_no_signal_metadata_by_default(self, sample_mat_file):
        """Should not return signal dict by default."""
        _, _, signal_dict = process_mat_file(sample_mat_file, extract_signal_metadata=False)
        assert signal_dict is None

    def test_filters_sync_words(self, processed_df):
        """Should not include sync word signals."""
        sync_words = ['VAR_1107', 'VAR_2670', 'VAR_5107', 'VAR_6670']
        signal_names = processed_df.select('signal_name').unique().to_series().to_list()
        for sync in sync_words:
            assert sync not in signal_names, f"Sync word not filtered: {sync}"

    def test_has_multiple_categories(self, processed_df):
        """Should have signals from multiple categories."""
        categories = processed_df.select('category').unique().to_series().to_list()
        assert len(categories) > 1, "Expected multiple categories"


# =============================================================================
# Tests: build_signal_dictionary
# =============================================================================

class TestBuildSignalDictionary:

    def test_builds_dictionary(self, mat_files):
        """Should build a signal dictionary."""
        if len(mat_files) < 2:
            pytest.skip("Need at least 2 MAT files")

        signal_dict = build_signal_dictionary(mat_files, sample_size=3)
        assert signal_dict is not None
        assert isinstance(signal_dict, pl.DataFrame)

    def test_dictionary_has_required_columns(self, mat_files):
        """Dictionary should have all metadata columns."""
        if len(mat_files) < 2:
            pytest.skip("Need at least 2 MAT files")

        signal_dict = build_signal_dictionary(mat_files, sample_size=3)
        required = ['signal_name', 'description', 'units', 'rate_hz', 'category', 'priority']
        for col in required:
            assert col in signal_dict.columns, f"Missing column: {col}"

    def test_signals_are_unique(self, mat_files):
        """Each signal should appear only once."""
        if len(mat_files) < 2:
            pytest.skip("Need at least 2 MAT files")

        signal_dict = build_signal_dictionary(mat_files, sample_size=3)
        total = len(signal_dict)
        unique = signal_dict.select('signal_name').n_unique()
        assert total == unique, "Duplicate signals in dictionary"


# =============================================================================
# Tests: validate_data
# =============================================================================

class TestValidateData:

    def test_returns_metrics(self, processed_df):
        """Should return validation metrics."""
        metrics = validate_data(processed_df)
        assert isinstance(metrics, dict)

    def test_has_required_metrics(self, processed_df):
        """Should have all required metrics."""
        metrics = validate_data(processed_df)
        required = ['num_rows', 'num_cols', 'num_signals', 'memory_mb']
        for key in required:
            assert key in metrics, f"Missing metric: {key}"

    def test_num_rows_positive(self, processed_df):
        """num_rows should be positive."""
        metrics = validate_data(processed_df)
        assert metrics['num_rows'] > 0


# =============================================================================
# Integration test
# =============================================================================

class TestIntegration:

    def test_full_pipeline(self, mat_files):
        """Test the full processing pipeline."""
        if not mat_files:
            pytest.skip("No MAT files found")

        # Process a file
        mat_file = Path(mat_files[0])
        df, metadata, signal_dict = process_mat_file(mat_file, extract_signal_metadata=True)

        assert df is not None
        assert metadata is not None
        assert signal_dict is not None

        # Validate
        metrics = validate_data(df)
        assert metrics['num_rows'] > 0
        assert metrics['num_signals'] > 0

        # Check categories are applied
        categories = df.select('category').unique().to_series().to_list()
        assert 'uncategorized' not in categories or len(categories) > 1