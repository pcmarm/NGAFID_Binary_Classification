#!/usr/bin/env python3
"""
NGAFID Dataset Dask Example

This script demonstrates how to use Dask to efficiently work with the larger NGAFID
'all_flights' dataset. Dask enables parallel and out-of-core computation for large
datasets that don't fit in memory.

The 'all_flights' dataset contains parquet files that can be efficiently loaded and
processed using Dask DataFrames.

Paper: "A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID"
"""

import os
import sys
import pandas as pd

# Add the project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from ngafiddataset.dataset.dataset import NGAFID_Dataset_Manager

import dask.dataframe as dd
from tqdm import tqdm


# ============================================================================
# DATA EXPLORATION FUNCTIONS
# ============================================================================

def explore_flight_data(flight_data_df, num_rows=5):
    """
    Explore the flight data dataframe.

    Args:
        flight_data_df: Dask DataFrame containing flight data
        num_rows: Number of rows to display
    """
    print("=" * 60)
    print("Flight Data Exploration")
    print("=" * 60)

    print("\nDask DataFrame Structure:")
    print(flight_data_df)

    print("\nFirst few rows:")
    print(flight_data_df.head(num_rows))

    print("\nColumn names:")
    print(flight_data_df.columns.tolist())

    print("\nData types:")
    print(flight_data_df.dtypes)

    return flight_data_df


def explore_flight_header(flight_header_df, num_rows=5):
    """
    Explore the flight header dataframe.

    Args:
        flight_header_df: Pandas DataFrame containing flight header information
        num_rows: Number of rows to display
    """
    print("=" * 60)
    print("Flight Header Exploration")
    print("=" * 60)

    print("\nFirst few rows:")
    print(flight_header_df.head(num_rows))

    print("\nColumn names:")
    print(flight_header_df.columns.tolist())

    print("\nData types:")
    print(flight_header_df.dtypes)

    print("\nBasic statistics:")
    print(flight_header_df.describe())

    print("\nUnique labels:")
    if 'label' in flight_header_df.columns:
        print(flight_header_df['label'].value_counts())

    return flight_header_df


def get_label_statistics(flight_header_df):
    """
    Get statistics about labels in the dataset.

    Args:
        flight_header_df: Pandas DataFrame containing flight header information
    """
    print("\n" + "=" * 60)
    print("Label Statistics")
    print("=" * 60)

    if 'label' in flight_header_df.columns:
        print("\nLabel value counts:")
        print(flight_header_df['label'].value_counts())

    if 'before_after' in flight_header_df.columns:
        print("\nBefore/After value counts:")
        print(flight_header_df['before_after'].value_counts())

    if 'fold' in flight_header_df.columns:
        print("\nFold distribution:")
        print(flight_header_df['fold'].value_counts())

    if 'class' in flight_header_df.columns:
        print("\nClass distribution:")
        print(flight_header_df['class'].value_counts())

    return {
        'num_flights': len(flight_header_df),
        'num_labels': flight_header_df['label'].nunique() if 'label' in flight_header_df.columns else 0,
        'num_classes': flight_header_df['class'].nunique() if 'class' in flight_header_df.columns else 0
    }


def get_data_statistics(flight_data_df):
    """
    Get statistics about the flight data using Dask.

    Args:
        flight_data_df: Dask DataFrame containing flight data
    """
    print("\n" + "=" * 60)
    print("Flight Data Statistics")
    print("=" * 60)

    # Compute basic statistics using Dask
    print("\nComputing mean for each column (lazy evaluation)...")
    means = flight_data_df.mean().compute()
    print(means)

    print("\nComputing standard deviation for each column...")
    stds = flight_data_df.std().compute()
    print(stds)

    print("\nNumber of partitions:", flight_data_df.npartitions)
    print("Number of rows:", len(flight_data_df))

    return {'means': means, 'stds': stds}


def filter_flights_by_label(flight_data_df, flight_header_df, label_name):
    """
    Filter flights by label name.

    Args:
        flight_data_df: Dask DataFrame containing flight data
        flight_header_df: Pandas DataFrame containing flight header information
        label_name: Name of the label to filter by

    Returns:
        Filtered Dask DataFrame
    """
    print(f"\nFiltering flights with label: {label_name}")

    # Get indices for flights with this label
    indices = flight_header_df[
        flight_header_df['label'] == label_name
    ].index.tolist()

    print(f"Found {len(indices)} flights with label '{label_name}'")

    if len(indices) > 0:
        # Filter the flight data
        filtered_df = flight_data_df[flight_data_df.index.get_level_values('Master Index').isin(indices)]
        print(f"Filtered DataFrame shape: {filtered_df.shape}")
        return filtered_df

    return None


def filter_flights_by_fold(flight_header_df, fold_num):
    """
    Filter flight headers by fold number.

    Args:
        flight_header_df: Pandas DataFrame containing flight header information
        fold_num: Fold number to filter by

    Returns:
        Filtered DataFrame
    """
    print(f"\nFiltering flights for fold: {fold_num}")

    filtered_df = flight_header_df[flight_header_df['fold'] == fold_num]
    print(f"Found {len(filtered_df)} flights in fold {fold_num}")

    return filtered_df


def filter_flights_before_after(flight_header_df, condition='before'):
    """
    Filter flights by before/after condition.

    Args:
        flight_header_df: Pandas DataFrame containing flight header information
        condition: 'before', 'after', or 'same'

    Returns:
        Filtered DataFrame
    """
    print(f"\nFiltering flights where before_after = '{condition}'")

    filtered_df = flight_header_df[flight_header_df['before_after'] == condition]
    print(f"Found {len(filtered_df)} flights with before_after = '{condition}'")

    return filtered_df


# ============================================================================
# DATA ANALYSIS EXAMPLES
# ============================================================================

def analyze_engine_parameters(flight_data_df, engine='E1'):
    """
    Analyze engine-related parameters.

    Args:
        flight_data_df: Dask DataFrame containing flight data
        engine: Engine identifier (e.g., 'E1' for engine 1)
    """
    print("\n" + "=" * 60)
    print(f"Analyzing {engine} Engine Parameters")
    print("=" * 60)

    # Find columns related to this engine
    engine_cols = [col for col in flight_data_df.columns if engine in col]

    if len(engine_cols) > 0:
        print(f"\nFound {len(engine_cols)} columns for {engine}:")
        print(engine_cols)

        # Compute statistics for engine columns
        print(f"\nStatistics for {engine} parameters:")
        engine_stats = flight_data_df[engine_cols].describe().compute()
        print(engine_stats)
    else:
        print(f"No columns found for {engine}")


def analyze_sensor_correlations(flight_data_df, num_samples=10000):
    """
    Analyze correlations between sensors.

    Args:
        flight_data_df: Dask DataFrame containing flight data
        num_samples: Number of samples to use for correlation analysis
    """
    print("\n" + "=" * 60)
    print("Sensor Correlation Analysis")
    print("=" * 60)

    # Sample data for correlation analysis
    print(f"\nSampling {num_samples} rows for correlation analysis...")
    sample_df = flight_data_df.head(num_samples)

    # Compute correlation matrix
    print("Computing correlation matrix...")
    corr_matrix = sample_df.corr()

    print("\nCorrelation matrix shape:", corr_matrix.shape)

    # Find highly correlated pairs
    print("\nHighly correlated sensor pairs (|correlation| > 0.8):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                print(f"  {corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def load_all_flights_dataset(destination=''):
    """
    Load the 'all_flights' dataset.

    Args:
        destination: Directory where the dataset is/will be stored

    Returns:
        Tuple of (flight_data_df, flight_header_df)
    """
    print("=" * 60)
    print("Loading All Flights Dataset")
    print("=" * 60)

    # Initialize dataset manager (this will download if needed)
    print("\nInitializing NGAFID Dataset Manager...")
    _ = NGAFID_Dataset_Manager('all_flights', destination=destination)

    # Load data with Dask
    print("\nLoading flight data with Dask...")
    flight_data_df = dd.read_parquet(os.path.join(destination, 'all_flights', 'one_parq'))

    # Load header data with Pandas
    print("Loading flight header data with Pandas...")
    flight_header_df = pd.read_csv(
        os.path.join(destination, 'all_flights', 'flight_header.csv'),
        index_col='Master Index'
    )

    print("\nDataset loaded successfully!")
    print(f"Flight data shape: {flight_data_df.shape}")
    print(f"Flight header shape: {flight_header_df.shape}")

    return flight_data_df, flight_header_df


def run_full_analysis(destination='', show_plots=False):
    """
    Run a full analysis on the all_flights dataset.

    Args:
        destination: Directory where the dataset is stored
        show_plots: Whether to show matplotlib plots (not implemented)
    """
    # Load data
    flight_data_df, flight_header_df = load_all_flights_dataset(destination)

    # Explore data
    explore_flight_data(flight_data_df)
    explore_flight_header(flight_header_df)

    # Get statistics
    label_stats = get_label_statistics(flight_header_df)
    data_stats = get_data_statistics(flight_data_df)

    # Example filters
    gasket_flights = filter_flights_by_label(
        flight_data_df,
        flight_header_df,
        'intake gasket leak/damage'
    )

    # Analyze engine parameters
    analyze_engine_parameters(flight_data_df, 'E1')

    return {
        'label_stats': label_stats,
        'data_stats': data_stats,
        'gasket_flights': gasket_flights
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NGAFID Dataset Dask Example - Load and Explore Large-Scale Data"
    )
    parser.add_argument(
        "--destination",
        type=str,
        default="",
        help="Destination directory for dataset"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Force download of the dataset"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick exploration only"
    )
    parser.add_argument(
        "--analyze_engine",
        type=str,
        default="E1",
        help="Engine to analyze (default: E1)"
    )
    parser.add_argument(
        "--filter_label",
        type=str,
        default=None,
        help="Filter flights by label"
    )
    parser.add_argument(
        "--filter_fold",
        type=int,
        default=None,
        help="Filter flights by fold number"
    )

    args = parser.parse_args()

    if args.quick:
        # Quick exploration
        flight_data_df, flight_header_df = load_all_flights_dataset(args.destination)
        explore_flight_data(flight_data_df)
        explore_flight_header(flight_header_df)
    elif args.filter_label:
        # Filter by label
        flight_data_df, flight_header_df = load_all_flights_dataset(args.destination)
        filtered = filter_flights_by_label(flight_data_df, flight_header_df, args.filter_label)
        if filtered is not None:
            print(f"\nFiltered data preview:")
            print(filtered.head())
    elif args.filter_fold is not None:
        # Filter by fold
        _, flight_header_df = load_all_flights_dataset(args.destination)
        filtered = filter_flights_by_fold(flight_header_df, args.filter_fold)
        print(f"\nFold {args.filter_fold} flights:")
        print(filtered.head())
    else:
        # Full analysis
        run_full_analysis(args.destination)
