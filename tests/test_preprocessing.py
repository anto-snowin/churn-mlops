"""
Tests for src/data_preprocessing.py

Covers:
    - test_load_returns_dataframe      → load_and_clean returns a pd.DataFrame
    - test_no_null_values_after_clean  → no NaN values remain after cleaning
    - test_split_correct_sizes         → train/test shapes match 80/20 split
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_preprocessing import load_and_clean, encode_features, split_data

# Path to the real Telco CSV (all tests depend on this file existing)
RAW_CSV = str(Path(__file__).resolve().parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def cleaned_df() -> pd.DataFrame:
    """Load and clean the raw dataset once for all tests in this module."""
    return load_and_clean(RAW_CSV)


@pytest.fixture(scope="module")
def encoded_df(cleaned_df) -> pd.DataFrame:
    """Encode the cleaned dataset once for all tests in this module."""
    return encode_features(cleaned_df)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_returns_dataframe(cleaned_df):
    """load_and_clean() must return a pandas DataFrame."""
    assert isinstance(cleaned_df, pd.DataFrame), (
        f"Expected pd.DataFrame, got {type(cleaned_df).__name__}"
    )
    # Should have rows and columns
    assert cleaned_df.shape[0] > 0, "DataFrame has no rows"
    assert cleaned_df.shape[1] > 0, "DataFrame has no columns"
    # customerID should be gone
    assert "customerID" not in cleaned_df.columns, (
        "customerID column should have been dropped"
    )


def test_no_null_values_after_clean(cleaned_df):
    """After load_and_clean(), there should be zero NaN values anywhere."""
    null_counts = cleaned_df.isnull().sum()
    total_nulls = null_counts.sum()
    assert total_nulls == 0, (
        f"Found {total_nulls} null values in columns: "
        f"{null_counts[null_counts > 0].to_dict()}"
    )
    # TotalCharges specifically must be numeric
    assert cleaned_df["TotalCharges"].dtype in ("float64", "float32"), (
        f"TotalCharges dtype is {cleaned_df['TotalCharges'].dtype}, expected float"
    )


def test_split_correct_sizes(encoded_df):
    """
    split_data() with test_size=0.2 must produce:
        - X_train ≈ 80% of total rows
        - X_test  ≈ 20% of total rows
        - Feature counts match between train and test
    """
    X_train, X_test, y_train, y_test = split_data(encoded_df)

    total = len(encoded_df)
    expected_test = int(total * 0.2)

    # Allow ±1 row rounding tolerance
    assert abs(len(X_test) - expected_test) <= 1, (
        f"Test set size {len(X_test)} doesn't match expected ~{expected_test}"
    )
    assert abs(len(X_train) - (total - expected_test)) <= 1, (
        f"Train set size {len(X_train)} doesn't match expected ~{total - expected_test}"
    )

    # Train + test = total
    assert len(X_train) + len(X_test) == total, (
        f"Train ({len(X_train)}) + Test ({len(X_test)}) != Total ({total})"
    )

    # Same number of features
    assert X_train.shape[1] == X_test.shape[1], (
        f"Feature count mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}"
    )

    # Target lengths match
    assert len(y_train) == len(X_train), "y_train length mismatch"
    assert len(y_test) == len(X_test), "y_test length mismatch"
