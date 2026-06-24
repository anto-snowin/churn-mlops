"""
Data preprocessing module for churn prediction.

Functions:
    load_and_clean  — Read CSV, fix TotalCharges, drop customerID
    encode_features — LabelEncode all object/string columns
    split_data      — Train/test split (80/20, stratified)

Usage:
    python src/data_preprocessing.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_PATH = "data/processed/churn_processed.csv"


def load_and_clean(path: str) -> pd.DataFrame:
    """
    Read the Telco churn CSV, fix known data issues, and return a clean DataFrame.

    Steps:
        1. Read CSV from *path*.
        2. Convert TotalCharges to numeric (some rows contain whitespace strings
           instead of numbers — coerce those to NaN, then fill with the column median).
        3. Drop the customerID column (not a predictive feature).

    Args:
        path: File path to the raw CSV.

    Returns:
        Cleaned DataFrame with TotalCharges as float64 and no customerID column.
    """
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    # ── Fix TotalCharges ──────────────────────────────────────────────────────
    # The IBM Telco CSV stores TotalCharges as object because some rows have " "
    # (a single space) instead of a number. Coerce → NaN → fill with median.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_total = df["TotalCharges"].median()
    n_missing = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total)
    logger.info(f"TotalCharges: filled {n_missing} missing values with median ({median_total:.2f})")

    # ── Drop customerID ───────────────────────────────────────────────────────
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Dropped customerID column")

    logger.info(f"Cleaned shape: {df.shape}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode every object/string column in the DataFrame.

    Each unique string value is mapped to an integer (0, 1, 2, …).
    The encoding is applied in-place on a copy of the input.

    Args:
        df: DataFrame with object-typed columns to encode.

    Returns:
        DataFrame with all former string columns converted to int64.
    """
    df = df.copy()
    label_encoders: dict[str, LabelEncoder] = {}

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    logger.info(f"Label-encoding {len(object_cols)} columns: {object_cols}")

    for col in object_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    logger.info("Encoding complete — all columns are now numeric")
    return df


def split_data(df: pd.DataFrame) -> tuple:
    """
    Separate features and target, then split into train/test sets.

    Target column: 'Churn'
    Split ratio:   80 % train / 20 % test
    Random state:  42 (reproducible)

    Args:
        df: Fully preprocessed (cleaned + encoded) DataFrame.

    Returns:
        (X_train, X_test, y_train, y_test) — numpy arrays or DataFrames.
    """
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logger.info(
        f"Split → Train: {X_train.shape[0]} samples | "
        f"Test: {X_test.shape[0]} samples"
    )
    return X_train, X_test, y_train, y_test


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load and clean
    df = load_and_clean(RAW_PATH)

    # 2. Encode
    df = encode_features(df)

    # 3. Save processed CSV
    out = Path(PROCESSED_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print(f"[OK] Processed data saved to {out}")
    print(f"     Shape: {df.shape}")
    print(f"     Columns: {list(df.columns)}")
    print(f"     Churn distribution:\n{df['Churn'].value_counts().to_string()}")
    print(f"{'='*60}")
