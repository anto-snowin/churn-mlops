"""
Phase 10 — Drift Monitoring with Evidently AI.

Compares a reference dataset (training data) against current/production data
to detect feature drift and classification quality degradation.

Outputs:
    - monitoring/drift_report.html — interactive Evidently HTML report

Usage:
    python src/monitor.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import DataDriftTable, ClassificationQualityMetric
from evidently.report import Report

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preprocessing import load_and_clean, encode_features, split_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths & constants ─────────────────────────────────────────────────────────
RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MONITORING_DIR = Path("monitoring")
TARGET_COL = "Churn"


# ══════════════════════════════════════════════════════════════════════════════
# Core drift reporting function
# ══════════════════════════════════════════════════════════════════════════════
def generate_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    """
    Generate an Evidently drift report comparing reference vs current data.

    Uses:
        - DataDriftTable()             — per-feature drift detection
        - ClassificationQualityMetric() — classification performance metrics

    Aligns current_df columns to reference_df to avoid shape mismatches.
    Saves the interactive HTML report to monitoring/drift_report.html.

    Args:
        reference_df: The reference (training) dataset with target column.
        current_df:   The current (production) dataset with target column.

    Returns:
        The Evidently Report object (can be further inspected via
        report.as_dict() or report.as_json()).
    """
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    # ── Align columns to prevent shape mismatch ──────────────────────────────
    current_df = current_df[reference_df.columns]

    logger.info(f"Reference shape: {reference_df.shape}")
    logger.info(f"Current shape:   {current_df.shape}")

    # ── Column mapping ────────────────────────────────────────────────────────
    column_mapping = ColumnMapping()
    column_mapping.target = TARGET_COL

    numeric_features = reference_df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET_COL in numeric_features:
        numeric_features.remove(TARGET_COL)
    column_mapping.numerical_features = numeric_features

    # ── Build report with DataDriftTable + ClassificationQualityMetric ────────
    logger.info("Building Evidently report (DataDriftTable + ClassificationQualityMetric)...")

    report = Report(metrics=[
        DataDriftTable(),
        ClassificationQualityMetric(),
    ])

    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    # ── Save HTML ─────────────────────────────────────────────────────────────
    report_path = MONITORING_DIR / "drift_report.html"
    report.save_html(str(report_path))
    logger.info(f"Drift report saved to {report_path}")

    # ── Log summary ───────────────────────────────────────────────────────────
    try:
        report_dict = report.as_dict()
        for metric_result in report_dict.get("metrics", []):
            result = metric_result.get("result", {})
            if "share_of_drifted_columns" in result:
                n_drifted = result.get("number_of_drifted_columns", 0)
                share = result.get("share_of_drifted_columns", 0.0)
                logger.info(f"Drifted columns: {n_drifted} ({share:.1%})")
    except Exception as e:
        logger.warning(f"Could not parse summary: {e}")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_clean(RAW_PATH)
    df = encode_features(df)

    # Split into reference (train) and current (test) for demo
    X_train, X_test, y_train, y_test = split_data(df)

    # Reconstruct full DataFrames with the target column for Evidently
    reference_df = X_train.copy()
    reference_df[TARGET_COL] = y_train.values

    current_df = X_test.copy()
    current_df[TARGET_COL] = y_test.values

    report = generate_drift_report(reference_df, current_df)

    print(f"\n{'='*60}")
    print("  DRIFT MONITORING COMPLETE")
    print(f"{'='*60}")
    print(f"  Report: monitoring/drift_report.html")
    print(f"  Open in browser to inspect per-feature drift details")
    print(f"{'='*60}")
