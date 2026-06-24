"""
Phase 10 — Automated Retrain Trigger.

Checks for data drift using Evidently's DataDriftPreset.
If the share of drifted columns exceeds DRIFT_THRESHOLD, triggers
a full pipeline retrain via ``dvc repro``.

All decisions are appended to monitoring/drift_log.jsonl for audit.

Usage:
    python src/retrain_trigger.py
    python src/retrain_trigger.py --reference data/processed/churn_processed.csv --current data/current/new_data.csv
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preprocessing import load_and_clean, encode_features, split_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DRIFT_THRESHOLD = 0.15
RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MONITORING_DIR = Path("monitoring")
TARGET_COL = "Churn"


# ══════════════════════════════════════════════════════════════════════════════
# Core drift-check + retrain function
# ══════════════════════════════════════════════════════════════════════════════
def check_drift_and_retrain(reference_path: str, current_path: str) -> dict:
    """
    Load reference and current CSVs, run Evidently DataDriftPreset,
    and trigger ``dvc repro`` if drift exceeds DRIFT_THRESHOLD.

    Decision logic:
        - Compute share_of_drifted_columns from the Evidently report.
        - If share > DRIFT_THRESHOLD (0.15): retrain via ``dvc repro``.
        - Otherwise: log "not_needed".
        - All outcomes are appended to monitoring/drift_log.jsonl.

    Args:
        reference_path: Path to the reference (training) CSV.
        current_path:   Path to the current (production) CSV.

    Returns:
        Status dict with keys:
            timestamp, reference_path, current_path,
            share_drifted, drift_threshold, retrain_status
    """
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info(f"Loading reference: {reference_path}")
    reference_df = pd.read_csv(reference_path)

    logger.info(f"Loading current:   {current_path}")
    current_df = pd.read_csv(current_path)

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

    # ── Run Evidently DataDriftPreset ─────────────────────────────────────────
    logger.info("Running Evidently DataDriftPreset...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    # ── Extract share_of_drifted_columns ──────────────────────────────────────
    report_dict = report.as_dict()
    share_drifted = 0.0
    n_drifted = 0
    drifted_features = []

    for metric_result in report_dict.get("metrics", []):
        result = metric_result.get("result", {})
        if "share_of_drifted_columns" in result:
            share_drifted = result["share_of_drifted_columns"]
            n_drifted = result.get("number_of_drifted_columns", 0)

        if "drift_by_columns" in result:
            for col_name, col_info in result["drift_by_columns"].items():
                if col_info.get("drift_detected", False):
                    drifted_features.append(col_name)

    logger.info(f"Share of drifted columns: {share_drifted:.2%} "
                f"(threshold: {DRIFT_THRESHOLD:.0%})")
    logger.info(f"Drifted features ({n_drifted}): {drifted_features}")

    # ── Build status dict ─────────────────────────────────────────────────────
    status = {
        "timestamp": datetime.now().isoformat(),
        "reference_path": reference_path,
        "current_path": current_path,
        "share_drifted": round(share_drifted, 4),
        "n_drifted_features": n_drifted,
        "drifted_features": drifted_features,
        "drift_threshold": DRIFT_THRESHOLD,
    }

    # ── Decision: retrain or not ──────────────────────────────────────────────
    if share_drifted > DRIFT_THRESHOLD:
        logger.warning(
            f"DRIFT EXCEEDS THRESHOLD ({share_drifted:.2%} > {DRIFT_THRESHOLD:.0%}) "
            f"— triggering retraining via 'dvc repro'..."
        )

        try:
            subprocess.run(["dvc", "repro"], check=True)
            status["retrain_status"] = "completed"
            logger.info("Retraining completed successfully via 'dvc repro'")
        except subprocess.CalledProcessError as e:
            status["retrain_status"] = f"failed: {e}"
            logger.error(f"Retraining failed: {e}")
        except FileNotFoundError:
            status["retrain_status"] = "failed: dvc not found"
            logger.error("DVC not found on PATH. Install with: pip install dvc")

    else:
        status["retrain_status"] = "not_needed"
        logger.info(
            f"Drift within threshold ({share_drifted:.2%} <= {DRIFT_THRESHOLD:.0%}) "
            f"— retraining not needed."
        )

    # ── Append status to JSONL log ────────────────────────────────────────────
    log_path = MONITORING_DIR / "drift_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(status) + "\n")
    logger.info(f"Status appended to {log_path}")

    return status


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check for data drift and trigger retraining if needed"
    )
    parser.add_argument(
        "--reference",
        default="data/processed/churn_processed.csv",
        help="Path to reference (training) data CSV",
    )
    parser.add_argument(
        "--current",
        default="data/processed/churn_processed.csv",
        help="Path to current (production) data CSV "
             "(defaults to processed for demo — replace with real current data)",
    )
    args = parser.parse_args()

    status = check_drift_and_retrain(
        reference_path=args.reference,
        current_path=args.current,
    )

    print(f"\n{'='*60}")
    print("  DRIFT CHECK COMPLETE")
    print(f"{'='*60}")
    print(f"  Share drifted: {status['share_drifted']:.2%}")
    print(f"  Threshold:     {status['drift_threshold']:.0%}")
    print(f"  Retrain:       {status['retrain_status']}")
    if status["drifted_features"]:
        print(f"  Drifted:       {status['drifted_features']}")
    print(f"{'='*60}")
