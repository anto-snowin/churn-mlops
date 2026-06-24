"""
Phase 9 — SHAP Explainability Layer.

Generates global and local SHAP explanations for the trained
GradientBoostingClassifier model.

Outputs (saved to reports/):
    1. shap_summary.png           — dot-style summary plot
    2. shap_importance.png        — bar-style mean |SHAP| importance
    3. shap_single_prediction.png — waterfall plot for the first test customer

Usage:
    python src/explain.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # server-safe backend — must be set before pyplot import
import matplotlib.pyplot as plt

import numpy as np
import shap

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
MODEL_DIR = Path("model")
REPORTS_DIR = Path("reports")


# ══════════════════════════════════════════════════════════════════════════════
# Core explainability function
# ══════════════════════════════════════════════════════════════════════════════
def generate_shap_explanations(model, X_train, X_test):
    """
    Generate SHAP explanations and save three plots to reports/.

    Uses TreeExplainer (optimal for GradientBoosting / tree-based models)
    to compute SHAP values for X_test.

    Plots saved:
        1. reports/shap_summary.png           — dot summary plot
        2. reports/shap_importance.png        — bar plot (mean |SHAP|)
        3. reports/shap_single_prediction.png — waterfall for first test customer

    Args:
        model:   A fitted tree-based sklearn model (e.g. GradientBoostingClassifier).
        X_train: Training features (used as background data for TreeExplainer).
        X_test:  Test features to explain.

    Returns:
        shap_values: The computed SHAP values array for X_test.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Create TreeExplainer ──────────────────────────────────────────────────
    logger.info("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model, data=X_train)

    # ── Compute SHAP values ───────────────────────────────────────────────────
    logger.info(f"Computing SHAP values for {X_test.shape[0]} test samples...")
    shap_values = explainer.shap_values(X_test)

    # For binary classifiers, shap_values can be a list [class_0, class_1].
    # We always want the positive-class (churn = 1) SHAP values.
    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1]
    else:
        shap_values_churn = shap_values

    logger.info(f"SHAP values shape: {shap_values_churn.shape}")

    # ── 1. Summary dot plot ───────────────────────────────────────────────────
    logger.info("Generating shap_summary.png (dot plot)...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_churn, X_test, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ shap_summary.png saved")

    # ── 2. Feature importance bar plot ────────────────────────────────────────
    logger.info("Generating shap_importance.png (bar plot)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_churn, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ shap_importance.png saved")

    # ── 3. Waterfall plot for the first test customer ─────────────────────────
    logger.info("Generating shap_single_prediction.png (waterfall)...")
    try:
        # Resolve expected_value for positive class
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
            base_value = float(ev[1])
        elif isinstance(ev, np.ndarray):
            base_value = float(ev[0])
        else:
            base_value = float(ev)

        # Build a shap.Explanation for the first test sample
        explanation = shap.Explanation(
            values=shap_values_churn[0],
            base_values=base_value,
            data=X_test.iloc[0].values if hasattr(X_test, "iloc") else X_test[0],
            feature_names=(
                X_test.columns.tolist() if hasattr(X_test, "columns")
                else [f"feature_{i}" for i in range(X_test.shape[1])]
            ),
        )

        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig(
            REPORTS_DIR / "shap_single_prediction.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()
        logger.info("  ✓ shap_single_prediction.png saved")

    except Exception as e:
        logger.warning(f"Waterfall plot failed: {e}")
        plt.close("all")

    # ── Log top features by mean |SHAP| ───────────────────────────────────────
    mean_abs = np.abs(shap_values_churn).mean(axis=0)
    feature_names = (
        X_test.columns.tolist() if hasattr(X_test, "columns")
        else [f"feature_{i}" for i in range(X_test.shape[1])]
    )
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)

    logger.info("\nTop 10 features by mean |SHAP value|:")
    for name, val in ranked[:10]:
        logger.info(f"  {name:25s}  {val:.4f}")

    logger.info(f"\nAll SHAP plots saved to {REPORTS_DIR}/")
    return shap_values


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import joblib

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_and_clean(RAW_PATH)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run: python src/train.py")
        sys.exit(1)

    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    # ── Generate explanations ─────────────────────────────────────────────────
    shap_values = generate_shap_explanations(model, X_train, X_test)

    print(f"\n{'═'*60}")
    print("  SHAP EXPLANATIONS COMPLETE")
    print(f"{'═'*60}")
    print(f"  Plots saved to: {REPORTS_DIR}/")
    print(f"    • shap_summary.png           (dot plot)")
    print(f"    • shap_importance.png        (bar chart)")
    print(f"    • shap_single_prediction.png (waterfall)")
    print(f"{'═'*60}")
