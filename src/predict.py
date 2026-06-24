"""
Prediction module.

Loads a trained model (local joblib or MLflow URI) and runs
batch or single inference. Designed for both CLI batch mode
and programmatic use by the FastAPI serving layer.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from the central config file."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model_uri: str | None = None):
    """
    Load a model from a local path or MLflow URI.

    Priority:
    1. Explicit model_uri argument
    2. MLflow URI (if it starts with 'models:/' or 'runs:/')
    3. Local joblib file
    """
    if model_uri is None:
        params = load_params()
        model_uri = params["api"]["model_uri"]

    # Try MLflow URI first
    if model_uri.startswith(("models:/", "runs:/")):
        try:
            import mlflow.sklearn
            logger.info(f"Loading model from MLflow: {model_uri}")
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            logger.warning(f"MLflow load failed: {e}. Falling back to local file.")
            params = load_params()
            model_uri = str(Path(params["model"]["save_dir"]) / params["model"]["artifact_name"])

    # Local file
    model_path = Path(model_uri)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from local file: {model_path}")
    return joblib.load(model_path)


def load_feature_names(model_dir: str = "model") -> list[str] | None:
    """Load expected feature names from the saved feature_names.json."""
    path = Path(model_dir) / "feature_names.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def predict_batch(model, input_df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Run batch prediction and return DataFrame with probabilities and labels.

    Automatically drops the target column if present in the input.
    """
    # Drop target column if present
    if target_col in input_df.columns:
        input_df = input_df.drop(columns=[target_col])

    probs = model.predict_proba(input_df)[:, 1]
    labels = (probs >= 0.5).astype(int)

    result = input_df.copy()
    result["churn_probability"] = np.round(probs, 4)
    result["churn_prediction"] = labels
    return result


def predict_single(model, features: dict) -> dict:
    """
    Run a single prediction from a feature dictionary.

    Returns dict with churn_probability and churn_prediction.
    """
    df = pd.DataFrame([features])
    proba = model.predict_proba(df)[0, 1]
    label = int(proba >= 0.5)
    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction": label,
    }


def main() -> None:
    """CLI entry point for batch prediction."""
    import argparse

    parser = argparse.ArgumentParser(description="Run batch prediction")
    parser.add_argument("--model-uri", default=None, help="MLflow URI or local path to model")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="reports/predictions.csv", help="Output CSV path")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    target_col = params["data"]["target_column"]

    model = load_model(args.model_uri)
    df = pd.read_csv(args.input)

    logger.info(f"Running batch prediction on {len(df)} rows")
    results = predict_batch(model, df, target_col=target_col)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")

    # Summary stats
    n_churn = (results["churn_prediction"] == 1).sum()
    logger.info(f"Predicted churn: {n_churn}/{len(results)} ({n_churn/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
