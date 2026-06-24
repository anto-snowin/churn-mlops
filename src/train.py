"""
Model training module with MLflow experiment tracking.

Trains multiple classifiers on the Telco churn dataset, logs parameters,
metrics, and model artifacts to MLflow, and registers each model in the
MLflow Model Registry.

Usage:
    python src/train.py
"""

import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
TRACKING_URI = "sqlite:///mlflow.db"


def train_and_log(model, model_name: str, X_train, X_test, y_train, y_test) -> float:
    """
    Train a model, evaluate it, and log everything to MLflow.

    Steps:
        1. Train the model on (X_train, y_train).
        2. Predict on X_test and compute accuracy, F1, and ROC-AUC.
        3. Log all model hyperparameters via mlflow.log_params().
        4. Log accuracy, f1, and roc_auc as MLflow metrics.
        5. Log the trained model artifact with mlflow.sklearn.log_model().
        6. Register the model in the MLflow Model Registry as "churn_{model_name}".

    Args:
        model:      An unfitted scikit-learn estimator.
        model_name: Short name (e.g. "LogisticRegression") used for the
                    MLflow run name and model registry entry.
        X_train:    Training features.
        X_test:     Test features.
        y_train:    Training labels.
        y_test:     Test labels.

    Returns:
        The ROC-AUC score on the test set.
    """
    with mlflow.start_run(run_name=model_name):
        # ── 1. Train ──────────────────────────────────────────────────────────
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)

        # ── 2. Predict & evaluate ─────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  ROC-AUC:  {auc:.4f}")

        # ── 3. Log parameters ────────────────────────────────────────────────
        mlflow.log_params(model.get_params())

        # ── 4. Log metrics ────────────────────────────────────────────────────
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # ── 5. Log model artifact ─────────────────────────────────────────────
        mlflow.sklearn.log_model(model, artifact_path="model")

        # ── 6. Register model ─────────────────────────────────────────────────
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        registered_name = f"churn_{model_name}"

        mlflow.register_model(model_uri, registered_name)
        logger.info(f"  Registered as '{registered_name}' in Model Registry")

    return auc


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Configure MLflow ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("churn-prediction")
    logger.info(f"MLflow tracking URI: {TRACKING_URI}")

    # ── Load and preprocess data ──────────────────────────────────────────────
    df = load_and_clean(RAW_PATH)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # ── Define models to train ────────────────────────────────────────────────
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    # ── Train each model ──────────────────────────────────────────────────────
    results: dict[str, float] = {}

    for name, model in models.items():
        auc = train_and_log(model, name, X_train, X_test, y_train, y_test)
        results[name] = auc

    # ── Print summary ─────────────────────────────────────────────────────────
    best_name = max(results, key=results.get)
    best_auc = results[best_name]

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE -- Results (ROC-AUC)")
    print(f"{'='*60}")
    for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " <-- BEST" if name == best_name else ""
        print(f"  {name:30s}  AUC = {auc:.4f}{marker}")
    print(f"{'='*60}")
    print(f"\nBest model: {best_name} (AUC = {best_auc:.4f})")
    print(f"Registered in MLflow as: churn_{best_name}")
    print(f"\nLaunch MLflow UI:  mlflow ui --backend-store-uri {TRACKING_URI}")
    print(f"{'='*60}")
