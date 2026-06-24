"""
Phase 8 — Bayesian Hyperparameter Optimization with Optuna.

Searches GradientBoostingClassifier hyperparameters using Optuna's
Tree-structured Parzen Estimator (TPE) sampler, evaluates via 3-fold
stratified cross-validation (ROC-AUC), and logs the best model + results
to MLflow.

Usage:
    python src/tune.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # server-safe backend — must be set before pyplot import
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preprocessing import load_and_clean, encode_features, split_data

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress Optuna's verbose per-trial logging (we log results ourselves)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths & constants ─────────────────────────────────────────────────────────
RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TRACKING_URI = "sqlite:///mlflow.db"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Optuna objective
# ══════════════════════════════════════════════════════════════════════════════
def objective(trial, X_train, y_train):
    """
    Optuna objective function.

    Samples hyperparameters for a GradientBoostingClassifier and evaluates
    performance via 3-fold cross-validation using ROC-AUC.

    Args:
        trial:   An optuna.Trial object for hyperparameter suggestion.
        X_train: Training features (pd.DataFrame or np.ndarray).
        y_train: Training labels  (pd.Series or np.ndarray).

    Returns:
        Mean ROC-AUC across the 3 cross-validation folds.
    """
    # ── Sample hyperparameters ────────────────────────────────────────────────
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
        "max_depth":        trial.suggest_int("max_depth", 2, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "random_state":     42,
    }

    model = GradientBoostingClassifier(**params)

    # ── 3-fold CV with ROC-AUC scoring ────────────────────────────────────────
    scores = cross_val_score(
        model, X_train, y_train,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
    )

    mean_auc = scores.mean()

    logger.info(
        f"Trial {trial.number:3d}  │  AUC={mean_auc:.4f}  "
        f"n_est={params['n_estimators']:3d}  depth={params['max_depth']}  "
        f"lr={params['learning_rate']:.4f}  sub={params['subsample']:.2f}  "
        f"split={params['min_samples_split']}"
    )

    return mean_auc


# ══════════════════════════════════════════════════════════════════════════════
# 2. Full tuning pipeline
# ══════════════════════════════════════════════════════════════════════════════
def tune_and_log(n_trials=50):
    """
    End-to-end Optuna tuning with MLflow logging.

    Steps:
        1. Load & preprocess the Telco churn dataset.
        2. Create an Optuna study (TPE sampler, maximize AUC).
        3. Run *n_trials* optimisation trials.
        4. Log best params, CV AUC, held-out test AUC, and n_trials to MLflow.
        5. Generate & log a hyperparameter importance plot as an artifact.
        6. Register the best model in the MLflow Model Registry as
           ``churn_tuned_gb``.

    Args:
        n_trials: Number of Optuna trials to run (default 50).

    Returns:
        dict with ``best_params``, ``best_cv_auc``, and ``test_auc``.
    """

    # ── 1. Load and preprocess ────────────────────────────────────────────────
    logger.info("Loading and preprocessing data...")
    df = load_and_clean(RAW_PATH)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    logger.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ── 2. Configure MLflow ───────────────────────────────────────────────────
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("churn-optuna-tuning")

    # ── 3. Create and run the Optuna study ────────────────────────────────────
    logger.info(f"Starting Optuna study — {n_trials} trials")

    study = optuna.create_study(
        direction="maximize",
        study_name="churn-gb-tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # ── 4. Extract best results ───────────────────────────────────────────────
    best_params = study.best_trial.params
    best_cv_auc = study.best_trial.value

    logger.info(f"\n{'═'*60}")
    logger.info(f"Best trial #{study.best_trial.number}")
    logger.info(f"  CV AUC-ROC : {best_cv_auc:.4f}")
    logger.info(f"  Params     : {best_params}")
    logger.info(f"{'═'*60}")

    # ── 5. Retrain best model on full train set & evaluate on test ────────────
    best_model = GradientBoostingClassifier(
        **best_params,
        random_state=42,
    )
    best_model.fit(X_train, y_train)

    y_prob = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    logger.info(f"  Test AUC-ROC: {test_auc:.4f}")

    # ── 6. Generate parameter importance plot ─────────────────────────────────
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    importance_path = reports_dir / "optuna_param_importance.png"

    try:
        importances = optuna.importance.get_param_importances(study)

        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(importances.keys())
        values = list(importances.values())

        bars = ax.barh(names, values, color="#4C72B0", edgecolor="white")
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title("Hyperparameter Importance (Optuna fANOVA)", fontsize=14)
        ax.invert_yaxis()

        # Add value labels on each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10,
            )

        plt.tight_layout()
        fig.savefig(importance_path, dpi=150)
        plt.close(fig)
        logger.info(f"Parameter importance plot saved → {importance_path}")

    except Exception as e:
        logger.warning(f"Could not generate importance plot: {e}")
        importance_path = None

    # ── 7. Log everything to MLflow ───────────────────────────────────────────
    with mlflow.start_run(run_name="optuna_best_gb"):

        # Log the best hyperparameters
        mlflow.log_params(best_params)

        # Log metrics
        mlflow.log_metric("best_cv_auc", best_cv_auc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("n_trials", n_trials)

        # Log the importance plot as an artifact
        if importance_path and importance_path.exists():
            mlflow.log_artifact(str(importance_path))

        # Log and register the best model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, "churn_tuned_gb")

        logger.info(f"MLflow run {run_id} — model registered as 'churn_tuned_gb'")

    # ── 8. Print summary ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  OPTUNA TUNING COMPLETE")
    print(f"{'═'*60}")
    print(f"  Trials completed : {len(study.trials)}")
    print(f"  Best CV AUC-ROC  : {best_cv_auc:.4f}")
    print(f"  Test AUC-ROC     : {test_auc:.4f}")
    print(f"  Best params      :")
    for k, v in best_params.items():
        print(f"    {k:20s} = {v}")
    print(f"{'═'*60}")
    print(f"  Model registered as: churn_tuned_gb")
    print(f"  MLflow UI: mlflow ui --backend-store-uri {TRACKING_URI}")
    print(f"{'═'*60}")

    return {
        "best_params": best_params,
        "best_cv_auc": best_cv_auc,
        "test_auc": test_auc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tune_and_log(n_trials=50)
