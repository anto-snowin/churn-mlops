"""
Churn Prediction API — FastAPI application.

Endpoints:
    GET  /           → root welcome message
    GET  /health     → health check with model status
    POST /predict    → single customer churn prediction
    GET  /analytics  → prediction stats from the last 7 days
    POST /explain    → SHAP feature contributions for a customer
"""

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import CustomerFeatures, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_URI = os.getenv("MODEL_URI", "models:/churn_GradientBoosting/Production")
DB_PATH = Path("predictions.db")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Global state ──────────────────────────────────────────────────────────────
model = None
feature_names: list[str] | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Database helpers
# ═══════════════════════════════════════════════════════════════════════════════

def init_db() -> None:
    """Create the predictions table if it does not exist."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            input_data  TEXT    NOT NULL,
            probability REAL   NOT NULL,
            will_churn  INTEGER NOT NULL,
            confidence  TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Prediction database ready at {DB_PATH}")


def log_prediction(input_data: dict, probability: float, will_churn: bool, confidence: str) -> None:
    """Insert a prediction record into SQLite."""
    import json

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        INSERT INTO predictions (timestamp, input_data, probability, will_churn, confidence)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            json.dumps(input_data),
            probability,
            int(will_churn),
            confidence,
        ),
    )
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
    """
    Load the model from MLflow registry first, fall back to local joblib.

    Priority:
        1. MLflow model URI (MODEL_URI env var)
        2. Local file: model/model.joblib
    """
    global model, feature_names

    # ── Try MLflow first ──────────────────────────────────────────────────────
    try:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info(f"Model loaded from MLflow: {MODEL_URI}")
    except Exception as e:
        logger.warning(f"MLflow load failed ({e}), trying local joblib fallback...")

        # ── Joblib fallback ───────────────────────────────────────────────────
        local_path = PROJECT_ROOT / "model" / "model.joblib"
        if local_path.exists():
            model = joblib.load(local_path)
            logger.info(f"Model loaded from local file: {local_path}")
        else:
            logger.error("No model found! Train a model first with: python src/train.py")
            model = None

    # ── Load feature names if available ───────────────────────────────────────
    fn_path = PROJECT_ROOT / "model" / "feature_names.json"
    if fn_path.exists():
        import json

        with open(fn_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        logger.info(f"Loaded {len(feature_names)} feature names")


# ═══════════════════════════════════════════════════════════════════════════════
# App lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + load model. Shutdown: cleanup."""
    init_db()
    load_model()
    yield


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI application
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Churn Prediction API",
    description="Production-grade REST API for customer churn prediction with MLflow model serving, SHAP explanations, and prediction analytics.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware (allow all origins) ───────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_confidence(probability: float) -> str:
    """
    Map prediction probability to a confidence label.

    'high'   if |prob - 0.5| > 0.3   (prob < 0.2 or prob > 0.8)
    'medium' if |prob - 0.5| > 0.15  (prob < 0.35 or prob > 0.65)
    'low'    otherwise               (prob between 0.35 and 0.65)
    """
    distance = abs(probability - 0.5)
    if distance > 0.3:
        return "high"
    elif distance > 0.15:
        return "medium"
    else:
        return "low"


def build_feature_df(features: CustomerFeatures) -> pd.DataFrame:
    """Convert a CustomerFeatures request into a model-ready DataFrame."""
    data = features.model_dump()
    df = pd.DataFrame([data])

    # Align columns to match training feature order
    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["General"])
async def root():
    """Root endpoint — welcome message."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["General"])
async def health():
    """Health check — returns model status."""
    return {
        "status": "healthy",
        "model": MODEL_URI,
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: CustomerFeatures):
    """
    Predict churn for a single customer.

    Returns the churn probability, boolean will_churn flag, and
    confidence level (high / medium / low).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    df = build_feature_df(features)

    # Predict
    proba = float(model.predict_proba(df)[0, 1])
    will_churn = proba >= 0.5
    confidence = compute_confidence(proba)

    # Log to SQLite
    try:
        log_prediction(features.model_dump(), proba, will_churn, confidence)
    except Exception as e:
        logger.warning(f"Failed to log prediction: {e}")

    return PredictionResponse(
        churn_probability=round(proba, 4),
        will_churn=will_churn,
        confidence=confidence,
    )


@app.get("/analytics", tags=["Analytics"])
async def analytics():
    """
    Prediction analytics for the last 7 days.

    Returns total predictions, average churn probability, and
    counts of high-risk vs low-risk predictions.
    """
    cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        """
        SELECT
            COUNT(*)                          AS total_predictions,
            COALESCE(AVG(probability), 0)     AS avg_churn_probability,
            SUM(CASE WHEN will_churn = 1 THEN 1 ELSE 0 END) AS high_risk_count,
            SUM(CASE WHEN will_churn = 0 THEN 1 ELSE 0 END) AS low_risk_count
        FROM predictions
        WHERE timestamp >= ?
        """,
        (cutoff,),
    ).fetchone()

    conn.close()

    return {
        "period": "last_7_days",
        "total_predictions": row["total_predictions"],
        "avg_churn_probability": round(row["avg_churn_probability"], 4),
        "high_risk_count": row["high_risk_count"] or 0,
        "low_risk_count": row["low_risk_count"] or 0,
    }


@app.post("/explain", tags=["Explainability"])
async def explain(features: CustomerFeatures):
    """
    SHAP explanation for a single customer prediction.

    Uses TreeExplainer for tree-based models (GradientBoosting, RandomForest).
    Returns per-feature SHAP contributions sorted by absolute importance.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    try:
        import shap
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="SHAP not installed. Run: pip install shap",
        )

    df = build_feature_df(features)

    # Create explainer
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="SHAP TreeExplainer only works with tree-based models. "
                   "Current model type may not be supported.",
        )

    shap_values = explainer.shap_values(df)

    # TreeExplainer output shape varies by model type:
    #   - GradientBoosting: shap_values is 2D (n_samples, n_features), expected_value is scalar
    #   - RandomForest:     shap_values is list [class_0, class_1], expected_value is array [ev0, ev1]
    if isinstance(shap_values, list):
        # RandomForest style — take class-1 (churn)
        contributions = shap_values[1][0]
    elif shap_values.ndim == 3:
        # 3D array (n_samples, n_features, n_classes)
        contributions = shap_values[0, :, 1]
    else:
        # GradientBoosting style — 2D array, already for the positive class
        contributions = shap_values[0]

    # Build feature -> contribution mapping
    cols = list(df.columns)
    feature_contributions = {
        col: round(float(val), 4)
        for col, val in zip(cols, contributions)
    }

    # Sort by absolute value (most important first)
    sorted_contributions = dict(
        sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    # Get prediction for context
    proba = float(model.predict_proba(df)[0, 1])

    # Handle expected_value — can be scalar or array
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
        base_value = float(ev[1])
    elif isinstance(ev, np.ndarray):
        base_value = float(ev[0])
    else:
        base_value = float(ev)

    return {
        "churn_probability": round(proba, 4),
        "base_value": round(base_value, 4),
        "feature_contributions": sorted_contributions,
    }
