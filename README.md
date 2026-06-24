# Churn MLOps

> End-to-end MLOps pipeline for customer churn prediction — from raw data to a production-ready REST API, tracked with DVC and MLflow, tuned with Optuna, explained with SHAP, monitored by Evidently AI, and deployed via Docker + GitHub Actions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        churn-mlops Pipeline                             │
│                                                                         │
│  Synthetic Data Generation                                              │
│      │                                                                  │
│      ▼                                                                  │
│  data_preprocessing.py ──► data/processed/churn_processed.csv          │
│      │                      (37 features, 7043 rows)                   │
│      ▼                                                                  │
│  train.py ──► MLflow Tracking ──► model/model.joblib                   │
│      │         (params, metrics,                                        │
│      │          confusion matrix)                                       │
│      ▼                                                                  │
│  tune.py (Optuna) ──► Best hyperparameters                             │
│      │                                                                  │
│      ▼                                                                  │
│  predict.py (batch) ──► reports/predictions.csv                        │
│      │                                                                  │
│  explain.py (SHAP) ──► reports/shap_*.png                              │
│      │                                                                  │
│  monitor.py (Evidently) ──► monitoring/drift_report_*.html             │
│      │                                                                  │
│  retrain_trigger.py ──► Auto-retrain on drift                          │
│      │                                                                  │
│      ▼                                                                  │
│  FastAPI /predict ──► Docker Container ──► Render / CI/CD              │
│                                                                         │
│  Streamlit Dashboard (port 8501)                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
churn-mlops/
├── data/
│   ├── raw/                  # synthetic dataset (DVC-tracked)
│   └── processed/            # feature-engineered CSVs
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── generate_data.py      # synthetic Telco churn data generator
│   ├── data_preprocessing.py # cleaning, feature engineering, encoding, scaling
│   ├── train.py              # model training + MLflow logging
│   ├── tune.py               # Optuna hyperparameter optimisation
│   ├── predict.py            # batch & single inference
│   ├── explain.py            # SHAP explainability reports
│   ├── monitor.py            # Evidently data drift detection
│   └── retrain_trigger.py    # automated retrain on drift
├── api/
│   ├── main.py               # FastAPI application (3 endpoints)
│   └── schemas.py            # Pydantic v2 request/response models
├── dashboard/
│   └── app.py                # Streamlit dashboard (5 pages)
├── tests/
│   ├── test_preprocessing.py # 13 unit tests
│   └── test_api.py           # API integration tests
├── model/                    # saved model artifacts
├── reports/                  # metrics, SHAP plots, predictions
├── monitoring/               # drift reports, retrain logs
├── .github/workflows/
│   ├── ci-cd.yml             # CI/CD: lint → test → build → deploy
│   └── monitor.yml           # scheduled drift monitoring
├── Dockerfile                # multi-stage build
├── docker-compose.yml        # API + MLflow + Dashboard
├── dvc.yaml                  # pipeline: generate → preprocess → train → evaluate
├── params.yaml               # centralised config (all hyperparams & paths)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Core ML** | scikit-learn, pandas, numpy |
| **Experiment Tracking** | MLflow |
| **Hyperparameter Tuning** | Optuna |
| **Explainability** | SHAP |
| **Monitoring** | Evidently AI |
| **API** | FastAPI, Pydantic v2, Uvicorn |
| **Dashboard** | Streamlit |
| **Data Versioning** | DVC |
| **Containerisation** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Database** | SQLite (retrain logs, MLflow backend) |
| **Deployment** | Render |

---

## Quick Start

### 1 — Clone & install

```bash
git clone https://github.com/<you>/churn-mlops.git
cd churn-mlops
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Generate synthetic data

```bash
python src/generate_data.py
# Creates data/raw/churn.csv (7,043 rows, 21 columns)
```

### 3 — Preprocess

```bash
python src/data_preprocessing.py
# Creates data/processed/churn_processed.csv (37 features)
# Saves model/scaler.joblib
```

### 4 — Train the model

```bash
python src/train.py
# Logs to MLflow, saves model/model.joblib
# Outputs reports/metrics.json + reports/confusion_matrix.png
```

### 5 — (Optional) Tune hyperparameters

```bash
python src/tune.py
# Runs 50 Optuna trials, logs to MLflow
# Saves reports/tuning_results.json

# To write best params back to params.yaml:
python src/tune.py --write-back
```

### 6 — Run predictions

```bash
python src/predict.py --input data/processed/churn_processed.csv --output reports/predictions.csv
```

### 7 — Generate SHAP explanations

```bash
python src/explain.py
# Outputs reports/shap_summary.png, shap_bar.png, shap_waterfall.png
```

### 8 — Run drift monitoring

```bash
python src/monitor.py
# Generates monitoring/drift_report_*.html + drift_summary_*.json
```

### 9 — Start the API

```bash
uvicorn api.main:app --reload --port 8000
# Interactive docs: http://localhost:8000/docs
```

### 10 — Launch the dashboard

```bash
streamlit run dashboard/app.py
# Opens http://localhost:8501
```

### 11 — Run with Docker Compose

```bash
docker-compose up --build
```

This starts:
- **churn-api** on `http://localhost:8000`
- **mlflow-server** on `http://localhost:5000`
- **dashboard** on `http://localhost:8501`

---

## DVC Pipeline

```bash
dvc repro  # runs all stages in order
```

| Stage | Script | Output |
|-------|--------|--------|
| `generate_data` | `src/generate_data.py` | `data/raw/churn.csv` |
| `preprocess` | `src/data_preprocessing.py` | `data/processed/churn_processed.csv` |
| `train` | `src/train.py` | `model/model.joblib`, `reports/metrics.json` |
| `evaluate` | `src/predict.py` | `reports/predictions.csv` |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness + model-loaded check |
| `POST` | `/predict` | Single-customer churn prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 1000) |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 65.5, "TotalCharges": 786.0, "SeniorCitizen": 0}'
```

**Example response:**

```json
{
  "churn_probability": 0.7123,
  "churn_prediction": 1
}
```

---

## Running Tests

```bash
python -m pytest tests/ -v --cov=src --cov=api --tb=short
```

---

## CI/CD (GitHub Actions)

### `ci-cd.yml` — runs on every push to `main`:
1. **Lint** — `ruff check src/ api/ tests/`
2. **Test** — `pytest tests/`
3. **Build** — Docker image pushed to GHCR
4. **Deploy** — Render deploy hook

### `monitor.yml` — runs every Monday at 08:00 UTC:
1. Runs drift detection
2. Uploads monitoring report
3. Creates a GitHub issue if drift detected

---

## Configuration

All hyperparameters and paths are centralised in `params.yaml`. Key sections:

| Section | Controls |
|---------|----------|
| `data` | Input/output paths, target column |
| `preprocessing` | Missing threshold, test split |
| `training` | Algorithm, hyperparameters |
| `tuning` | Optuna trials, CV folds, scoring |
| `mlflow` | Tracking URI, experiment name |
| `monitoring` | Drift threshold, report directory |
| `explainability` | SHAP display features |
| `api` | Host, port, model path |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend |
| `MODEL_URI` | `model/model.joblib` | Model path |
| `RENDER_DEPLOY_HOOK_URL` | _(secret)_ | Render deploy webhook |

---

## License

MIT © 2024
