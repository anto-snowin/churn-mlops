<p align="center">
  <h1 align="center">Customer Churn Prediction — MLOps Pipeline</h1>
  <p align="center">
    End-to-end production ML system: data versioning, experiment tracking, model serving, monitoring, CI/CD, and live deployment.
  </p>
</p>

<p align="center">
  <a href="https://github.com/YOUR_USERNAME/churn-mlops/actions/workflows/ci-cd.yml">
    <img src="https://github.com/YOUR_USERNAME/churn-mlops/actions/workflows/ci-cd.yml/badge.svg" alt="CI/CD">
  </a>
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  <a href="https://churn-mlops-v2i2.onrender.com/docs">
    <img src="https://img.shields.io/badge/API-live-1D9E75?logo=render&logoColor=white" alt="Live API">
  </a>
  <img src="https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/DVC-versioned-945DD5?logo=dvc&logoColor=white" alt="DVC">
</p>

---

## What Is This?

A **production-grade MLOps pipeline** that predicts customer churn using the IBM Telco dataset (7,043 customers). Not just a notebook — this is a deployable system with data versioning, experiment tracking, a REST API, SHAP explainability, drift monitoring, Docker orchestration, and CI/CD that auto-deploys on push.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌────────────────┐
│  Raw Data   │────▶│  DVC         │────▶│  Preprocessing    │────▶│  MLflow        │
│  (CSV)      │     │  Versioning  │     │  (LabelEncode,    │     │  Experiments   │
│             │     │              │     │   train/test split)│     │  (3 models)    │
└─────────────┘     └──────────────┘     └───────────────────┘     └───────┬────────┘
                                                                           │
                    ┌──────────────┐     ┌───────────────────┐     ┌───────▼────────┐
                    │  Render      │◀────│  Docker           │◀────│  Model         │
                    │  (Live URL)  │     │  Compose          │     │  Registry      │
                    └──────┬───────┘     └───────────────────┘     └────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼─────┐ ┌───▼────┐ ┌─────▼──────┐
       │  FastAPI    │ │  SHAP  │ │  Streamlit │
       │  /predict   │ │  /explain│ │  Dashboard │
       │  /analytics │ │        │ │            │
       └────────────┘ └────────┘ └────────────┘
              │
       ┌──────▼──────┐     ┌───────────────┐
       │  GitHub     │────▶│  Auto-Deploy  │
       │  Actions    │     │  (Render      │
       │  CI/CD      │     │   webhook)    │
       └─────────────┘     └───────────────┘
```

**Pipeline flow:** `Data → DVC → Preprocess → MLflow → Registry → FastAPI → Docker → CI/CD → Live`

---

## Results

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 79.9% | 0.594 | 0.841 |
| Random Forest | 79.2% | 0.562 | 0.823 |
| **Gradient Boosting** | **80.1%** | **0.575** | **0.845** |
| Optuna-Tuned GB | **80.5%** | **0.583** | **0.851** |

> Best model: **GradientBoostingClassifier** (AUC = 0.845), registered in MLflow Model Registry and served via FastAPI.

---

## Live Demo

| Service | URL |
|---------|-----|
| **API Docs (Swagger)** | [churn-mlops-v2i2.onrender.com/docs](https://churn-mlops-v2i2.onrender.com/docs) |
| **Health Check** | [churn-mlops-v2i2.onrender.com/health](https://churn-mlops-v2i2.onrender.com/health) |
| **Streamlit Dashboard** | [Deploy on Streamlit Cloud](#deploy-streamlit) |

### Test the live API:

```bash
curl -X POST https://churn-mlops-v2i2.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 65.5, "TotalCharges": 786.0, "Contract": 0, "InternetService": 1, "OnlineSecurity": 0, "TechSupport": 0, "PaymentMethod": 2}'
```

Response:
```json
{
  "churn_probability": 0.0461,
  "will_churn": false,
  "confidence": "high"
}
```

> **Note:** Free-tier Render has a ~30s cold start after 15 minutes of inactivity. Subsequent requests are <100ms.

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/churn-mlops.git
cd churn-mlops
docker compose up --build
```

Then open:
- **API:** http://localhost:8000/docs
- **MLflow:** http://localhost:5000

### Without Docker:

```bash
pip install -r requirements.txt
python src/data_preprocessing.py    # Preprocess raw data
python src/train.py                 # Train + log to MLflow
python -m uvicorn api.main:app --port 8000
```

---

## Project Structure

```
churn-mlops/
├── .github/workflows/
│   ├── ci-cd.yml                 # Test → Build → Deploy pipeline
│   └── monitor.yml               # Weekly drift check + auto-retrain
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app (5 endpoints)
│   └── schemas.py                # Pydantic request/response models
├── dashboard/
│   └── app.py                    # Streamlit interactive demo
├── data/
│   ├── raw/                      # Original CSV (DVC-tracked)
│   └── processed/                # Cleaned + encoded CSV
├── model/
│   ├── model.joblib              # Serialized best model
│   └── feature_names.json        # Feature column order
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Load → clean → encode → split
│   ├── train.py                  # Train 3 models + MLflow logging
│   ├── tune.py                   # Optuna hyperparameter tuning
│   ├── predict.py                # Batch prediction CLI
│   ├── explain.py                # SHAP explainability
│   ├── monitor.py                # Evidently drift detection
│   └── retrain_trigger.py        # Auto-retrain on drift
├── tests/
│   ├── test_preprocessing.py     # 3 data pipeline tests
│   └── test_api.py               # 3 API integration tests
├── Dockerfile                    # Python 3.10-slim + layer caching
├── docker-compose.yml            # API + MLflow multi-service
├── dvc.yaml                      # Reproducible pipeline (preprocess → train)
├── params.yaml                   # Model hyperparameters
├── requirements.txt              # All dependencies
└── README.md                     # This file
```

---

## Tech Stack

| Tool | Purpose | Why This Tool |
|------|---------|---------------|
| **scikit-learn** | Model training | Industry standard for tabular ML, fast iteration |
| **MLflow** | Experiment tracking + model registry | Tracks params, metrics, artifacts; enables model versioning |
| **DVC** | Data & pipeline versioning | Git for data — reproducible pipelines, Google Drive remote |
| **FastAPI** | REST API serving | Async, auto-docs (Swagger), Pydantic validation, 10x faster than Flask |
| **SHAP** | Model explainability | TreeExplainer gives per-feature contribution — critical for stakeholder trust |
| **Evidently** | Drift monitoring | Detects data/prediction drift — triggers retraining automatically |
| **Optuna** | Hyperparameter tuning | Bayesian optimization, pruning, 50 trials in minutes |
| **Docker** | Containerization | Identical dev/prod environments, multi-service compose |
| **GitHub Actions** | CI/CD | Auto test → build → deploy on every push to main |
| **Render** | Cloud deployment | Free tier, Docker support, deploy hooks for CI/CD |
| **Streamlit** | Interactive dashboard | Rapid UI prototyping with Plotly charts + SHAP visualizations |
| **SQLite** | Prediction logging | Lightweight, zero-config, tracks every API prediction for analytics |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message + navigation links |
| `GET` | `/health` | Health check + model status |
| `POST` | `/predict` | Single customer churn prediction |
| `GET` | `/analytics` | Last 7 days prediction stats |
| `POST` | `/explain` | SHAP feature contributions |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Interview Questions & Answers

**Q1: Why did you choose GradientBoosting over Random Forest?**
> GradientBoosting achieved 0.845 AUC vs Random Forest's 0.823. GBM builds trees sequentially, correcting errors from previous trees, which gives better performance on tabular data with mixed feature types. I validated this across accuracy, F1, and AUC-ROC — GBM won on all three.

**Q2: How do you handle model drift in production?**
> I use Evidently to generate weekly drift reports comparing current prediction distributions against the training baseline. If the Kolmogorov-Smirnov test detects significant drift (p < 0.05), `retrain_trigger.py` automatically retrains the model, pushes updated data via DVC, and commits the new `dvc.lock`. This runs as a Monday 9 AM UTC cron job in GitHub Actions.

**Q3: Why FastAPI over Flask?**
> Three reasons: (1) FastAPI auto-generates interactive Swagger docs at `/docs` — great for demos and onboarding. (2) Pydantic validation catches bad inputs before they hit the model (returns 422 with detailed error messages). (3) Async support means better performance under concurrent load without threading complexity.

**Q4: How do you ensure reproducibility?**
> DVC tracks the raw data and pipeline stages (`dvc.yaml`). `dvc.lock` records the exact MD5 hash of every input, output, and parameter. MLflow logs every training run's hyperparameters, metrics, and model artifacts. Anyone can clone the repo, run `dvc pull && dvc repro`, and get the exact same model.

**Q5: What happens if the model server crashes?**
> The Docker container has `restart: unless-stopped`, so it auto-restarts. The API has a `/health` endpoint for monitoring. On Render, the service auto-restarts on crash and the CI/CD pipeline can redeploy. The model loads from a local joblib file as fallback if MLflow is unreachable, so the API is always available.

**Q6: How would you improve this if you had more time?**
> Three things: (1) A/B testing framework to compare model versions in production. (2) Feature store (Feast) to standardize feature engineering across training and serving. (3) Kubernetes deployment with horizontal auto-scaling instead of single-container Render, plus Prometheus/Grafana for real-time metrics dashboards.

---

## Resume Bullets

Copy-paste these directly into your resume:

> - **Built an end-to-end MLOps pipeline** for customer churn prediction (7,043 customers, AUC 0.845) using scikit-learn, MLflow experiment tracking, DVC data versioning, and Optuna hyperparameter tuning
>
> - **Deployed a production REST API** with FastAPI serving real-time predictions (<100ms latency), SHAP explainability, and prediction analytics — containerized with Docker and auto-deployed via GitHub Actions CI/CD
>
> - **Implemented automated drift monitoring** using Evidently with weekly scheduled checks and auto-retraining triggers, ensuring model performance doesn't degrade silently in production
>
> - **Designed a full MLOps architecture** including reproducible DVC pipelines, MLflow model registry, Docker Compose multi-service setup, and a Streamlit dashboard for stakeholder demos

---

## License

MIT

---

<p align="center">
  Built with 🔥 as a production MLOps portfolio project
</p>
