# ═══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Churn Prediction API
# ═══════════════════════════════════════════════════════════════════════════════
#
# Build:  docker build -t churn-api .
# Run:    docker run -p 8000:8000 churn-api
#
# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# ── Install dependencies FIRST (layer caching) ───────────────────────────────
#
# WHY: Docker caches each layer. By copying requirements.txt before the rest of
# the source code, pip install is only re-run when requirements.txt actually
# changes. If we copied everything first, ANY code change (even a one-line
# fix in train.py) would invalidate the cache and force a full reinstall of
# all packages — adding 2-5 minutes to every build.
#
# With this approach:
#   - Edit src/train.py       → only the COPY layers rebuild (~2 seconds)
#   - Edit requirements.txt   → pip install runs again (~2 minutes)
#
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────────────────────
COPY src/ src/
COPY api/ api/
COPY model/ model/
COPY params.yaml .

# ── Environment variables ────────────────────────────────────────────────────
#
# PYTHONPATH: Ensures "from api.schemas import ..." and "from src.data_..."
# resolve correctly. Without this, Python can't find sibling packages and
# you get "ModuleNotFoundError: No module named 'src'" — the #1 Docker error
# for this project.
#
ENV PYTHONPATH=/app
ENV MODEL_URI=models:/churn_GradientBoosting/Production

# ── Expose API port ──────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start the FastAPI server ──────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
