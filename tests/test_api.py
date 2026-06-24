"""
API integration tests for the Churn Prediction FastAPI application.

Tests:
    - test_health_check         → GET /health returns 200 + status "healthy"
    - test_predict_valid_input  → POST /predict returns 200 + valid probability
    - test_predict_invalid_input → POST /predict returns 422 for bad payload

IMPORTANT: TestClient must be used as a context manager (via fixture) so that
FastAPI's lifespan startup event fires and the model gets loaded. Using
`client = TestClient(app)` at module level skips the lifespan entirely.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ── Make project root importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient using a context manager so the lifespan fires.

    Without `with TestClient(app)`, the startup event never runs →
    model stays None → /predict returns 503 instead of 200.
    """
    with TestClient(app) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════════
# Valid input constant
# ═══════════════════════════════════════════════════════════════════════════════

VALID_CUSTOMER = {
    "tenure": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "Contract": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "TechSupport": 0,
    "PaymentMethod": 2,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_health_check(client):
    """GET /health must return 200 with status 'healthy'."""
    response = client.get("/health")

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}"
    )

    data = response.json()
    assert data["status"] == "healthy", (
        f"Expected status 'healthy', got '{data['status']}'"
    )
    assert "model_loaded" in data, (
        "Response missing 'model_loaded' field"
    )


def test_predict_valid_input(client):
    """POST /predict with valid customer data must return 200 and a probability between 0 and 1."""
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}. Body: {response.text}"
    )

    data = response.json()

    # churn_probability must exist and be between 0 and 1
    assert "churn_probability" in data, "Response missing 'churn_probability'"
    prob = data["churn_probability"]
    assert 0.0 <= prob <= 1.0, (
        f"churn_probability {prob} is not between 0 and 1"
    )

    # will_churn must be a boolean
    assert "will_churn" in data, "Response missing 'will_churn'"
    assert isinstance(data["will_churn"], bool), (
        f"will_churn should be bool, got {type(data['will_churn']).__name__}"
    )

    # confidence must be one of high/medium/low
    assert "confidence" in data, "Response missing 'confidence'"
    assert data["confidence"] in ("high", "medium", "low"), (
        f"confidence '{data['confidence']}' not in ('high', 'medium', 'low')"
    )


def test_predict_invalid_input(client):
    """POST /predict with invalid/missing fields must return 422 Unprocessable Entity."""
    # Completely empty payload
    response = client.post("/predict", json={})
    assert response.status_code == 422, (
        f"Empty payload should return 422, got {response.status_code}"
    )

    # Wrong types (string instead of int)
    bad_payload = {
        "tenure": "not_a_number",
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0,
        "Contract": 0,
        "InternetService": 1,
        "OnlineSecurity": 0,
        "TechSupport": 0,
        "PaymentMethod": 2,
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422, (
        f"Bad type should return 422, got {response.status_code}"
    )

    # Value out of range (tenure > 100)
    out_of_range = VALID_CUSTOMER.copy()
    out_of_range["tenure"] = 999
    response = client.post("/predict", json=out_of_range)
    assert response.status_code == 422, (
        f"Out-of-range value should return 422, got {response.status_code}"
    )

    # Verify the 422 response includes a detail array
    data = response.json()
    assert "detail" in data, "422 response should include 'detail' array"
    assert len(data["detail"]) > 0, "detail array should not be empty"
