"""
Pydantic v2 request/response schemas for the Churn Prediction API.

Models:
    CustomerFeatures   — validated input for /predict
    PredictionResponse — structured output with churn probability & confidence
"""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """
    Input features for a single customer churn prediction.

    All categorical columns are label-encoded integers matching the
    encoding produced by ``data_preprocessing.encode_features()``.
    """

    tenure: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of months the customer has stayed with the company",
    )
    MonthlyCharges: float = Field(
        ...,
        ge=0,
        description="Monthly charge amount in dollars",
    )
    TotalCharges: float = Field(
        ...,
        ge=0,
        description="Total charges accumulated over tenure",
    )
    Contract: int = Field(
        ...,
        ge=0,
        le=2,
        description="Contract type: 0=Month-to-month, 1=One year, 2=Two year",
    )
    InternetService: int = Field(
        ...,
        ge=0,
        le=2,
        description="Internet service: 0=DSL, 1=Fiber optic, 2=No",
    )
    OnlineSecurity: int = Field(
        ...,
        ge=0,
        le=2,
        description="Online security: 0=No, 1=No internet service, 2=Yes",
    )
    TechSupport: int = Field(
        ...,
        ge=0,
        le=2,
        description="Tech support: 0=No, 1=No internet service, 2=Yes",
    )
    PaymentMethod: int = Field(
        ...,
        ge=0,
        le=3,
        description=(
            "Payment method: 0=Bank transfer, 1=Credit card, "
            "2=Electronic check, 3=Mailed check"
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 12,
                    "MonthlyCharges": 65.5,
                    "TotalCharges": 786.0,
                    "Contract": 0,
                    "InternetService": 1,
                    "OnlineSecurity": 0,
                    "TechSupport": 0,
                    "PaymentMethod": 2,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Structured output for a single churn prediction."""

    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that the customer will churn (0.0 to 1.0)",
    )
    will_churn: bool = Field(
        ...,
        description="True if churn_probability >= 0.5",
    )
    confidence: str = Field(
        ...,
        description="Prediction confidence: 'high', 'medium', or 'low'",
    )
