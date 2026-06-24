"""
Synthetic Telco churn dataset generator.

Produces a realistic dataset modeled after the IBM Telco Customer Churn dataset
with ~7,000 rows and 20 features. Designed so the full pipeline runs
out-of-the-box without downloading external CSVs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from the central config file."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_churn_dataset(n_samples: int = 7043, random_state: int = 42, churn_rate: float = 0.265) -> pd.DataFrame:
    """
    Generate a synthetic Telco-style churn dataset.

    Features mirror the real IBM Telco dataset:
    - Demographics: gender, SeniorCitizen, Partner, Dependents
    - Account: tenure, Contract, PaperlessBilling, PaymentMethod
    - Services: PhoneService, MultipleLines, InternetService,
                OnlineSecurity, OnlineBackup, DeviceProtection,
                TechSupport, StreamingTV, StreamingMovies
    - Charges: MonthlyCharges, TotalCharges
    - Target: Churn (binary)
    """
    rng = np.random.default_rng(random_state)

    # ── Demographics ──────────────────────────────────────────────────────────
    gender = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])

    # ── Account info ──────────────────────────────────────────────────────────
    tenure = rng.integers(0, 73, size=n_samples)  # 0-72 months
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.55, 0.21, 0.24],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # ── Services ──────────────────────────────────────────────────────────────
    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines_raw = rng.choice(["Yes", "No", "No phone service"], size=n_samples, p=[0.42, 0.48, 0.10])
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_samples,
        p=[0.34, 0.44, 0.22],
    )

    def service_col(internet_service_arr: np.ndarray, yes_prob: float = 0.40) -> list[str]:
        """Generate a service column conditioned on having internet."""
        result = []
        for inet in internet_service_arr:
            if inet == "No":
                result.append("No internet service")
            else:
                result.append(rng.choice(["Yes", "No"], p=[yes_prob, 1 - yes_prob]))
        return result

    online_security = service_col(internet_service, 0.29)
    online_backup = service_col(internet_service, 0.34)
    device_protection = service_col(internet_service, 0.34)
    tech_support = service_col(internet_service, 0.29)
    streaming_tv = service_col(internet_service, 0.38)
    streaming_movies = service_col(internet_service, 0.39)

    # ── Charges ───────────────────────────────────────────────────────────────
    base_charge = 20.0
    monthly_charges = base_charge + rng.uniform(0, 80, size=n_samples)

    # Fiber optic customers pay more
    for i in range(n_samples):
        if internet_service[i] == "Fiber optic":
            monthly_charges[i] += rng.uniform(10, 30)
        elif internet_service[i] == "No":
            monthly_charges[i] = rng.uniform(18, 30)

    monthly_charges = np.round(monthly_charges, 2)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 50, size=n_samples), 2)
    total_charges = np.maximum(total_charges, 0)  # no negatives

    # ── Churn (correlated with features) ──────────────────────────────────────
    # Build a logistic model for realistic churn correlation
    churn_logit = (
        -1.5
        + 0.5 * (np.array(contract) == "Month-to-month").astype(float)
        - 0.8 * (np.array(contract) == "Two year").astype(float)
        + 0.4 * (np.array(internet_service) == "Fiber optic").astype(float)
        - 0.3 * np.array([1 if s == "Yes" else 0 for s in online_security])
        - 0.3 * np.array([1 if s == "Yes" else 0 for s in tech_support])
        + 0.3 * (np.array(payment_method) == "Electronic check").astype(float)
        - 0.02 * tenure
        + 0.3 * senior_citizen
        + 0.005 * monthly_charges
        + rng.normal(0, 0.3, size=n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))

    # Calibrate threshold to hit desired churn rate
    threshold = np.quantile(churn_prob, 1 - churn_rate)
    churn = (churn_prob >= threshold).astype(int)

    logger.info(f"Generated churn rate: {churn.mean():.3f} (target: {churn_rate:.3f})")

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "customerID": [f"C{str(i).zfill(5)}" for i in range(1, n_samples + 1)],
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines_raw,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })

    return df


def main() -> None:
    """Generate synthetic dataset and save to data/raw/churn.csv."""
    params = load_params()
    gen_cfg = params["generate"]

    df = generate_churn_dataset(
        n_samples=gen_cfg["n_samples"],
        random_state=gen_cfg["random_state"],
        churn_rate=gen_cfg["churn_rate"],
    )

    output_path = Path(params["data"]["raw_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(df)} rows to {output_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Churn distribution:\n{df['Churn'].value_counts()}")


if __name__ == "__main__":
    main()
