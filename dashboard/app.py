"""
Streamlit dashboard for churn prediction MLOps pipeline.

Pages:
1. 🏠 Overview — model performance metrics
2. 🔍 Predict — single-customer prediction form
3. 📊 Explainability — SHAP feature importance
4. 📈 Monitoring — drift detection status
5. 📋 Retrain History — log of retrain events
"""

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_params() -> dict:
    """Load params.yaml from project root."""
    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn MLOps Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "🔍 Predict", "📊 Explainability", "📈 Monitoring", "📋 Retrain History"],
)

params = load_params()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-header">Churn Prediction MLOps Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load metrics
    metrics_path = PROJECT_ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        st.subheader("📊 Model Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("F1 Score", f"{metrics['f1_score']:.2%}")
        col3.metric("Precision", f"{metrics['precision']:.2%}")
        col4.metric("Recall", f"{metrics['recall']:.2%}")
        col5.metric("AUC-ROC", f"{metrics['roc_auc']:.2%}")

        st.markdown("---")

        # Confusion matrix
        cm_path = PROJECT_ROOT / "reports" / "confusion_matrix.png"
        if cm_path.exists():
            st.subheader("🔢 Confusion Matrix")
            st.image(str(cm_path), width=500)
    else:
        st.warning("⚠️ No metrics found. Run `python src/train.py` to train the model first.")

    # Dataset info
    st.markdown("---")
    st.subheader("📁 Dataset Info")
    processed_path = PROJECT_ROOT / params["data"]["processed_path"]
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Features", f"{df.shape[1] - 1}")
        col3.metric("Churn Rate", f"{df[params['data']['target_column']].mean():.1%}")

        with st.expander("Preview data"):
            st.dataframe(df.head(20), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Predict
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.markdown('<p class="main-header">Customer Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown("---")

    model_path = PROJECT_ROOT / params["model"]["save_dir"] / params["model"]["artifact_name"]

    if not model_path.exists():
        st.error("❌ No trained model found. Run `python src/train.py` first.")
    else:
        import joblib
        model = joblib.load(model_path)

        feature_names_path = PROJECT_ROOT / params["model"]["save_dir"] / "feature_names.json"
        feature_names = None
        if feature_names_path.exists():
            with open(feature_names_path, "r", encoding="utf-8") as f:
                feature_names = json.load(f)

        st.subheader("Enter Customer Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=5.0)
            senior = st.selectbox("Senior Citizen", [0, 1])

        with col2:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])

        with col3:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            partner = st.selectbox("Partner", ["Yes", "No"])

        total_charges = monthly_charges * tenure
        charges_ratio = monthly_charges / total_charges if total_charges > 0 else 0.0
        avg_monthly = total_charges / tenure if tenure > 0 else monthly_charges

        if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
            # Build feature dict with all expected columns
            features = {
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "SeniorCitizen": senior,
                "charges_ratio": charges_ratio,
                "avg_monthly_charges": avg_monthly,
            }

            # Add one-hot encoded features
            features["gender_Male"] = 1  # default
            features["Partner_Yes"] = 1 if partner == "Yes" else 0
            features["Dependents_Yes"] = 0
            features["PhoneService_Yes"] = 1 if phone == "Yes" else 0
            features["Contract_One year"] = 1 if contract == "One year" else 0
            features["Contract_Two year"] = 1 if contract == "Two year" else 0
            features["PaperlessBilling_Yes"] = 1 if paperless == "Yes" else 0
            features["InternetService_Fiber optic"] = 1 if internet == "Fiber optic" else 0
            features["InternetService_No"] = 1 if internet == "No" else 0
            features["PaymentMethod_Electronic check"] = 1 if payment == "Electronic check" else 0
            features["PaymentMethod_Mailed check"] = 1 if payment == "Mailed check" else 0
            features["PaymentMethod_Credit card (automatic)"] = 1 if payment == "Credit card (automatic)" else 0

            # Tenure group
            if tenure <= 12:
                tg = "0-12"
            elif tenure <= 24:
                tg = "13-24"
            elif tenure <= 48:
                tg = "25-48"
            elif tenure <= 60:
                tg = "49-60"
            else:
                tg = "61+"

            features["tenure_group_13-24"] = 1 if tg == "13-24" else 0
            features["tenure_group_25-48"] = 1 if tg == "25-48" else 0
            features["tenure_group_49-60"] = 1 if tg == "49-60" else 0
            features["tenure_group_61+"] = 1 if tg == "61+" else 0

            # Build DataFrame
            df_input = pd.DataFrame([features])

            # Align columns
            if feature_names:
                for col in feature_names:
                    if col not in df_input.columns:
                        df_input[col] = 0
                df_input = df_input[feature_names]

            proba = model.predict_proba(df_input)[0, 1]
            prediction = int(proba >= 0.5)

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Churn Probability", f"{proba:.1%}")
            with col2:
                if prediction == 1:
                    st.error("⚠️ **HIGH RISK** — This customer is likely to churn!")
                else:
                    st.success("✅ **LOW RISK** — This customer is likely to stay.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Explainability
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Explainability":
    st.markdown('<p class="main-header">Model Explainability (SHAP)</p>', unsafe_allow_html=True)
    st.markdown("---")

    reports_dir = PROJECT_ROOT / params["explainability"]["report_dir"]

    # SHAP plots
    summary_path = reports_dir / "shap_summary.png"
    bar_path = reports_dir / "shap_bar.png"
    waterfall_path = reports_dir / "shap_waterfall.png"
    importance_path = reports_dir / "feature_importance.json"

    if bar_path.exists():
        st.subheader("📊 Feature Importance (mean |SHAP|)")
        st.image(str(bar_path), use_container_width=True)
    else:
        st.info("Run `python src/explain.py` to generate SHAP explanations.")

    if summary_path.exists():
        st.subheader("🐝 SHAP Summary (Beeswarm)")
        st.image(str(summary_path), use_container_width=True)

    if waterfall_path.exists():
        st.subheader("💧 Single Instance Explanation (Waterfall)")
        st.image(str(waterfall_path), use_container_width=True)

    if importance_path.exists():
        with open(importance_path, "r", encoding="utf-8") as f:
            importance = json.load(f)

        st.subheader("🏆 Top Features")
        imp_df = pd.DataFrame({
            "Feature": importance["features"][:15],
            "Mean |SHAP|": importance["importance"][:15],
        })
        st.dataframe(imp_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Monitoring
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Monitoring":
    st.markdown('<p class="main-header">Data Drift Monitoring</p>', unsafe_allow_html=True)
    st.markdown("---")

    mon_dir = PROJECT_ROOT / params["monitoring"]["report_dir"]

    # Find latest drift summary
    summaries = sorted(mon_dir.glob("drift_summary_*.json"), reverse=True)

    if summaries:
        with open(summaries[0], "r", encoding="utf-8") as f:
            latest_drift = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset Drift", "⚠️ Yes" if latest_drift["dataset_drift"] else "✅ No")
        col2.metric("Drifted Features", latest_drift["n_drifted_features"])
        col3.metric("Drift Share", f"{latest_drift['share_drifted_features']:.1%}")

        if latest_drift["drifted_features"]:
            st.subheader("Drifted Features")
            st.dataframe(
                pd.DataFrame({"Feature": latest_drift["drifted_features"]}),
                use_container_width=True,
                hide_index=True,
            )

        # Show all available reports
        st.markdown("---")
        st.subheader("📄 Available Reports")
        reports = sorted(mon_dir.glob("drift_report_*.html"), reverse=True)
        for report in reports[:5]:
            st.write(f"- `{report.name}` ({report.stat().st_size / 1024:.0f} KB)")
    else:
        st.info("Run `python src/monitor.py` to generate drift reports.")

    if st.button("🔄 Run Drift Detection Now", type="primary"):
        st.info("Running drift detection... Run `python src/monitor.py` from terminal for full results.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Retrain History
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Retrain History":
    st.markdown('<p class="main-header">Retrain History</p>', unsafe_allow_html=True)
    st.markdown("---")

    db_path = PROJECT_ROOT / params["monitoring"]["retrain_log_db"]

    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            df = pd.read_sql("SELECT * FROM retrain_log ORDER BY timestamp DESC", conn)
            if len(df) > 0:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No retrain events logged yet.")
        except Exception:
            st.info("No retrain events logged yet.")
        finally:
            conn.close()
    else:
        st.info("No retrain database found. Run `python src/retrain_trigger.py` to start tracking.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Churn MLOps v1.0**  \n"
    "Built with ❤️ using  \n"
    "Streamlit • scikit-learn • MLflow  \n"
    "Evidently • SHAP • Optuna"
)
