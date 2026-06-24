"""
Streamlit Dashboard — Churn Prediction Interactive Demo.

Connects to the FastAPI backend (/predict + /explain) and renders:
    - Customer input form (sidebar)
    - Churn probability metric cards
    - Plotly gauge chart
    - SHAP feature contribution bar chart
    - Model info expander

Usage:
    streamlit run dashboard/app.py
"""

import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

# Replace with your live Render URL after deployment
API_URL = "https://churn-mlops-v2i2.onrender.com"

# For local development, uncomment this line:
# API_URL = "http://127.0.0.1:8000"

# ── Color palette ─────────────────────────────────────────────────────────────
HIGH_RISK_RED = "#E24B4A"
LOW_RISK_GREEN = "#1D9E75"
MEDIUM_AMBER = "#FAEEDA"

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    .subtitle {
        color: #8892a0;
        font-size: 0.95rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    .metric-card {
        background: #f8f9fb;
        border-radius: 14px;
        padding: 22px 20px;
        text-align: center;
        border: 1px solid #e9ecef;
        transition: transform 0.15s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }

    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #8892a0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
    }

    .risk-high {
        color: #E24B4A;
    }

    .risk-low {
        color: #1D9E75;
    }

    .risk-medium {
        color: #D4930D;
    }

    .divider {
        border: none;
        border-top: 1px solid #e9ecef;
        margin: 30px 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fb 0%, #eef0f4 100%);
    }

    div[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 0;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — Customer Input Form
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("Customer Profile")
st.sidebar.caption("Enter customer details to predict churn risk")

st.sidebar.markdown("---")
st.sidebar.subheader("Billing & Tenure")

tenure = st.sidebar.slider(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=24,
    help="How many months the customer has been with the company",
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)",
    min_value=18.0,
    max_value=120.0,
    value=65.0,
    step=0.5,
    format="$%.1f",
)

# Auto-calculate TotalCharges
total_charges = monthly_charges * tenure
st.sidebar.metric("Total Charges (auto)", f"${total_charges:,.2f}")

st.sidebar.markdown("---")
st.sidebar.subheader("Service Details")

# ── Contract ──────────────────────────────────────────────────────────────────
contract_options = ["Month-to-month", "One year", "Two year"]
contract_label = st.sidebar.selectbox("Contract", contract_options, index=0)
contract_value = contract_options.index(contract_label)

# ── Internet Service ──────────────────────────────────────────────────────────
internet_options = ["DSL", "Fiber optic", "No internet"]
internet_label = st.sidebar.selectbox("Internet Service", internet_options, index=0)
internet_value = internet_options.index(internet_label)

# ── Online Security ───────────────────────────────────────────────────────────
security_options = ["No", "Yes", "No internet service"]
security_label = st.sidebar.selectbox("Online Security", security_options, index=0)
security_value = security_options.index(security_label)

# ── Tech Support ──────────────────────────────────────────────────────────────
tech_options = ["No", "Yes", "No internet service"]
tech_label = st.sidebar.selectbox("Tech Support", tech_options, index=0)
tech_value = tech_options.index(tech_label)

# ── Payment Method ────────────────────────────────────────────────────────────
payment_options = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]
payment_label = st.sidebar.selectbox("Payment Method", payment_options, index=0)
payment_value = payment_options.index(payment_label)

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button(
    "Predict Churn Risk",
    type="primary",
    use_container_width=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Area — Header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-title">Churn Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">'
    'Powered by GradientBoosting + SHAP | '
    'Real-time predictions via FastAPI'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction logic
# ═══════════════════════════════════════════════════════════════════════════════

if predict_clicked:
    # Build the request payload
    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract_value,
        "InternetService": internet_value,
        "OnlineSecurity": security_value,
        "TechSupport": tech_value,
        "PaymentMethod": payment_value,
    }

    try:
        # ── Call /predict ─────────────────────────────────────────────────────
        pred_response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        pred_response.raise_for_status()
        pred_data = pred_response.json()

        probability = pred_data["churn_probability"]
        will_churn = pred_data["will_churn"]
        confidence = pred_data["confidence"]

        # ── Call /explain ─────────────────────────────────────────────────────
        explain_data = None
        try:
            explain_response = requests.post(f"{API_URL}/explain", json=payload, timeout=30)
            explain_response.raise_for_status()
            explain_data = explain_response.json()
        except Exception:
            pass  # SHAP explanation is optional — don't block the main prediction

        # ── Determine styling ─────────────────────────────────────────────────
        if will_churn:
            risk_icon = "🔴"
            risk_label = "HIGH RISK"
            risk_class = "risk-high"
            gauge_color = HIGH_RISK_RED
        else:
            risk_icon = "🟢"
            risk_label = "LOW RISK"
            risk_class = "risk-low"
            gauge_color = LOW_RISK_GREEN

        confidence_class = {
            "high": "risk-high" if will_churn else "risk-low",
            "medium": "risk-medium",
            "low": "risk-medium",
        }.get(confidence, "risk-medium")

        # ═════════════════════════════════════════════════════════════════════
        # Metric Cards — 3 columns
        # ═════════════════════════════════════════════════════════════════════

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Churn Probability</div>
                <div class="metric-value {risk_class}">{probability:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Status</div>
                <div class="metric-value {risk_class}">{risk_icon} {risk_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value {confidence_class}">{confidence.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ═════════════════════════════════════════════════════════════════════
        # Charts — 2 columns
        # ═════════════════════════════════════════════════════════════════════

        chart_col1, chart_col2 = st.columns([1, 1])

        # ── Plotly Gauge Chart ────────────────────────────────────────────────
        with chart_col1:
            st.subheader("Churn Probability Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 2, "dtick": 20},
                    "bar": {"color": gauge_color, "thickness": 0.75},
                    "bgcolor": "#f0f0f0",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#e8f5e9"},
                        {"range": [30, 60], "color": MEDIUM_AMBER},
                        {"range": [60, 100], "color": "#ffebee"},
                    ],
                    "threshold": {
                        "line": {"color": "#333", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=300,
                margin=dict(l=30, r=30, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── SHAP Waterfall Bar Chart ──────────────────────────────────────────
        with chart_col2:
            st.subheader("Feature Contributions (SHAP)")
            if explain_data and "feature_contributions" in explain_data:
                contributions = explain_data["feature_contributions"]

                # Take top 10 features by absolute contribution
                sorted_items = sorted(
                    contributions.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:10]

                # Reverse for horizontal bar chart (top feature at top)
                features = [item[0] for item in reversed(sorted_items)]
                values = [item[1] for item in reversed(sorted_items)]
                colors = [HIGH_RISK_RED if v > 0 else LOW_RISK_GREEN for v in values]

                fig_shap = go.Figure(go.Bar(
                    x=values,
                    y=features,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.3f}" for v in values],
                    textposition="outside",
                    textfont={"size": 11},
                ))
                fig_shap.update_layout(
                    height=300,
                    margin=dict(l=10, r=60, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="SHAP Value (impact on churn prediction)",
                    font={"family": "Inter", "size": 12},
                    yaxis={"tickfont": {"size": 11}},
                )
                fig_shap.add_vline(x=0, line_width=1, line_color="#ccc")
                st.plotly_chart(fig_shap, use_container_width=True)

                st.caption(
                    "**Red** bars push toward churn | "
                    "**Green** bars push away from churn"
                )
            else:
                st.info(
                    "SHAP explanations unavailable. "
                    "Ensure `shap` is installed on the API server."
                )

        # ═════════════════════════════════════════════════════════════════════
        # Input summary
        # ═════════════════════════════════════════════════════════════════════

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        with st.expander("About this model"):
            st.markdown("""
            | Property | Value |
            |----------|-------|
            | **Algorithm** | GradientBoostingClassifier |
            | **Training Data** | IBM Telco Customer Churn (7,043 customers) |
            | **Features** | 19 encoded features (LabelEncoder) |
            | **Best AUC-ROC** | 0.8449 |
            | **Accuracy** | 80.1% |
            | **F1 Score** | 0.5745 |
            | **Explainability** | SHAP TreeExplainer |
            | **Tracking** | MLflow (sqlite backend) |
            | **Data Versioning** | DVC |

            **Confidence levels:**
            - **High**: Model is very sure (probability < 20% or > 80%)
            - **Medium**: Moderately confident (probability 20-35% or 65-80%)
            - **Low**: Uncertain — near the 50% decision boundary (35-65%)
            """)

    except requests.exceptions.ConnectionError:
        st.error(
            "**API not reachable.** Make sure the FastAPI server is running.\n\n"
            f"Expected URL: `{API_URL}`\n\n"
            "Start locally with: `python -m uvicorn api.main:app --port 8000`"
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"**API returned an error:** {e.response.status_code} — {e.response.text}")
    except Exception as e:
        st.error(f"**Error:** {e}")

else:
    # ── Default state (before button click) ───────────────────────────────────
    st.info(
        "👈 **Fill in the customer profile on the left and click "
        "'Predict Churn Risk'** to see the prediction, gauge chart, "
        "and SHAP feature explanations."
    )

    # Show a placeholder with sample layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Churn Probability</div>
            <div class="metric-value" style="color: #ccc;">—</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Risk Status</div>
            <div class="metric-value" style="color: #ccc;">—</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value" style="color: #ccc;">—</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: #8892a0; font-size: 0.8rem;'>"
    "Churn MLOps v1.0<br>"
    "Built with Streamlit + FastAPI<br>"
    "scikit-learn | MLflow | SHAP"
    "</div>",
    unsafe_allow_html=True,
)
