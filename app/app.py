import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📉",
    layout="centered",
)

# ---------------------------------------------------------------
# Load the trained pipeline (cached so it only loads once)
# ---------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR.parent / "Model" / "churn_model.joblib"
@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

artifact = load_artifact()
model = artifact["model"]
threshold = artifact["threshold"]
addon_cols = artifact["addon_cols"]
feature_names = artifact["feature_names"]


@st.cache_resource
def load_explainer(_classifier):
    return shap.TreeExplainer(_classifier)


explainer = load_explainer(model.named_steps["classifier"])


def risk_category(prob: float) -> tuple[str, str]:
    """Bucket a churn probability into a human-readable risk tier."""
    if prob < 0.20:
        return "Low Risk", "🟢"
    elif prob < 0.45:
        return "Medium Risk", "🟡"
    elif prob < 0.70:
        return "High Risk", "🟠"
    else:
        return "Very High Risk", "🔴"

st.title("📉 Telco Customer Churn Predictor")
st.write(
    "Enter a customer's details below to estimate their probability of "
    "churning, using a tuned XGBoost model trained on the Telco Customer "
    "Churn dataset."
)

# ---------------------------------------------------------------
# Input form
# ---------------------------------------------------------------
with st.form("churn_form"):
    st.subheader("Customer profile")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0
        )

    st.subheader("Services")
    col3, col4 = st.columns(2)

    with col3:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox("Online Security", ["No", "Yes"])

    with col4:
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    submitted = st.form_submit_button("Predict churn risk")

# ---------------------------------------------------------------
# Build the row, engineer features exactly like training, predict
# ---------------------------------------------------------------
if submitted:
    row = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
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
    }

    input_df = pd.DataFrame([row])

    # Engineered features, same logic as train_model.py
    input_df["num_addons"] = (input_df[addon_cols] == "Yes").sum(axis=1)
    input_df["charge_per_addon"] = (
        input_df["MonthlyCharges"] / (input_df["num_addons"] + 1)
    ).round(2)

    prob = float(model.predict_proba(input_df)[0, 1])
    pred = int(prob >= threshold)
    risk_label, risk_emoji = risk_category(prob)

    st.divider()
    st.subheader("Result")

    col_a, col_b = st.columns(2)
    col_a.metric("Churn probability", f"{prob:.1%}")
    col_b.metric("Prediction", "⚠️ Will churn" if pred else "✅ Will stay")

    risk_colors = {
        "Low Risk": "#2ca02c",
        "Medium Risk": "#e6b800",
        "High Risk": "#e67e22",
        "Very High Risk": "#d62728",
    }
    st.markdown(
        f"""
        <div style="
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            background-color: {risk_colors[risk_label]}22;
            border: 1px solid {risk_colors[risk_label]};
            font-size: 1.1rem;
            margin: 0.5rem 0 1rem 0;
        ">
            {risk_emoji} <strong>Risk category: {risk_label}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(max(prob, 0.0), 1.0))

    if pred:
        st.warning(
            f"This customer's churn probability ({prob:.1%}) is above the "
            f"decision threshold ({threshold:.0%}). Consider proactive "
            "retention outreach (e.g. contract upgrade, added support)."
        )
    else:
        st.success(
            f"This customer's churn probability ({prob:.1%}) is below the "
            f"decision threshold ({threshold:.0%})."
        )

    # -----------------------------------------------------------
    # SHAP explanation: why did the model predict this?
    # -----------------------------------------------------------
    st.divider()
    st.subheader("Why this prediction? (SHAP explanation)")

    transformed = model.named_steps["preprocessor"].transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class, for older SHAP/XGB combos

    contrib = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0],
    })
    contrib["abs_value"] = contrib["shap_value"].abs()
    top = contrib.sort_values("abs_value", ascending=False).head(10)
    top = top.sort_values("shap_value")  # nicer order for a horizontal bar chart

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in top["shap_value"]]
    ax.barh(top["feature"], top["shap_value"], color=colors)
    ax.set_xlabel("Impact on churn probability (SHAP value)")
    ax.set_title("Top factors driving this prediction")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "🔴 Red bars push the prediction *toward* churn. "
        "🟢 Green bars push it *toward* staying. "
        "Bar length shows how much that feature mattered for this specific customer."
    )

st.divider()
st.caption(
    "Model: tuned XGBoost + SMOTE, trained on the IBM Telco Customer "
    "Churn dataset. Decision threshold set at 0.45 based on precision/"
    "recall tradeoff analysis."
)
