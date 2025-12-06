# -*- coding: utf-8 -*-
"""
Optimized Mobile/Desktop Fraud Detection Dashboard
Displays Fraud Label, PCA features (friendly names), Risk Probability & Risk Level
@author: HP
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide",
    page_icon="💳"
)

st.markdown(
    """
    <div style='text-align: center;'>
        <h1>💳 Automated Financial Fraud Detection System</h1>
        <p style='font-size:16px; color:gray;'>
            Real-time Credit Card Transaction Fraud Detection powered by XGBoost
        </p>
        <p style='font-size:14px; color:#2ECC71;'>
            ✅ Secure • ✅ Private • ✅ No data is stored
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- EXPECTED FEATURES ----------------
EXPECTED_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
FRIENDLY_NAMES = {f"V{i}": f"Pattern {i}" for i in range(1, 29)}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model_deploy.pkl")

model = load_model()

# ---------------- FILE UPLOAD ----------------
st.markdown("### 📂 Upload Transaction File")
uploaded_file = st.file_uploader(
    "Upload CSV file containing transactions",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Please upload a CSV file to begin analysis.")
    st.stop()

# ---------------- DATA PREP ----------------
df = pd.read_csv(uploaded_file)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Drop label column if present
if "Class" in df.columns:
    df = df.drop(columns=["Class"])

# Add missing columns with default 0
for col in EXPECTED_COLUMNS:
    if col not in df.columns:
        df[col] = 0.0

# Keep only expected columns and correct order
df = df[EXPECTED_COLUMNS]

# Ensure all columns are numeric
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# ---------------- PREDICTION ----------------
st.info("⏳ The system is analyzing transactions. Please wait…")
progress = st.progress(0)
status = st.empty()

with st.spinner("🤖 Running fraud detection model..."):
    try:
        time.sleep(0.5)
        status.text("🔍 Step 1/4: Predicting fraud labels...")
        df["Fraud_Prediction"] = model.predict(df)
        progress.progress(35)

        time.sleep(0.5)
        status.text("📊 Step 2/4: Calculating fraud probabilities...")
        df["Fraud_Probability"] = model.predict_proba(df)[:, 1]
        progress.progress(65)

        time.sleep(0.5)
        status.text("🏷️ Step 3/4: Assigning fraud labels...")
        df["Prediction_Label"] = df["Fraud_Prediction"].map({1: "Fraudulent", 0: "Non-Fraudulent"})
        progress.progress(85)

        def risk_level(p):
            if p < 0.3:
                return "Low"
            elif p < 0.7:
                return "Medium"
            return "High"

        status.text("⚠️ Step 4/4: Assessing transaction risk levels...")
        df["Risk_Level"] = df["Fraud_Probability"].apply(risk_level)
        progress.progress(100)

    except Exception as e:
        st.error(f"Prediction failed. Ensure the uploaded CSV matches the expected format. Error: {e}")
        st.stop()

status.empty()
progress.empty()
st.success("✅ Analysis completed successfully!")

# ---------------- METRICS ----------------
col1, col2, col3, col4 = st.columns(4)
total_tx = len(df)
fraud_tx = int(df["Fraud_Prediction"].sum())
fraud_rate = fraud_tx / total_tx * 100
high_risk_tx = (df["Risk_Level"] == "High").sum()

col1.metric("Total Transactions", total_tx)
col2.metric("Detected Frauds", fraud_tx)
col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}%")
col4.metric("High-Risk Transactions", high_risk_tx)

# ---------------- FRAUD PROBABILITY DISTRIBUTION ----------------
st.markdown("### ⚠️ Fraud Probability Distribution")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["Fraud_Probability"], bins=30, kde=True, ax=ax, color="orange")
ax.set_xlabel("Fraud Probability")
ax.set_ylabel("Count")
st.pyplot(fig)

# ---------------- HIGH/MEDIUM RISK TABLE ----------------
st.markdown("### 🔥 Medium & High-Risk Transactions")
display_df = df[df["Risk_Level"].isin(["Medium", "High"])].copy()
display_df.rename(columns=FRIENDLY_NAMES, inplace=True)

display_cols = ["Prediction_Label", "Fraud_Probability", "Risk_Level"] + list(FRIENDLY_NAMES.values())
display_df = display_df[[c for c in display_cols if c in display_df.columns]]

st.dataframe(display_df, height=400)

# ---------------- DOWNLOAD ----------------
st.download_button(
    "⬇️ Download Risk Transactions as CSV",
    display_df.to_csv(index=False),
    "fraud_risk_transactions.csv",
    "text/csv"
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; font-size:13px; color:gray;'>
        🔒 This system does not store uploaded data<br>
        © 2025 Automated Fraud Detection System
    </div>
    """,
    unsafe_allow_html=True
)
