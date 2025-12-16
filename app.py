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
        <h1> FinRisk-ML</h1>
        <p style='font-size:16px; color:gray;'>
            A Machine Learning-powered Automated Intelligent System that Identify and Classify Fraudulent Credit Card Transactions 💳
        </p>
        <p style='font-size:14px; color:#2ECC71;'>
            ✅ Secure • ✅ Private • ✅ No data is stored
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- FRIENDLY PCA FEATURE NAMES ----------------
friendly_feature_names = {
    "V1": "Pattern 1 (unusual spending behaviour)",
    "V2": "Pattern 2 (irregular transaction rhythm)",
    "V3": "Pattern 3 (sudden deviation from normal behaviour)",
    "V4": "Pattern 4 (rare deviation in spending flow)",
    "V5": "Pattern 5 (anomalous usage pattern)",
    "V6": "Pattern 6 (weak anomaly indicator)",
    "V7": "Pattern 7 (moderate behavioural deviation)",
    "V8": "Pattern 8 (suspicious transaction style)",
    "V9": "Pattern 9 (irregular customer activity)",
    "V10": "Pattern 10 (atypical spending signal)",
    "V11": "Pattern 11 (behavioural fluctuation)",
    "V12": "Pattern 12 (change in spending balance)",
    "V13": "Pattern 13 (unusual feature blend)",
    "V14": "Strong anomaly pattern (major deviation)",
    "V15": "Pattern 15 (sudden behavioural shift)",
    "V16": "Pattern 16 (distorted transaction pattern)",
    "V17": "Pattern 17 (weak fraud signal)",
    "V18": "Pattern 18 (rare anomaly)",
    "V19": "Pattern 19 (latent abnormality)",
    "V20": "Pattern 20 (small behaviour change)",
    "V21": "Pattern 21 (hidden unusual pattern)",
    "V22": "Pattern 22 (subtle anomaly)",
    "V23": "Pattern 23 (weak behaviour deviation)",
    "V24": "Pattern 24 (slight spending anomaly)",
    "V25": "Pattern 25 (light irregularity)",
    "V26": "Pattern 26 (rare behaviour noise)",
    "V27": "Pattern 27 (low-level anomaly)",
    "V28": "Pattern 28 (minor unusual pattern)"
}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model_deploy.pkl")

model = load_model()

# ---------------- FILE UPLOAD ----------------
st.markdown("### 📂 Upload Transaction File")
uploaded_file = st.file_uploader("Upload CSV file containing transactions", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.info("⏳ The system is analyzing transactions. Please wait…")


    progress = st.progress(0)
    status = st.empty()

    with st.spinner("🤖 Running fraud detection model..."):
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
        df["Prediction_Label"] = df["Fraud_Prediction"].map(
            {1: "Fraudulent", 0: "Non-Fraudulent"}
        )
        progress.progress(85)

        def risk_level(prob):
            if prob < 0.3:
                return "Low"
            elif prob < 0.7:
                return "Medium"
            else:
                return "High"

        status.text("⚠️ Step 4/4: Assessing transaction risk levels...")
        df["Risk_Level"] = df["Fraud_Probability"].apply(risk_level)
        progress.progress(100)

    status.empty()
    progress.empty()
    st.success("✅ Analysis completed successfully! Review the results below.")

    # ---------------- METRICS ----------------
    total_tx = len(df)
    fraud_tx = df["Fraud_Prediction"].sum()
    fraud_rate = (fraud_tx / total_tx) * 100
    high_risk_tx = len(df[df["Risk_Level"] == "High"])

    st.markdown("## 📊 Fraud Detection Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", total_tx)
    col2.metric("Detected Frauds", fraud_tx)
    col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}%")
    col4.metric("High-Risk Transactions", high_risk_tx)

    if high_risk_tx > 0:
        st.warning(f"🚨 {high_risk_tx} high-risk transactions detected.")
    else:
        st.success("✅ No high-risk transactions detected.")

    # ---------------- DISTRIBUTION ----------------
    st.markdown("---")
    st.markdown("### ⚠️ Fraud Probability Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Fraud_Probability"], bins=30, kde=True, color="orange", ax=ax)
    ax.set_xlabel("Fraud Probability")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # ---------------- TABLE ----------------
    st.markdown("---")
    st.markdown("### 🔥 Identified Transaction Risk ")

    display_df = df[df["Risk_Level"].isin(["Medium", "High"])].copy()

    rename_map = {k: v for k, v in friendly_feature_names.items() if k in display_df.columns}
    display_df.rename(columns=rename_map, inplace=True)

    display_cols = ["Prediction_Label", "Fraud_Probability", "Risk_Level"] + list(rename_map.values())
    display_df = display_df[display_cols]

    styled_df = display_df.style.applymap(
        lambda v: "background-color:#E74C3C;color:white;font-weight:bold;"
        if v == "High"
        else "background-color:#F39C12;color:black;font-weight:bold;"
        if v == "Medium"
        else "",
        subset=["Risk_Level"]
    )

    st.dataframe(styled_df, height=450)

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
