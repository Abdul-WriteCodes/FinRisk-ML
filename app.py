# -*- coding: utf-8 -*-
"""
FinRisk-ML | Intelligent Fraud Detection Dashboard
Sidebar-driven controls with immersive analytics view
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="FinRisk-ML | Intelligent Fraud Detection",
    layout="wide",
    page_icon="💳"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <div style='text-align:center;'>
        <h1>💳 FinRisk-ML</h1>
        <p style='font-size:16px; color:gray;'>
            A Machine Learning-powered Automated System for Credit Card Fraud Detection
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

# ---------------- LOAD CSV ----------------
@st.cache_data(show_spinner=False)
def load_csv(file):
    return pd.read_csv(file)


# SIDEBAR — CONTROL PLANE

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")

    uploaded_file = st.file_uploader(
        "📂 Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file:
        df = load_csv(uploaded_file)
        st.session_state["uploaded_df"] = df
        st.success("File loaded successfully")
        st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    analyse_clicked = st.button(
        "🚀 Run Fraud Analysis",
        type="primary",
        use_container_width=True,
        disabled=("uploaded_df" not in st.session_state)
    )

    if st.button("🔄 Reset Application", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    st.caption("• Algorithm: XGBoost")
    st.caption("• Output: Fraud Probability")
    st.caption("• Risk Levels: Low / Medium / High")
    
   # ---------------- Dummy Data Link ----------------
st.markdown("---")
st.markdown("### ℹ️ Get Dummy Data Here")
st.markdown(
    """
    <div style="text-align: center; font-size: 0.85em; color: #8a8a8a; line-height: 1.7em;">
        🔬 Get Dummy Data: 
        <a href="https://app.box.com/s/yih10tldxgnagreecb0hlicnph8d2h6l" target="_blank">
            Download
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

    


# MAIN CANVAS — INSIGHT PLANE

if "uploaded_df" not in st.session_state:
    st.info("👈 Upload a CSV file from the sidebar to begin analysis.")
    st.stop()

df = st.session_state["uploaded_df"]

# ---------------- DATA PREVIEW ----------------
with st.expander("👀 Preview Uploaded Data", expanded=False):
    st.dataframe(df.head(), use_container_width=True)

# ---------------- RUN ANALYSIS ----------------
if analyse_clicked:
    st.markdown("## 🔍 Fraud Analysis Overview")

    progress = st.progress(0)
    status = st.empty()

    with st.spinner("🤖 Running fraud detection engine..."):
        time.sleep(0.4)
        status.text("Step 1/4: Predicting fraud labels")
        df["Fraud_Prediction"] = model.predict(df)
        progress.progress(30)

        time.sleep(0.4)
        status.text("Step 2/4: Calculating fraud probabilities")
        df["Fraud_Probability"] = model.predict_proba(df)[:, 1]
        progress.progress(60)

        status.text("Step 3/4: Assigning prediction labels")
        df["Prediction_Label"] = df["Fraud_Prediction"].map(
            {1: "Fraudulent", 0: "Non-Fraudulent"}
        )
        progress.progress(80)

        def risk_level(prob):
            if prob < 0.3:
                return "Low"
            elif prob < 0.7:
                return "Medium"
            else:
                return "High"

        status.text("Step 4/4: Assessing transaction risk levels")
        df["Risk_Level"] = df["Fraud_Probability"].apply(risk_level)
        progress.progress(100)

    progress.empty()
    status.empty()
    st.success("✅ Fraud analysis completed successfully")

    # ---------------- METRICS ----------------
    st.markdown("### 📊 Key Risk Indicators")

    total_tx = len(df)
    fraud_tx = int(df["Fraud_Prediction"].sum())
    fraud_rate = fraud_tx / total_tx * 100
    high_risk_tx = (df["Risk_Level"] == "High").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", total_tx)
    c2.metric("Detected Frauds", fraud_tx)
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    c4.metric("High-Risk Transactions", high_risk_tx)

    # ---------------- DISTRIBUTION ----------------
    st.markdown("---")
    st.markdown("### ⚠️ Fraud Probability Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df["Fraud_Probability"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Fraud Probability")
    ax.set_ylabel("Transaction Count")
    st.pyplot(fig)

    # ---------------- HIGH-RISK TABLE ----------------
    st.markdown("---")
    st.markdown("### 🔥 Medium & High-Risk Transactions")

    display_df = df[df["Risk_Level"].isin(["Medium", "High"])].copy()

    # Apply friendly PCA names ONLY for display
    rename_map = {
        k: v for k, v in friendly_feature_names.items()
        if k in display_df.columns
    }
    display_df.rename(columns=rename_map, inplace=True)

    display_cols = (
        ["Prediction_Label", "Fraud_Probability", "Risk_Level"]
        + list(rename_map.values())
    )
    display_df = display_df[display_cols]

    styled_df = display_df.style.applymap(
        lambda v: (
            "background-color:#E74C3C;color:white;font-weight:bold;"
            if v == "High"
            else "background-color:#F39C12;color:black;font-weight:bold;"
            if v == "Medium"
            else ""
        ),
        subset=["Risk_Level"]
    )

    st.dataframe(styled_df, height=450, use_container_width=True)

    # ---------------- DOWNLOAD ----------------
    st.download_button(
        "⬇️ Download Risk Transactions as CSV",
        display_df.to_csv(index=False),
        file_name="fraud_risk_transactions.csv",
        mime="text/csv"
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 0.85em; color: #8a8a8a; line-height: 1.7em;">
        <strong>FinRisk-ML</strong> — An Automated Machine Learning System for 
        <strong>FinTech & E-commerce Payment Risk Analysis</strong><br>
        🔬 Learn More About Developer by 
        <a href="https://abdul-writecodes.github.io/portfolio/" target="_blank">
            Abdul
        </a><br>
        <strong>Disclaimer:</strong> This application does not collect, process, or store any personal or financial data.<br>
        © 2025 Abdul Write & Codes. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)