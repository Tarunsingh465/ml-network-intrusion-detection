import streamlit as st
import requests

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ML-Based Network Intrusion Detection System",
    layout="wide"
)

# =========================
# TITLE
# =========================
st.title("🔐 ML-Based Network Intrusion Detection System")

# =========================
# SYSTEM OVERVIEW
# =========================
st.markdown("""
### 🧾 System Overview

This system is a **Machine Learning–based Network Intrusion Detection System (NIDS)**  

It is designed to:
- Detect malicious network traffic
- Distinguish between **BENIGN** and **ATTACK** flows
- Provide real-time predictions via a connected ML backend

This dashboard acts as the **central control panel** for navigating
different modules and testing system health.
""")

st.caption("Dataset: CICIDS 2017 | Model: Random Forest | Author: Tarun Singh")

st.markdown("---")

# =========================
# NAVIGATION MODULES
# =========================
st.subheader("📂 Dashboard Modules")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    if st.button("📊 Traffic Visualizations", use_container_width=True):
        st.switch_page("pages/1_visualizations.py")



with col2:
    if st.button("🧠 Predict Network Traffic", use_container_width=True):
        st.switch_page("pages/2_predict_input.py")


with col3:
    if st.button("📜 Prediction History", use_container_width=True):
        st.switch_page("pages/3_history.py")

with col4:
    if st.button("🔍 Model Explainability", use_container_width=True):
        st.switch_page("pages/4_explainability.py")

st.markdown("---")

# =========================
# SYSTEM TEST BUTTONS
# =========================
st.subheader("🧪 System Tests")

test_col1, test_col2 = st.columns(2)

# 🔌 Backend Connection Test
with test_col1:
    if st.button("🔌 Backend Connection Test", use_container_width=True):
        try:
            res = requests.get("http://127.0.0.1:5000/ping")
            if res.status_code == 200:
                st.success("Backend is reachable and running.")
            else:
                st.error("Backend responded but with an error.")
        except:
            st.error("Backend is not reachable.")

# 🧠 Live ML Prediction Test
with test_col2:
    if st.button("🧠 Live ML Prediction Test", use_container_width=True):
        try:
            sample_data = {"features": [0]*78}
            res = requests.post(
                "http://127.0.0.1:5000/predict",
                json=sample_data
            )
            result = res.json()
            st.success(f"Prediction: {result['label']}")
        except:
            st.error("Prediction failed. Check backend and model.")
