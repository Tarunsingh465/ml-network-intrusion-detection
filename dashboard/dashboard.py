import streamlit as st
import requests
import pandas as pd
import os
import time
from utils.ui import apply_theme, render_sidebar

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Network Security Control Center",
    layout="wide"
)

apply_theme()
threshold = render_sidebar("Dashboard")


# ============================================================
# BASIC CLEAN CSS (Subtle Borders + Spacing)
# ============================================================
st.markdown("""
<style>

.section-divider {
    border-top: 1px solid #1f2937;
    margin-top: 30px;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.title("🛡️ Network Security Control Center")
st.caption("Real-Time Intrusion Detection & Monitoring Console")


# ============================================================
# BACKEND HEARTBEAT STATUS
# ============================================================

try:
    res = requests.get("http://127.0.0.1:5000/ping", timeout=2)
    backend_alive = res.status_code == 200
except:
    backend_alive = False


if backend_alive:
    heartbeat_html = """
    <style>
    .heartbeat {
        display: flex;
        align-items: center;
        font-weight: 600;
        color: #22c55e;
    }

    .dot {
        height: 12px;
        width: 12px;
        background-color: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(34,197,94, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(34,197,94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34,197,94, 0); }
    }
    </style>

    <div class="heartbeat">
        <div class="dot"></div>
        Backend API: ONLINE
    </div>
    """
else:
    heartbeat_html = """
    <style>
    .heartbeat-offline {
        display: flex;
        align-items: center;
        font-weight: 600;
        color: #ef4444;
    }

    .dot-offline {
        height: 12px;
        width: 12px;
        background-color: #ef4444;
        border-radius: 50%;
        margin-right: 8px;
    }
    </style>

    <div class="heartbeat-offline">
        <div class="dot-offline"></div>
        Backend API: OFFLINE
    </div>
    """

st.markdown('<div class="metric-box">', unsafe_allow_html=True)
st.markdown(heartbeat_html, unsafe_allow_html=True)

with st.expander("About Backend Connection"):
    st.write(
        "The heartbeat indicator continuously checks whether "
        "the Flask backend server is reachable. "
        "If OFFLINE, prediction services are unavailable."
    )

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# READ LOG STATS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "..", "logs", "history_logs.csv")

total_predictions = 0
total_attacks = 0
attack_rate = 0
risk_level = "LOW"

if os.path.exists(LOG_PATH):
    df_logs = pd.read_csv(LOG_PATH)

    if not df_logs.empty:
        total_predictions = int(df_logs["total_flows"].sum())
        total_attacks = int(df_logs["attack_count"].sum())

        if total_predictions > 0:
            attack_rate = (total_attacks / total_predictions) * 100

        if attack_rate < 5:
            risk_level = "LOW"
        elif attack_rate < 10:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"


# ============================================================
# INTERACTIVE CARD CSS (CLEAN + PROFESSIONAL)
# ============================================================

st.markdown("""
<style>

/* Card Container */
.interactive-card {
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
    background-color: #0b1220;
    transition: all 0.25s ease;
}

/* Hover Effect */
.interactive-card:hover {
    border-color: #2563eb;
    background-color: #111827;
    transform: translateY(-3px);
}

/* Title */
.card-title {
    font-size: 13px;
    color: #94a3b8;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Value */
.card-value {
    font-size: 26px;
    font-weight: 600;
    margin-top: 6px;
    color: #f8fafc;
}

/* Info Text */
.card-info {
    margin-top: 12px;
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.6;
}

/* Optional: Risk Highlight */
.high-risk {
    border-color: #7f1d1d !important;
    background-color: #1f0a0a !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# INTERACTIVE SYSTEM STATUS
# ============================================================

st.subheader("🔎 System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="interactive-card">
        <div class="card-title">Model Type</div>
        <div class="card-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ About Model Type"):
        st.write(
            "Random Forest is an ensemble learning algorithm that "
            "combines multiple decision trees to improve classification "
            "accuracy and reduce overfitting."
        )

with col2:
    st.markdown(f"""
    <div class="interactive-card">
        <div class="card-title">Detection Threshold</div>
        <div class="card-value">{threshold}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ About Detection Threshold"):
        st.write(
            "The detection threshold defines the minimum attack "
            "probability required to classify a network flow as ATTACK. "
            "Lower threshold = more sensitive detection."
        )

with col3:
    st.markdown("""
    <div class="interactive-card">
        <div class="card-title">System Mode</div>
        <div class="card-value">Binary Classification</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ About System Mode"):
        st.write(
            "The IDS operates in binary classification mode, "
            "where traffic is categorized as either BENIGN or ATTACK."
        )


# ============================================================
# INTERACTIVE NETWORK OVERVIEW
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📡 Network Activity Overview")

col4, col5, col6 = st.columns(3)

# ----- Total Predictions Card -----
with col4:
    st.markdown(f"""
    <div class="interactive-card">
        <div class="card-title">Total Predictions</div>
        <div class="card-value">{total_predictions}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ About Total Predictions"):
        st.write(
            "Represents the cumulative number of network flows "
            "processed across all batch prediction runs."
        )


# ----- Attack Rate Card -----
attack_class = ""
if attack_rate > 10:
    attack_class = "high-risk"

with col5:
    st.markdown(f"""
    <div class="interactive-card {attack_class}">
        <div class="card-title">Attack Rate (%)</div>
        <div class="card-value">{attack_rate:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ About Attack Rate"):
        st.write(
            "Attack Rate is calculated as the percentage of "
            "detected malicious flows relative to total processed flows. "
            "Higher values indicate increased suspicious activity."
        )


# ----- Risk Level Card -----
risk_class = ""
if risk_level == "HIGH":
    risk_class = "high-risk"

with col6:
    st.markdown(f"""
    <div class="interactive-card {risk_class}">
        <div class="card-title">Current Risk Level</div>
        <div class="card-value">{risk_level}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ About Risk Level"):
        st.write(
            "Risk Level is derived from the attack rate threshold. "
            "LOW (<5%), MEDIUM (5–10%), HIGH (>10%). "
            "Used to indicate current network threat severity."
        )



# ============================================================
# CRITICAL ALERT (Blinking if HIGH)
# ============================================================

if risk_level == "HIGH":
    st.markdown("""
    <style>
    .critical-alert {
        border: 1px solid #7f1d1d;
        border-radius: 10px;
        padding: 12px;
        margin-top: 10px;
        background-color: #1f0a0a;
        text-align: center;
        animation: blink 1.2s linear infinite;
        color: #ef4444;
        font-weight: 600;
    }

    @keyframes blink {
        50% { opacity: 0.6; }
    }
    </style>

    <div class="critical-alert">
        🚨 CRITICAL ALERT: High Suspicious Network Activity Detected
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# CLEAN ADMINISTRATIVE MODULES
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🧭 Administrative Modules")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)


def clean_module(title, description, page_path):
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(f"{description}")
        st.markdown("")
        if st.button("Open Module", use_container_width=True, key=title):
            st.switch_page(page_path)


with col1:
    clean_module(
        "📊 Traffic Visualization Console",
        "Analyze batch predictions, charts, and flow distribution.",
        "pages/1_visualizations.py"
    )

with col2:
    clean_module(
        "🧪 Manual Flow Testing",
        "Test individual network flows and observe prediction confidence.",
        "pages/2_predict_input.py"
    )

with col3:
    clean_module(
        "📜 Prediction Audit Logs",
        "Review historical batch runs and detection statistics.",
        "pages/3_history.py"
    )

with col4:
    clean_module(
        "🧠 Model Intelligence & Explainability",
        "Explore feature importance and interpretability analysis.",
        "pages/4_explainability.py"
    )



# ============================================================
# LIVE EVENT STREAM
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("🖥️ Live Event Stream")

live_toggle = st.toggle("Enable Live Monitoring")

if live_toggle:
    placeholder = st.empty()

    events = [
        "Incoming TCP traffic spike detected",
        "Suspicious port scanning activity",
        "Normal HTTP traffic observed",
        "Potential brute-force attempt",
        "DNS query anomaly detected"
    ]

    for event in events:
        placeholder.info(f"🔍 {event}")
        time.sleep(1)


# ============================================================
# FOOTER
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.caption(
    "ML-Based Network Intrusion Detection System | "
    "Security Operations Dashboard | Tarun Singh"
)
