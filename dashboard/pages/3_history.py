import streamlit as st
import pandas as pd
import os
import plotly.express as px

from utils.ui import apply_theme, render_sidebar


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Prediction History",
    layout="wide"
)

apply_theme()
threshold = render_sidebar("History")


# ==============================
# TITLE
# ==============================
st.title("📜 Prediction History")
st.caption(
    "Audit trail of all CSV uploads and prediction runs "
    "(used for monitoring, debugging, and reporting)."
)


# ==============================
# LOG FILE PATH
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "..", "..", "logs", "history_logs.csv")


# ==============================
# LOAD HISTORY
# ==============================
if not os.path.exists(LOG_PATH):
    st.warning("No history available yet. Upload a CSV file first.")
    st.stop()

df = pd.read_csv(LOG_PATH)

if df.empty:
    st.warning("History file exists but no records found.")
    st.stop()

df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)


# ==============================
# COLUMN SAFETY CHECK
# ==============================
if "benign_count" not in df.columns and "normal_count" in df.columns:
    df["benign_count"] = df["normal_count"]


# ==============================
# FILTER SECTION
# ==============================
st.markdown("### 🔍 Filter History")

filtered_df = df.copy()

with st.expander("Apply Filters"):
    csv_filter = st.multiselect(
        "Filter by CSV file",
        options=sorted(df["csv_name"].unique())
    )

    if csv_filter:
        filtered_df = filtered_df[
            filtered_df["csv_name"].isin(csv_filter)
        ]


# ==============================
# TOP METRICS
# ==============================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Runs", len(filtered_df))
c2.metric("Total Flows", int(filtered_df["total_flows"].sum()))
c3.metric("Total Attacks", int(filtered_df["attack_count"].sum()))
c4.metric("Total Benign", int(filtered_df["benign_count"].sum()))


# ==============================
# SELECT RUN FOR EXPLAINABILITY
# ==============================
st.markdown("### 🔗 Open Explainability for a Run")

if not filtered_df.empty:

    selected_row = st.selectbox(
        "Select a prediction run",
        options=filtered_df.to_dict("records"),
        format_func=lambda x: f"{x['timestamp']} | {x['csv_name']}"
    )

    if st.button("🧠 View Explainability for this Run"):
        st.session_state["selected_run"] = selected_row
        st.switch_page("pages/4_explainability.py")


# ==============================
# HISTORY TABLE
# ==============================
st.markdown("### 🗂️ History Records")

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=400
)


# ==============================
# VISUAL SUMMARY
# ==============================
st.markdown("### 📊 Visual Summary")

v1, v2 = st.columns(2)


# ---- PIE CHART
with v1:
    st.markdown("#### Attack vs Benign (Overall)")

    pie_df = pd.DataFrame({
        "Type": ["Benign", "Attack"],
        "Count": [
            filtered_df["benign_count"].sum(),
            filtered_df["attack_count"].sum()
        ]
    })

    pie_fig = px.pie(
        pie_df,
        names="Type",
        values="Count",
        color="Type",
        color_discrete_map={
            "Benign": "#22c55e",
            "Attack": "#ef4444"
        },
        hole=0.4
    )

    pie_fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(pie_fig, use_container_width=True)


# ---- BAR CHART
with v2:
    st.markdown("#### Attacks per Run")

    bar_df = filtered_df.copy()
    bar_df["Run Number"] = range(1, len(bar_df) + 1)

    bar_fig = px.bar(
        bar_df,
        x="Run Number",
        y="attack_count",
        color_discrete_sequence=["#ef4444"]
    )

    bar_fig.update_layout(
        height=300,
        xaxis_title="Run Number",
        yaxis_title="Attack Count",
        showlegend=False
    )

    st.plotly_chart(bar_fig, use_container_width=True)

# ==============================
# ADVANCED ANALYTICS
# ==============================
st.markdown("### 🚀 Advanced Run Insights")

analysis_df = filtered_df.copy()

# ------------------------------
# Attack Rate Calculation
# ------------------------------
analysis_df["Attack Rate (%)"] = (
    analysis_df["attack_count"] /
    analysis_df["total_flows"]
) * 100


# ------------------------------
# ATTACK RATE TABLE
# ------------------------------
st.markdown("#### 📊 Attack Rate per Run")

rate_table = analysis_df[[
    "timestamp",
    "csv_name",
    "attack_count",
    "total_flows",
    "Attack Rate (%)"
]].copy()

rate_table["Attack Rate (%)"] = rate_table["Attack Rate (%)"].round(2)

st.dataframe(rate_table, use_container_width=True)


# ------------------------------
# TREND OVER TIME
# ------------------------------
st.markdown("#### 📈 Attack Trend Over Time")

trend_df = analysis_df.copy()
trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"])

trend_fig = px.line(
    trend_df,
    x="timestamp",
    y="Attack Rate (%)",
    markers=True
)

trend_fig.update_layout(
    height=350,
    xaxis_title="Time",
    yaxis_title="Attack Rate (%)"
)

st.plotly_chart(trend_fig, use_container_width=True)


# ------------------------------
# HIGH RISK RUN DETECTOR
# ------------------------------
st.markdown("#### 🚨 High-Risk Runs (Attack Rate > 50%)")

high_risk = analysis_df[
    analysis_df["Attack Rate (%)"] > 50
]

if not high_risk.empty:
    st.error(
        f"{len(high_risk)} high-risk run(s) detected!"
    )
    st.dataframe(
        high_risk[[
            "timestamp",
            "csv_name",
            "Attack Rate (%)"
        ]],
        use_container_width=True
    )
else:
    st.success("No high-risk runs detected.")

# ==============================
# DOWNLOAD
# ==============================
st.markdown("### ⬇️ Download History Logs")

csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download History as CSV",
    data=csv_bytes,
    file_name="prediction_history.csv",
    mime="text/csv"
)


# ==============================
# FOOTER
# ==============================
st.caption(
    "History logs are automatically generated by the backend "
    "during every batch prediction run."
)
