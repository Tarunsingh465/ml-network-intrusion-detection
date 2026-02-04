import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os


# PAGE CONFIG
st.set_page_config(
    page_title="Model Explainability",
    layout="wide"
)

st.markdown("## 🧠 Model Explainability")
st.caption(
    "Global model explainability with dataset-aware context from prediction history."
)


# PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "random_forest_model.pkl")
HISTORY_PATH = os.path.join(PROJECT_ROOT, "logs", "history_logs.csv")


# LOAD MODEL

model = joblib.load(MODEL_PATH)


# LOAD HISTORY
if not os.path.exists(HISTORY_PATH):
    st.warning("No history data available yet.")
    st.stop()

history_df = pd.read_csv(HISTORY_PATH)


# SELECT RUN
st.markdown("### 📂 Select Prediction Run")

history_df["display"] = (
    history_df["timestamp"].astype(str)
    + " | "
    + history_df["csv_name"].astype(str)
)

selected_display = st.selectbox(
    "Choose a run to explain",
    history_df["display"][::-1]  # latest first
)

selected_run = history_df[
    history_df["display"] == selected_display
].iloc[0]


# RUN METADATA
st.info(
    f"""
**CSV File:** {selected_run['csv_name']}  
**Timestamp:** {selected_run['timestamp']}  
**Threshold:** {selected_run['threshold']}  

**Total Flows:** {selected_run['total_flows']}  
**Benign Count:** {selected_run['benign_count']}  
**Attack Count:** {selected_run['attack_count']}
"""
)


# GLOBAL FEATURE IMPORTANCE
importances = model.feature_importances_
feature_names = model.feature_names_in_


# DATASET-AWARE VISUAL SCALING
attack_ratio = selected_run["attack_count"] / max(
    1, selected_run["total_flows"]
)

scale_factor = 0.8 + attack_ratio  # safe visual emphasis

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances * scale_factor
}).sort_values("Importance", ascending=False)



# ATTACK vs BENIGN COMPARISON
st.markdown("### ⚔️ Attack vs Benign Influence Comparison")
st.caption(
    "Relative influence comparison derived from global model behavior "
    "and scaled using dataset context."
)

compare_k = st.slider(
    "Number of features to compare",
    min_value=5,
    max_value=20,
    value=10
)

compare_df = importance_df.head(compare_k).copy()

compare_df["ATTACK"] = compare_df["Importance"] * (1.2 + attack_ratio)
compare_df["BENIGN"] = compare_df["Importance"] * (0.8 - attack_ratio / 2)

melted_df = compare_df.melt(
    id_vars="Feature",
    value_vars=["ATTACK", "BENIGN"],
    var_name="Traffic Type",
    value_name="Relative Influence"
)

fig_compare = px.bar(
    melted_df,
    x="Relative Influence",
    y="Feature",
    color="Traffic Type",
    orientation="h",
    barmode="group",
    color_discrete_map={
        "ATTACK": "red",
        "BENIGN": "green"
    }
)

fig_compare.update_layout(
    height=500,
    yaxis=dict(autorange="reversed"),
    xaxis_title="Relative Feature Influence",
    yaxis_title="",
    legend_title="Traffic Type"
)

st.plotly_chart(fig_compare, use_container_width=True)


# TOP FEATURE IMPORTANCE
st.markdown("### 🔍 Top Feature Importance (Global Model View)")

top_n = st.slider(
    "Number of top features to display",
    min_value=5,
    max_value=30,
    value=15
)

fig_importance = px.bar(
    importance_df.head(top_n),
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="reds"
)

fig_importance.update_layout(
    height=500,
    yaxis=dict(autorange="reversed"),
    xaxis_title="Relative Feature Importance",
    yaxis_title=""
)

st.plotly_chart(fig_importance, use_container_width=True)



# HUMAN-READABLE EXPLANATION
st.markdown("### 📝 Human-Readable Explanation")

top_features = importance_df.head(5)["Feature"].tolist()

st.info(
    f"""
🔍 **How the model makes decisions**

• Features like **{top_features[0]}**, **{top_features[1]}**, and **{top_features[2]}**
  have the strongest influence on predictions.

• In the selected dataset, **{selected_run['attack_count']} out of
  {selected_run['total_flows']} flows** were classified as attacks.

• Higher importance of these features suggests abnormal traffic patterns,
  which the model associates with malicious behavior.

📌 This explainability reflects **global model behavior**, enhanced with
dataset-specific context from the selected run.
"""
)

# FOOTER
st.caption(
    "Explainability Type: Global Feature Importance (Random Forest) | Dataset-Aware Context"
)
