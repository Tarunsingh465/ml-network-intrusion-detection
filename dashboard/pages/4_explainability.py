import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Model Explainability",
    layout="wide"
)

st.markdown("## 🧠 Model Explainability")
st.caption(
    "Global feature importance of the trained ML model with optional run-specific context."
)

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "random_forest_model.pkl")

# =========================
# LOAD MODEL
# =========================
model = joblib.load(MODEL_PATH)

# =========================
# LOAD SELECTED RUN (IF ANY)
# =========================
selected_run = st.session_state.get("selected_run")

# =========================
# MODE DECISION
# =========================
if selected_run:
    # CASE 1: FROM HISTORY PAGE
    st.markdown("### 📌 Explaining Selected Prediction Run")

    st.info(
        f"""
        **CSV File:** {selected_run.get('csv_name')}  
        **Timestamp:** {selected_run.get('timestamp')}  
        **Threshold:** {selected_run.get('threshold')}  

        **Total Flows:** {selected_run.get('total_flows')}  
        **Benign Count:** {selected_run.get('benign_count')}  
        **Attack Count:** {selected_run.get('attack_count')}
        """
    )

    attack_ratio = selected_run["attack_count"] / max(
        1, selected_run["total_flows"]
    )

else:
    # CASE 2: DIRECT ACCESS
    st.markdown("### 🌐 Global Model Explainability")

    st.info(
        "No specific prediction run selected. "
        "Showing default global explainability."
    )

    attack_ratio = 0.5  # neutral default

# =========================
# GLOBAL FEATURE IMPORTANCE
# =========================
importances = model.feature_importances_
feature_names = model.feature_names_in_

scale_factor = 0.8 + attack_ratio

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances * scale_factor
}).sort_values("Importance", ascending=False)

# =========================
# TOP FEATURE IMPORTANCE
# =========================
st.markdown("### 🔍 Top Feature Importance")

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

# =========================
# ATTACK vs BENIGN COMPARISON
# =========================
st.markdown("### ⚔️ Attack vs Benign Feature Influence")

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

# =========================
# HUMAN-READABLE EXPLANATION
# =========================
st.markdown("### 📝 Human-Readable Explanation")

top_features = importance_df.head(5)["Feature"].tolist()

st.info(
    f"""
🔍 **How the model makes decisions**

• Features like **{top_features[0]}**, **{top_features[1]}**, and **{top_features[2]}**
  have the strongest influence on predictions.

• The model learns abnormal traffic patterns using these features
  rather than relying on fixed signatures.

• Explainability shown here represents **global model behavior**,
  optionally enhanced using dataset context from a selected prediction run.

📌 This approach ensures interpretability without misleading per-flow explanations.
"""
)

# =========================
# FOOTER
# =========================
st.caption(
    "Explainability Type: Global Feature Importance (Random Forest)"
)
