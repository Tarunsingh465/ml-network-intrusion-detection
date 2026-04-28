import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import pandas as pd
import joblib
import os
import numpy as np

from utils.ui import apply_theme, render_sidebar


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Predict Network Traffic",
    layout="wide"
)

apply_theme()
threshold = render_sidebar("Predict Input")


# ==============================
# LOAD MODEL + FEATURES
# ==============================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "random_forest_model.pkl")
FEATURE_PATH = os.path.join(PROJECT_ROOT, "model", "feature_columns.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "cicids2017_clean.csv")

model = joblib.load(MODEL_PATH)
all_features = joblib.load(FEATURE_PATH)

importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

top15_dynamic = importance_df.head(15)["Feature"].tolist()


# ==============================
# LOAD STRONGEST ATTACK SAMPLE
# ==============================
df_data = pd.read_csv(DATA_PATH)

attack_rows = df_data[df_data["Label_encoded"] == 1]

attack_features = attack_rows.drop(
    columns=["Label", "Label_encoded"],
    errors="ignore"
)

attack_features = attack_features.replace(
    [np.inf, -np.inf], 0
)
attack_features = attack_features.fillna(0)

# Predict probability for all attack samples
attack_probs = model.predict_proba(attack_features)[:, 1]

# Select most confident attack sample
strongest_index = attack_probs.argmax()

attack_sample = attack_features.iloc[strongest_index]


# ==============================
# FEATURE DESCRIPTIONS
# ==============================
feature_descriptions = {
    "Flow Duration": "Total duration of the network flow.",
    "Total Fwd Packets": "Packets sent forward.",
    "Total Backward Packets": "Packets sent backward.",
    "Flow Bytes/s": "Bytes transferred per second.",
    "Flow Packets/s": "Packets transferred per second."
}


st.title("🧪 Intelligent Configurable Flow Prediction")


# ==============================
# RESET BUTTON
# ==============================
if "selected_features" not in st.session_state:
    st.session_state.selected_features = top15_dynamic.copy()

if st.button("🔄 Reset To Top 15 Features"):
    st.session_state.selected_features = top15_dynamic.copy()


# ==============================
# FEATURE SELECTION
# ==============================
st.markdown("### 🎛 Select 15 Features")

selected_features = []

for i in range(15):

    remaining_options = [
        f for f in all_features
        if f not in selected_features
    ]

    default_value = st.session_state.selected_features[i]

    feature = st.selectbox(
        f"Feature {i+1}",
        options=remaining_options,
        index=remaining_options.index(default_value)
        if default_value in remaining_options else 0,
        key=f"feature_select_{i}"
    )

    importance_score = importance_df[
        importance_df["Feature"] == feature
    ]["Importance"].values[0]

    description = feature_descriptions.get(
        feature,
        "No description available."
    )

    st.caption(
        f"Importance: {importance_score:.4f} | ℹ {description}"
    )

    selected_features.append(feature)

st.session_state.selected_features = selected_features


# ==============================
# INPUT VALUES (Strong Attack Defaults)
# ==============================
st.markdown("---")
st.markdown("### ✏ Enter Feature Values (Strongest Attack Sample Loaded)")

feature_values = {}
cols = st.columns(3)

for idx, feature in enumerate(selected_features):

    default_val = float(attack_sample.get(feature, 0.0))

    with cols[idx % 3]:
        value = st.number_input(
            feature,
            value=default_val,
            step=1.0,
            key=f"value_{feature}"
        )

        feature_values[feature] = value


# ==============================
# BUILD FULL VECTOR
# ==============================
features = [0] * len(all_features)

for i, fname in enumerate(all_features):
    if fname in feature_values:
        features[i] = feature_values[fname]


# ==============================
# PREDICTION
# ==============================
st.markdown("### 🚀 Run Prediction")

if st.button("Predict Network Flow", use_container_width=True):

    try:
        res = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"features": features},
            timeout=5
        )

        result = res.json()

        label = result["label"]
        attack_prob = result["attack_confidence"]
        benign_prob = result["benign_confidence"]

        # Risk classification
        if attack_prob < 0.2:
            risk = "LOW RISK"
            risk_color = "green"
        elif attack_prob < 0.5:
            risk = "MEDIUM RISK"
            risk_color = "orange"
        else:
            risk = "HIGH RISK"
            risk_color = "red"

        st.markdown(
            f"## 🔐 Prediction: {label} | Risk Level: :{risk_color}[{risk}]"
        )

        # ==============================
        # ATTACK SEVERITY METER (LEFT + DESCRIPTION RIGHT)
        # ==============================
        st.markdown("## 🚨 Attack Severity Meter")

        col1, col2 = st.columns([2, 1])

        with col1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=attack_prob * 100,
                title={"text": "Attack Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "red"},
                    "steps": [
                        {"range": [0, 25], "color": "green"},
                        {"range": [25, 50], "color": "orange"},
                        {"range": [50, 100], "color": "red"},
                    ],
                }
            ))

            st.plotly_chart(gauge, use_container_width=True)

        with col2:
            st.markdown("### 📘 What This Shows")
            st.write(
                """
                This meter represents the **probability that the given network flow is malicious**.

                - 🟢 **0–25%** → Likely Safe (Benign Traffic)
                - 🟠 **25–50%** → Suspicious Activity
                - 🔴 **50–100%** → High Confidence Attack

                It helps in quickly assessing **threat severity** in real-time.
                """
            )


        # ==============================
        # PROBABILITY DISTRIBUTION (LEFT + DESCRIPTION RIGHT)
        # ==============================
        st.markdown("## 📊 Probability Distribution")

        col3, col4 = st.columns([2, 1])

        with col3:
            df = pd.DataFrame({
                "Type": ["Benign", "Attack"],
                "Probability": [benign_prob, attack_prob]
            })

            pie = px.pie(
                df,
                names="Type",
                values="Probability",
                color="Type",
                color_discrete_map={
                    "Benign": "#22c55e",
                    "Attack": "#ef4444"
                },
                hole=0.4
            )

            pie.update_layout(showlegend=False)

            st.plotly_chart(pie, use_container_width=True)

        with col4:
            st.markdown("### 📘 What This Shows")
            st.write(
                """
                This chart shows how the model distributes confidence between:

                - 🟢 **Benign Traffic**
                - 🔴 **Attack Traffic**

                Instead of just a label, this helps you understand:
                - Model certainty
                - Confidence balance
                - Risk interpretation

                Useful for **decision-making and monitoring systems**.
                """
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.caption(
    "Dynamic Feature Selection | Strongest Attack Auto-Loaded | Random Forest IDS"
)
