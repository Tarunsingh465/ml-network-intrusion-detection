import streamlit as st
import plotly.express as px
import requests
import pandas as pd


# PAGE CONFIG
st.set_page_config(
    page_title="Single Flow Prediction",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>
.card {
    background:#0f172a;
    padding:20px;
    border-radius:12px;
    border:1px solid #1e293b;
    transition:0.3s;
}
.card:hover {
    box-shadow:0 0 15px rgba(56,189,248,0.3);
}
.stButton>button {
    background:#2563eb;
    color:white;
    font-weight:600;
    height:3em;
    border-radius:8px;
}
.stButton>button:hover {
    background:#1e40af;
    transform:scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown("## 🧪 Single Flow Prediction")
st.info(
    "This page is for **single network flow testing only**. "
    "For bulk CSV analysis, use the **Visualization page**."
)

# INPUT FEATURES
st.markdown("### 🔧 Configure Network Flow")

c1, c2 = st.columns(2)

with c1:
    st.markdown("#### 🟦 Traffic Volume Features")
    flow_duration = st.number_input("Flow Duration", value=1000.0)
    fwd_packets = st.number_input("Total Forward Packets", value=10.0)
    bwd_packets = st.number_input("Total Backward Packets", value=5.0)
    flow_bytes = st.number_input("Flow Bytes/s", value=500.0)

with c2:
    st.markdown("#### 🟧 Packet & Activity Features")
    pkt_len_mean = st.number_input("Packet Length Mean", value=50.0)
    pkt_len_std = st.number_input("Packet Length Std", value=10.0)
    active_mean = st.number_input("Active Mean", value=100.0)
    idle_mean = st.number_input("Idle Mean", value=200.0)


# BUILD FEATURE VECTOR (78)
features = [0] * 78
features[1]  = flow_duration
features[2]  = fwd_packets
features[3]  = bwd_packets
features[14] = flow_bytes
features[40] = pkt_len_mean
features[41] = pkt_len_std
features[74] = active_mean
features[77] = idle_mean


# PREDICT
st.markdown("### 🚀 Run Prediction")

if st.button("Predict Network Flow"):
    try:
        res = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"features": features},
            timeout=5
        )

        if res.status_code != 200:
            st.error("Backend error during prediction")
            st.stop()

        result = res.json()

        label = result["label"]
        attack_prob = result["attack_confidence"]
        benign_prob = result["benign_confidence"]

        
        # RESULT SUMMARY 
        risk = (
            "LOW RISK" if attack_prob < 0.2 else
            "MEDIUM RISK" if attack_prob < 0.5 else
            "HIGH RISK"
        )

        color = "#16a34a" if label == "BENIGN" else "#dc2626"

        st.markdown(
            f"""
            <div style="
                background:#020617;
                border-left:6px solid {color};
                padding:20px;
                border-radius:10px;
                color:white;
            ">
                <h2>Prediction: {label}</h2>
                <p><b>Attack Probability:</b> {attack_prob:.2f}</p>
                <p><b>Benign Probability:</b> {benign_prob:.2f}</p>
                <p><b>Risk Level:</b> {risk}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(int(attack_prob * 100))
        st.caption("Attack probability (0% = Safe, 100% = Highly Malicious)")

        
        # VISUALIZATION DATA
        df = pd.DataFrame({
            "Type": ["Benign", "Attack"],
            "Probability": [benign_prob, attack_prob]
        })

        
        # PIE + BAR
        left, right = st.columns(2)

        with left:
            pie = px.pie(
                df,
                names="Type",
                values="Probability",
                color="Type",
                color_discrete_map={
                    "Benign": "green",
                    "Attack": "red"
                },
                hole=0.4
            )
            pie.update_traces(
                hovertemplate="<b>%{label}</b><br>Probability: %{value:.2f}"
            )
            pie.update_layout(title="Traffic Probability Distribution")
            st.plotly_chart(pie, use_container_width=True)

        with right:
            bar = px.bar(
                df,
                x="Type",
                y="Probability",
                color="Type",
                color_discrete_map={
                    "Benign": "green",
                    "Attack": "red"
                },
                text_auto=True
            )
            bar.update_layout(
                title="Attack vs Benign Probability",
                yaxis_title="Probability",
                xaxis_title=""
            )
            st.plotly_chart(bar, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# FOOTER
st.caption(
    "Model: Random Forest | Single Flow Probability-Based Intrusion Detection"
)
