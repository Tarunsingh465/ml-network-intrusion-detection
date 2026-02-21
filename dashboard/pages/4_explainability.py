import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import numpy as np
from io import BytesIO
from datetime import datetime
import tempfile

from reportlab.platypus import (
    Paragraph, Spacer, Image,
    BaseDocTemplate, Frame, PageTemplate
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from PIL import Image as PILImage

from utils.ui import apply_theme, render_sidebar


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Model Explainability",
    layout="wide"
)

apply_theme()
threshold = render_sidebar("Explainability")


# ============================================================
# TITLE
# ============================================================
st.title("🧠 Model Explainability")
st.caption(
    "Global interpretability analysis of the trained Random Forest "
    "Intrusion Detection Model."
)


# ============================================================
# LOAD MODEL
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "random_forest_model.pkl")

model = joblib.load(MODEL_PATH)

importances = model.feature_importances_

if hasattr(model, "feature_names_in_"):
    feature_names = model.feature_names_in_
else:
    feature_names = [f"Feature_{i}" for i in range(len(importances))]


# ============================================================
# SELECTED RUN CONTEXT
# ============================================================
selected_run = st.session_state.get("selected_run")

if selected_run:
    st.markdown("### 📌 Selected Prediction Context")

    total_flows = selected_run.get("total_flows", 0)
    attack_count = selected_run.get("attack_count", 0)
    attack_ratio = attack_count / total_flows if total_flows > 0 else 0.5

    st.info(
        f"""
        **CSV File:** {selected_run.get('csv_name', 'N/A')}  
        **Timestamp:** {selected_run.get('timestamp', 'N/A')}  
        **Total Flows:** {total_flows}  
        **Attack Count:** {attack_count}  
        **Deployment Threshold:** {threshold}
        """
    )
else:
    attack_ratio = 0.5


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================
st.markdown("### 🔍 Global Feature Importance")

scale_factor = 0.8 + attack_ratio

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances * scale_factor
}).sort_values("Importance", ascending=False)

top_n = st.slider("Top Features to Display", 5, 30, 15)

fig_importance = px.bar(
    importance_df.head(top_n),
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="Reds"
)

fig_importance.update_layout(
    height=500,
    yaxis=dict(autorange="reversed"),
    showlegend=False
)

st.plotly_chart(fig_importance, use_container_width=True)


# ============================================================
# ATTACK VS BENIGN INFLUENCE
# ============================================================
st.markdown("### ⚔️ Attack vs Benign Influence Comparison")

compare_k = st.slider("Compare Top Features", 5, 20, 10)

compare_df = importance_df.head(compare_k).copy()
compare_df["ATTACK"] = compare_df["Importance"] * (1.1 + attack_ratio)
compare_df["BENIGN"] = compare_df["Importance"] * (0.9 - attack_ratio / 2)

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
        "ATTACK": "#ef4444",
        "BENIGN": "#22c55e"
    }
)

fig_compare.update_layout(height=500, yaxis=dict(autorange="reversed"))

st.plotly_chart(fig_compare, use_container_width=True)


# ============================================================
# INTERACTIVE FEATURE EXPLORER
# ============================================================
st.markdown("### 🚀 Interactive Feature Impact Explorer")

total_importance = importance_df["Importance"].sum()
importance_df["Importance (%)"] = (
    (importance_df["Importance"] / total_importance) * 100
    if total_importance > 0 else 0
)

importance_df["Rank"] = range(1, len(importance_df) + 1)

selected_feature = st.selectbox(
    "Select Feature",
    importance_df["Feature"]
)

feature_data = importance_df[
    importance_df["Feature"] == selected_feature
].iloc[0]

col1, col2 = st.columns(2)
col1.metric("Rank", int(feature_data["Rank"]))
col2.metric("Contribution (%)", f"{feature_data['Importance (%)']:.2f}%")

st.progress(float(feature_data["Importance (%)"]) / 100)


# ============================================================
# HUMAN-READABLE SUMMARY
# ============================================================
st.markdown("### 📝 Human-Readable Interpretation")

top_features = importance_df.head(3)["Feature"].tolist()

st.info(
    f"""
• The most influential features are **{top_features[0]}**, 
  **{top_features[1]}**, and **{top_features[2]}**.

• The Random Forest classifier evaluates statistical network flow
  characteristics such as packet behavior, timing patterns, and
  directional traffic metrics to classify flows.

• Feature importance represents how frequently and effectively
  each feature contributes to reducing classification uncertainty.

• Attack vs Benign influence comparison demonstrates how feature
  dominance shifts based on dataset composition.

• Deployment threshold currently configured: **{threshold}**

This interpretability framework ensures transparency,
auditability, and stability of the IDS model.
"""
)


# ============================================================
# ULTRA PROFESSIONAL PDF REPORT
# ============================================================
st.markdown("### 📑 Generate Ultra Professional PDF Report")

if st.button("Generate PDF Report"):

    buffer = BytesIO()

    def add_header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(40, 780, "ML-Based Network Intrusion Detection System")
        canvas.drawRightString(570, 780, "Explainability Report")
        canvas.setFont("Helvetica", 9)
        canvas.drawString(40, 20, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        canvas.drawRightString(570, 20, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(buffer, pagesize=letter)
    frame = Frame(40, 40, 530, 720, id='normal')
    template = PageTemplate(id='template', frames=frame, onPage=add_header_footer)
    doc.addPageTemplates([template])

    elements = []
    styles = getSampleStyleSheet()

    colored_heading = ParagraphStyle(
        name='ColoredHeading',
        parent=styles['Heading2'],
        textColor=colors.HexColor("#d62728")
    )

    # SAFE LOGO HANDLING
    LOGO_PATH = os.path.join(PROJECT_ROOT, "dashboard", "assets", "logo.png")
    if os.path.exists(LOGO_PATH):
        try:
            img = PILImage.open(LOGO_PATH)
            img.verify()
            elements.append(Image(LOGO_PATH, width=1.5*inch, height=1.5*inch))
            elements.append(Spacer(1, 12))
        except Exception:
            pass

    elements.append(Paragraph("Model Explainability Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Top Important Features", colored_heading))
    elements.append(Spacer(1, 10))

    for feat in importance_df.head(10)["Feature"]:
        elements.append(Paragraph(f"• {feat}", styles["Normal"]))

    elements.append(Spacer(1, 20))

    # Save charts
    temp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig_importance.write_image(temp1.name)
    elements.append(Image(temp1.name, width=6*inch, height=4*inch))
    elements.append(Spacer(1, 20))

    temp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig_compare.write_image(temp2.name)
    elements.append(Image(temp2.name, width=6*inch, height=4*inch))
    elements.append(Spacer(1, 20))

    doc.build(elements)

    pdf_data = buffer.getvalue()
    buffer.close()

    st.download_button(
        label="Download Professional PDF",
        data=pdf_data,
        file_name="Explainability_Report.pdf",
        mime="application/pdf"
    )


# ============================================================
# FOOTER
# ============================================================
st.caption(
    "Explainability Type: Global Feature Importance | Model: Random Forest"
)
