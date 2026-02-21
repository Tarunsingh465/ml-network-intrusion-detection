import streamlit as st


# ==============================
# APPLY GLOBAL THEME
# ==============================
def apply_theme():

    st.markdown("""
    <style>

    /* Hide default Streamlit multipage navigation */
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    /* Main app background */
    .stApp {
        background-color: #0b1220;
        color: white;
    }

    /* Improve button styling */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #334155;
        padding: 0.5rem 1rem;
    }

    /* Improve slider color */
    div[data-baseweb="slider"] > div {
        color: #ef4444;
    }

    </style>
    """, unsafe_allow_html=True)


# ==============================
# RENDER SIDEBAR
# ==============================
def render_sidebar(current_page):

    st.sidebar.markdown("## 🛡 Network Intrusion Detection System")
    st.sidebar.caption("ML-Based IDS | CICIDS 2017")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 📂 Navigation")

    pages = {
        "Dashboard": "dashboard.py",
        "Visualizations": "pages/1_visualizations.py",
        "Predict Input": "pages/2_predict_input.py",
        "History": "pages/3_history.py",
        "Explainability": "pages/4_explainability.py",
    }

    for name, path in pages.items():
        if name == current_page:
            st.sidebar.page_link(path, label=f"👉 {name}")
        else:
            st.sidebar.page_link(path, label=name)

    st.sidebar.markdown("---")

    st.sidebar.markdown("### ⚙ Page Settings")

    threshold = st.sidebar.slider(
        "Attack Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### ℹ System Info")
    st.sidebar.info("""
    Model: Random Forest  
    Features: 78  
    Classification: Binary (Benign vs Attack)  
    Deployment: Flask + Streamlit  
    """)

    return threshold
