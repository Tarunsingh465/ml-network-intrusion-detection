import streamlit as st


# ==============================
# APPLY GLOBAL SOC THEME
# ==============================
def apply_theme():

    st.markdown("""
    <style>

    /* ==============================
       SOC BACKGROUND
    ============================== */
    .stApp {
        background: radial-gradient(circle at top, #020617, #020617 40%, #000000);
        color: #e2e8f0;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ==============================
       HIDE DEFAULT NAV
    ============================== */
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* ==============================
       SIDEBAR
    ============================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #020617);
        border-right: 1px solid #1e293b;
    }

    /* ==============================
       NEON TEXT ACCENT
    ============================== */
    h1, h2 {
        color: #38bdf8;
        text-shadow: 0 0 8px rgba(56,189,248,0.4);
    }

    /* ==============================
       GLASS PANELS (SOC CARDS)
    ============================== */
    .soc-card {
        background: rgba(2, 6, 23, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        transition: 0.3s;
    }

    .soc-card:hover {
        border: 1px solid rgba(56,189,248,0.6);
        box-shadow: 0 0 20px rgba(56,189,248,0.3);
    }

    /* ==============================
       BUTTONS (NEON STYLE)
    ============================== */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #38bdf8);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(56,189,248,0.6);
    }

    /* ==============================
       INPUTS
    ============================== */
    div[data-baseweb="select"] > div,
    .stNumberInput input {
        background: #020617 !important;
        border: 1px solid #1e293b !important;
        color: #e2e8f0 !important;
        border-radius: 8px;
    }

    /* ==============================
       DROPDOWN
    ============================== */
    ul[role="listbox"] {
        background-color: #020617 !important;
        color: #e2e8f0 !important;
        border: 1px solid #1e293b;
    }

    /* ==============================
       LABELS
    ============================== */
    label {
        color: #94a3b8 !important;
    }

    /* ==============================
       SLIDER
    ============================== */
    div[data-baseweb="slider"] > div {
        color: #ef4444;
    }

    /* ==============================
       BLINKING ALERT (USE ANYWHERE)
    ============================== */
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }

    .alert-blink {
        animation: blink 1s infinite;
        color: #ef4444;
        font-weight: bold;
    }

    /* ==============================
       TERMINAL STYLE TEXT
    ============================== */
    .terminal {
        font-family: monospace;
        color: #22c55e;
        background: #000;
        padding: 10px;
        border-radius: 6px;
    }

    </style>
    """, unsafe_allow_html=True)


# ==============================
# PAGE-SPECIFIC THEMES
# ==============================
def apply_page_theme(page):

    colors = {
        "dashboard": "#020617",
        "predict": "#020617",
        "visual": "#02111f",
        "history": "#03121a",
        "explain": "#060e1a"
    }

    color = colors.get(page, "#020617")

    st.markdown(f"""
    <style>
    .stApp {{
        background: radial-gradient(circle at top, {color}, #000);
    }}
    </style>
    """, unsafe_allow_html=True)


# ==============================
# RENDER SIDEBAR (SOC STYLE)
# ==============================
def render_sidebar(current_page):

    page_map = {
        "Dashboard": "dashboard",
        "Visualizations": "visual",
        "Predict Input": "predict",
        "History": "history",
        "Explainability": "explain"
    }

    apply_page_theme(page_map.get(current_page, "dashboard"))

    # ==============================
    # SIDEBAR HEADER
    # ==============================
    st.sidebar.markdown("""
    <h2 style="color:#38bdf8;">🛡 IDS Console</h2>
    """, unsafe_allow_html=True)

    st.sidebar.caption("Intrusion Detection System")
    st.sidebar.markdown("---")

    # ==============================
    # NAVIGATION
    # ==============================
    st.sidebar.markdown("### 📂 Modules")

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

    # ==============================
    # THRESHOLD CONTROL
    # ==============================
    st.sidebar.markdown("### ⚙ Detection Sensitivity")

    threshold = st.sidebar.slider(
        "Attack Threshold",
        0.0, 1.0, 0.30, 0.01
    )

    st.sidebar.caption("Lower → More sensitive detection")

    st.sidebar.markdown("---")

    # ==============================
    # SYSTEM STATUS PANEL
    # ==============================
    st.sidebar.markdown("### 🛰 System Status")

    st.sidebar.markdown("""
    <div class="soc-card">
    <b>Model:</b> Random Forest <br>
    <b>Features:</b> 78 <br>
    <b>Mode:</b> Binary Detection <br>
    <b>Status:</b> <span style="color:#22c55e;">ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

    return threshold