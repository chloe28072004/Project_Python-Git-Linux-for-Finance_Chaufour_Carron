"""
Finance Dashboard - Home Page
"""

import streamlit as st

st.set_page_config(
    page_title="Finance Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #0a0a0c 0%, #111114 100%); }
    [data-testid="stHeader"] { background-color: transparent; }
    [data-testid="stSidebar"] { background-color: #111827; display: none; }
    .modebar { display: none !important; }

    /* Hide default navigation */
    [data-testid="stSidebarNav"] { display: none; }

    /* Text colors */
    p, span, label { color: #71717a; }
    h1, h2, h3 { color: #fafafa; }

    /* Hero section */
    .hero-title {
        font-size: 48px;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin: 80px 0 20px 0;
        letter-spacing: -1px;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #71717a;
        text-align: center;
        margin-bottom: 60px;
        line-height: 1.7;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 6px 14px;
        margin-bottom: 24px;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Module cards */
    .module-card {
        background: rgba(24, 24, 27, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 32px 28px;
        border: 1px solid rgba(255,255,255,0.06);
        transition: all 0.2s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        text-decoration: none;
        display: block;
    }

    .module-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }

    .module-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .module-title {
        color: #fafafa;
        font-size: 18px;
        font-weight: 600;
        margin: 0 0 10px 0;
    }

    .module-desc {
        color: #71717a;
        font-size: 14px;
        line-height: 1.6;
        margin: 0 0 20px 0;
    }

    .module-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.1);
        color: #a5b4fc;
        padding: 5px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 8px;
        margin-top: 8px;
    }

    /* Stats */
    .stat-value {
        color: #fafafa;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .stat-label {
        color: #52525b;
        font-size: 13px;
        font-weight: 500;
    }

    /* Buttons - force white text */
    button[kind="primary"],
    button[kind="secondary"],
    .stButton > button,
    .stButton button,
    button {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    button p {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("---")
st.markdown("# Quantitative Analysis Platform")
st.markdown("### Professional-grade backtesting and portfolio optimization")
st.markdown("*Powered by real-time market data from Yahoo Finance*")

# Stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Risk Metrics", "14+")
with col2:
    st.metric("Strategies", "4")
with col3:
    st.metric("Fixed Capital", "$100")

st.markdown("---")

# Modules Section
st.markdown("## Analysis Modules")
st.markdown("*Click on any module to start analyzing*")

# Module Cards
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### Single Asset Analysis")
    st.markdown("Backtest trading strategies on individual securities with advanced risk metrics.")
    st.markdown("**Features:** Buy & Hold • Momentum • RSI • Mean Reversion")
    if st.button("Open Single Asset →", key="single", use_container_width=True):
        st.switch_page("pages/1_Single_Asset.py")

with col2:
    st.markdown("### Portfolio Management")
    st.markdown("Build multi-asset portfolios with correlation analysis and performance tracking.")
    st.markdown("**Features:** Custom Weights • Correlation Matrix • Normalized Values")
    if st.button("Open Portfolio →", key="portfolio", use_container_width=True):
        st.switch_page("pages/2_Portfolio.py")

# Footer
st.markdown("---")
st.markdown("*Finance Dashboard 2026 • Streamlit + Flask + Pandas • Academic Project*")
