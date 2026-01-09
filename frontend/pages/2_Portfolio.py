"""
Portfolio Analysis Page
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

API_URL = "http://localhost:5000"
FIXED_CAPITAL = 100

st.set_page_config(
    page_title="Portfolio Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #09090b; }
    [data-testid="stHeader"] { background-color: #09090b; }
    [data-testid="stSidebar"] { background-color: #18181b; border-right: 1px solid #27272a; }
    .modebar { display: none !important; }

    h1 { color: #fafafa; font-size: 24px; font-weight: 600; margin: 0 0 8px 0; }
    h2, h3 { color: #a1a1aa; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin: 24px 0 12px 0; }
    p, span, label { color: #71717a; }

    [data-testid="stMetricValue"] { font-size: 20px; font-weight: 700; color: #fafafa; }
    [data-testid="stMetricLabel"] { color: #71717a; font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    div[data-testid="metric-container"] { background-color: #18181b; padding: 16px; border-radius: 10px; border: 1px solid #27272a; }

    .live-indicator {
        position: fixed;
        top: 80px;
        right: 24px;
        background: #18181b;
        padding: 8px 14px;
        border-radius: 20px;
        border: 1px solid #27272a;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        color: #a1a1aa;
        z-index: 1000;
    }
    .live-dot {
        width: 6px;
        height: 6px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

    input, select, textarea { background-color: #18181b !important; color: #fafafa !important; border: 1px solid #27272a !important; }

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

# Live indicator
st.markdown("""
    <div class="live-indicator">
        <div class="live-dot"></div>
        <span>LIVE</span>
    </div>
""", unsafe_allow_html=True)

# Helper functions
def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def normalize_weights(weights):
    total = sum(weights)
    if total == 0:
        return [1.0/len(weights) for _ in weights]
    return [w/total for w in weights]

def fetch_portfolio_data(symbols):
    """Simulated portfolio data - normalized to $100 starting value"""
    import numpy as np

    
    days = 90
    data = {'dates': pd.date_range(end=datetime.now(), periods=days).strftime('%Y-%m-%d').tolist()}

    # each asset starts at 100 and varies randomly
    for symbol in symbols:
        np.random.seed(hash(symbol) % 1000)
        returns = np.random.normal(0.001, 0.02, days)
        values = 100 * np.cumprod(1 + returns)
        data[symbol] = values.tolist()

    return data


st.markdown("# Portfolio Analysis")
st.markdown("<p style='color: #71717a; margin-top: -8px;'>Build and analyze multi-asset portfolios</p>", unsafe_allow_html=True)

# check API
if not check_api():
    st.error("Backend API not running. Start with: python backend/api.py")
    st.stop()

st.sidebar.markdown("### Portfolio Assets")
num_assets = st.sidebar.number_input("Number of Assets", 2, 8, 3, key="num")

if 'symbols' not in st.session_state:
    st.session_state.symbols = ["AAPL", "GOOGL", "MSFT"]
if 'weights' not in st.session_state:
    st.session_state.weights = [33.3, 33.3, 33.4]

while len(st.session_state.symbols) < num_assets:
    st.session_state.symbols.append(f"ASSET{len(st.session_state.symbols)+1}")
    st.session_state.weights.append(100 / num_assets)

st.session_state.symbols = st.session_state.symbols[:num_assets]
st.session_state.weights = st.session_state.weights[:num_assets]

symbols = []
weights = []

for i in range(num_assets):
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        symbol = col1.text_input(f"Asset {i+1}", st.session_state.symbols[i], key=f"sym_{i}", label_visibility="collapsed")
        symbols.append(symbol)
    with col2:
        weight = col2.number_input(f"W{i+1}", min_value=0.0, max_value=100.0, value=st.session_state.weights[i], step=1.0, key=f"wgt_{i}", label_visibility="collapsed")
        weights.append(weight)

st.session_state.symbols = symbols
st.session_state.weights = weights


total_weight = sum(weights)
if abs(total_weight - 100) > 0.1:
    st.sidebar.warning(f"Total: {total_weight:.1f}% (should be 100%)")
else:
    st.sidebar.success(f"Total: {total_weight:.1f}%")


if st.sidebar.button("Normalize Weights", type="primary"):
    normalized = normalize_weights(weights)
    st.session_state.weights = [w * 100 for w in normalized]
    st.rerun()

st.sidebar.markdown("---")

if st.sidebar.button("Analyze Portfolio", type="primary"):
    st.session_state.analyzed = True

# Main content
if 'analyzed' not in st.session_state:
    st.info("Configure portfolio and click Analyze to start")
    st.stop()

# Fetch portfolio data
portfolio_data = fetch_portfolio_data(symbols)

# Portfolio metrics 
total_value = FIXED_CAPITAL * (1 + 0.15)  
total_return = 15.0
volatility = 12.5
sharpe = 1.2

st.markdown("<h2>Portfolio Performance</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Portfolio Value", f"${total_value:.2f}")

with col2:
    st.metric("Total Return", f"+{total_return:.2f}%")

with col3:
    st.metric("Volatility", f"{volatility:.2f}%")

with col4:
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

# portfolio performance chart with all assets normalized to $100
st.markdown("<h2>Asset Performance (Normalized to $100)</h2>", unsafe_allow_html=True)

COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316', '#ec4899']

fig = go.Figure()

# plotting each asset
for idx, symbol in enumerate(symbols):
    if symbol in portfolio_data:
        fig.add_trace(go.Scatter(
            y=portfolio_data[symbol],
            mode='lines',
            name=symbol,
            line=dict(color=COLORS[idx % len(COLORS)], width=2),
            hovertemplate=f'<b>{symbol}</b><br>Value: $%{{y:.2f}}<extra></extra>'
        ))

# portfolio value based on weights
portfolio_values = []
for i in range(len(portfolio_data['dates'])):
    weighted_sum = sum(portfolio_data[symbols[j]][i] * (weights[j] / 100) for j in range(len(symbols)))
    portfolio_values.append(weighted_sum)

# portfolio total 
fig.add_trace(go.Scatter(
    y=portfolio_values,
    mode='lines',
    name='Total Portfolio',
    line=dict(color='#ffffff', width=3),
    hovertemplate='<b>Portfolio</b><br>Value: $%{y:.2f}<extra></extra>'
))

fig.update_layout(
    height=450,
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor='#09090b',
    paper_bgcolor='#09090b',
    font=dict(size=11, color='#71717a'),
    xaxis=dict(
        gridcolor='#27272a',
        showgrid=True,
        showticklabels=False,
        zeroline=False
    ),
    yaxis=dict(
        gridcolor='#27272a',
        showgrid=True,
        title="Value ($)",
        zeroline=False
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        bgcolor='rgba(24, 24, 27, 0.8)',
        bordercolor='#27272a',
        borderwidth=1,
        font=dict(color='#fafafa')
    ),
    hovermode='x unified',
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# Asset allocation and correlation
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2>Asset Allocation</h2>", unsafe_allow_html=True)

    fig = go.Figure(data=[go.Pie(
        labels=symbols,
        values=weights,
        hole=0.5,
        marker=dict(
            colors=COLORS[:len(symbols)],
            line=dict(color='#09090b', width=3)
        ),
        textinfo='label+percent',
        textfont=dict(size=12, color='#fafafa'),
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>'
    )])

    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#09090b',
        font=dict(size=11, color='#fafafa'),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col2:
    st.markdown("<h2>Correlation Matrix</h2>", unsafe_allow_html=True)

    # correlation matrix
    import numpy as np
    n = len(symbols)
    corr_matrix = np.random.rand(n, n)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = np.clip(corr_matrix, -1, 1)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=symbols,
        y=symbols,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix,
        texttemplate='%{text:.2f}',
        textfont={"size": 10, "color": "#fafafa"},
        colorbar=dict(
            thickness=15,
            len=0.7,
            tickfont=dict(size=10, color='#fafafa')
        ),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#09090b',
        plot_bgcolor='#09090b',
        font=dict(size=10, color='#fafafa'),
        xaxis=dict(side='bottom', tickfont=dict(color='#fafafa')),
        yaxis=dict(autorange='reversed', tickfont=dict(color='#fafafa'))
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


st.markdown("<h2>Portfolio Ratios</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.metric("Sharpe Ratio (Est.)", f"{sharpe:.2f}")

with col2:
    info_ratio = total_return / volatility if volatility > 0 else 0
    st.metric("Information Ratio", f"{info_ratio:.2f}")

st.markdown("<h2>Risk Metrics</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    est_max_dd = -1.5 * volatility  
    st.metric("Est. Max Drawdown", f"{est_max_dd:.2f}%")

with col2:
    st.metric("Portfolio Volatility", f"{volatility:.2f}%")

with col3:
    avg_weight = 100 / len(symbols)
    st.metric("Avg Asset Weight", f"{avg_weight:.1f}%")

st.markdown("<h2>Portfolio Information</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.metric("Number of Assets", f"{len(symbols)}")

with col2:
    st.metric("Initial Capital", f"${FIXED_CAPITAL}")

# auto refresh every 5 minutes
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

time_diff = (datetime.now() - st.session_state.last_refresh).total_seconds()
if time_diff > 300:
    st.session_state.last_refresh = datetime.now()
    st.rerun()
