"""
Single Asset Analysis Page
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

API_URL = "http://localhost:5000"
FIXED_CAPITAL = 100

POPULAR_TICKERS = {
    'A': ['AAPL', 'AMZN', 'AMD', 'ADBE', 'AVGO', 'AXP', 'ABNB'],
    'B': ['BA', 'BAC', 'BABA', 'BRK-B', 'BKNG'],
    'C': ['CSCO', 'C', 'CRM', 'CVX', 'COST', 'CMCSA'],
    'D': ['DIS', 'DDOG', 'DHR'],
    'E': ['EBAY', 'EA'],
    'F': ['FB', 'F', 'FDX'],
    'G': ['GOOGL', 'GOOG', 'GS', 'GM', 'GILD'],
    'H': ['HD', 'HON'],
    'I': ['INTC', 'IBM', 'INTU'],
    'J': ['JNJ', 'JPM', 'JD'],
    'K': ['KO'],
    'L': ['LMT', 'LOW'],
    'M': ['MSFT', 'MA', 'MCD', 'META', 'MRK', 'MMM'],
    'N': ['NVDA', 'NFLX', 'NKE', 'NEE'],
    'O': ['ORCL'],
    'P': ['PFE', 'PG', 'PYPL', 'PM'],
    'Q': ['QCOM'],
    'R': ['RTX'],
    'S': ['SBUX', 'SNOW', 'SQ', 'SHOP'],
    'T': ['TSLA', 'TSM', 'TXN', 'TMO'],
    'U': ['UNH', 'UPS', 'UBER'],
    'V': ['V', 'VZ', 'VRTX'],
    'W': ['WMT', 'WFC', 'DIS'],
    'X': ['XOM'],
    'Z': ['ZM', 'ZION']
}

st.set_page_config(
    page_title="Single Asset Analysis",
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

    /* Headers */
    h1 { color: #fafafa; font-size: 24px; font-weight: 600; margin: 0 0 8px 0; }
    h2, h3 { color: #a1a1aa; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin: 24px 0 12px 0; }
    p, span, label { color: #71717a; }

    /* Metrics */
    [data-testid="stMetricValue"] { font-size: 20px; font-weight: 700; color: #fafafa; }
    [data-testid="stMetricLabel"] { color: #71717a; font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    div[data-testid="metric-container"] { background-color: #18181b; padding: 16px; border-radius: 10px; border: 1px solid #27272a; }

    /* Status indicator */
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

    /* Inputs */
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

st.markdown("""
    <div class="live-indicator">
        <div class="live-dot"></div>
        <span>LIVE</span>
    </div>
""", unsafe_allow_html=True)

def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def fetch_data(symbol, days):
    try:
        response = requests.get(f"{API_URL}/api/quant-a/historical/{symbol}?days={days}", timeout=10)
        if response.status_code == 200:
            return response.json().get('data', [])
        return []
    except:
        return []

def fetch_metrics(symbol):
    try:
        response = requests.get(f"{API_URL}/api/quant-a/metrics/{symbol}", timeout=10)
        if response.status_code == 200:
            return response.json().get('metrics', {})
        return {}
    except:
        return {}

st.markdown("# Single Asset Analysis")
st.markdown("<p style='color: #71717a; margin-top: -8px;'>Backtest strategies on individual securities</p>", unsafe_allow_html=True)

# check API
if not check_api():
    st.error("Backend API not running. Start with: python backend/api.py")
    st.stop()

st.sidebar.markdown("### Configuration")

symbol_input = st.sidebar.text_input("Ticker Symbol", value="AAPL", key="ticker").upper()

# show suggestions if user types
if symbol_input and len(symbol_input) > 0:
    first_letter = symbol_input[0].upper()
    if first_letter in POPULAR_TICKERS:
        matching_tickers = [t for t in POPULAR_TICKERS[first_letter] if t.startswith(symbol_input)]
        if matching_tickers and symbol_input not in matching_tickers:
            st.sidebar.info(f"💡 Suggestions: {', '.join(matching_tickers[:5])}")

symbol = symbol_input
days = st.sidebar.slider("Period (days)", 30, 365, 90, key="days")

st.sidebar.markdown("### Strategy")
strategy = st.sidebar.selectbox(
    "Select Strategy",
    ["Buy & Hold", "Momentum", "RSI", "Mean Reversion"],
    key="strategy"
)

if st.sidebar.button("Analyze", type="primary"):
    st.session_state.analyzed = True

# Main content
if 'analyzed' not in st.session_state:
    st.info("Configure parameters and click Analyze to start")
    st.stop()

metrics = fetch_metrics(symbol)
historical = fetch_data(symbol, days)

st.markdown(f"<h2>Performance Metrics</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    price = metrics.get('current_price', 0)
    st.metric("Current Price", f"${price:.2f}" if price else "N/A")

with col2:
    ret = metrics.get('total_return')
    st.metric("Total Return", f"{ret:.2f}%" if ret is not None else "N/A")

with col3:
    vol = metrics.get('volatility')
    st.metric("Volatility", f"{vol:.2f}%" if vol is not None else "N/A")

with col4:
    sharpe = metrics.get('sharpe_ratio')
    st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "N/A")

st.markdown(f"<h2>Strategy Performance</h2>", unsafe_allow_html=True)

if historical and len(historical) > 0:
    df = pd.DataFrame(historical)

    if 'Close' in df.columns and 'Date' in df.columns:
        # Normalize to 100
        initial_price = df['Close'].iloc[0]
        df['BuyHold'] = (df['Close'] / initial_price) * 100

        # momentum 
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # momentum strategy: buy when SMA20 > SMA50, otherwise hold cash
        df['Position_Momentum'] = 0
        df.loc[df['SMA_20'] > df['SMA_50'], 'Position_Momentum'] = 1
        df['Momentum'] = 100.0 

        for i in range(1, len(df)):
            if df['Position_Momentum'].iloc[i] == 1:
                # In market
                price_change = df['Close'].iloc[i] / df['Close'].iloc[i-1]
                df.loc[df.index[i], 'Momentum'] = df['Momentum'].iloc[i-1] * price_change
            else:
                # Out of market
                df.loc[df.index[i], 'Momentum'] = df['Momentum'].iloc[i-1]

        # RSI = 100 - (100 / (1 + RS))
        # RS = Average Gain / Average Loss over period
        period = 14
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

    
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

        # RSI Strategy: Buy when RSI < 30, Sell when RSI > 70 
        df['Position_RSI'] = 0
        df.loc[df['RSI'] < 30, 'Position_RSI'] = 1  # Buy signal
        df.loc[df['RSI'] > 70, 'Position_RSI'] = -1  # Sell signal

        # Forward fill positions
        df['Position_RSI'] = df['Position_RSI'].replace(0, pd.NA).fillna(method='ffill').fillna(0)

        df['RSI_Strategy'] = 100.0
        for i in range(1, len(df)):
            if df['Position_RSI'].iloc[i] == 1:
                # In market
                price_change = df['Close'].iloc[i] / df['Close'].iloc[i-1]
                df.loc[df.index[i], 'RSI_Strategy'] = df['RSI_Strategy'].iloc[i-1] * price_change
            else:
                # Out of market
                df.loc[df.index[i], 'RSI_Strategy'] = df['RSI_Strategy'].iloc[i-1]

        # Mean Reversion Strategy: Bollinger Bands
        # Buy when price touches lower band, sell when touches upper band
        window = 20
        df['SMA'] = df['Close'].rolling(window=window).mean()
        df['STD'] = df['Close'].rolling(window=window).std()
        df['Upper_Band'] = df['SMA'] + (2 * df['STD'])
        df['Lower_Band'] = df['SMA'] - (2 * df['STD'])

        df['Position_MR'] = 0
        df.loc[df['Close'] <= df['Lower_Band'], 'Position_MR'] = 1  # Buy when price hits lower band
        df.loc[df['Close'] >= df['Upper_Band'], 'Position_MR'] = -1  # Sell when price hits upper band

        df['Position_MR'] = df['Position_MR'].replace(0, pd.NA).fillna(method='ffill').fillna(0)

        df['MeanReversion'] = 100.0
        for i in range(1, len(df)):
            if df['Position_MR'].iloc[i] == 1:
                # In market
                price_change = df['Close'].iloc[i] / df['Close'].iloc[i-1]
                df.loc[df.index[i], 'MeanReversion'] = df['MeanReversion'].iloc[i-1] * price_change
            else:
                # Out of market
                df.loc[df.index[i], 'MeanReversion'] = df['MeanReversion'].iloc[i-1]

       
        fig = go.Figure()

        # Buy & hold
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['BuyHold'],
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#6366f1', width=2.5),
            hovertemplate='<b>Buy & Hold</b><br>$%{y:.2f}<extra></extra>'
        ))


        if strategy == "Momentum":
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Momentum'],
                mode='lines',
                name='Momentum',
                line=dict(color='#10b981', width=2.5),
                hovertemplate='<b>Momentum</b><br>$%{y:.2f}<extra></extra>'
            ))
        elif strategy == "RSI":
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['RSI_Strategy'],
                mode='lines',
                name='RSI',
                line=dict(color='#f59e0b', width=2.5),
                hovertemplate='<b>RSI</b><br>$%{y:.2f}<extra></extra>'
            ))
        elif strategy == "Mean Reversion":
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['MeanReversion'],
                mode='lines',
                name='Mean Reversion',
                line=dict(color='#ec4899', width=2.5),
                hovertemplate='<b>Mean Reversion</b><br>$%{y:.2f}<extra></extra>'
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
                tickfont=dict(size=9, color='#3f3f46'),
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

        # Strategy comparison
        if strategy != "Buy & Hold":
            st.markdown("<h2>Strategy Comparison</h2>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                final_bh = df['BuyHold'].iloc[-1]
                bh_return = ((final_bh - 100) / 100) * 100

                st.markdown(f"""
                    <div style="background: #18181b; padding: 20px; border-radius: 10px; border: 1px solid #27272a;">
                        <div style="color: #71717a; font-size: 11px; text-transform: uppercase; margin-bottom: 8px;">Buy & Hold</div>
                        <div style="color: #fafafa; font-size: 24px; font-weight: 600;">${final_bh:.2f}</div>
                        <div style="color: {'#10b981' if bh_return >= 0 else '#ef4444'}; font-size: 14px; margin-top: 8px;">
                            {'+' if bh_return >= 0 else ''}{bh_return:.2f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                if strategy == "Momentum":
                    final_strat = df['Momentum'].iloc[-1]
                    strat_name = "Momentum"
                elif strategy == "RSI":
                    final_strat = df['RSI_Strategy'].iloc[-1]
                    strat_name = "RSI"
                elif strategy == "Mean Reversion":
                    final_strat = df['MeanReversion'].iloc[-1]
                    strat_name = "Mean Reversion"

                strat_return = ((final_strat - 100) / 100) * 100

                st.markdown(f"""
                    <div style="background: #18181b; padding: 20px; border-radius: 10px; border: 1px solid #27272a;">
                        <div style="color: #71717a; font-size: 11px; text-transform: uppercase; margin-bottom: 8px;">{strat_name}</div>
                        <div style="color: #fafafa; font-size: 24px; font-weight: 600;">${final_strat:.2f}</div>
                        <div style="color: {'#10b981' if strat_return >= 0 else '#ef4444'}; font-size: 14px; margin-top: 8px;">
                            {'+' if strat_return >= 0 else ''}{strat_return:.2f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No price data available")
else:
    st.info("No historical data available. Configure data source in backend.")

st.markdown("<h2>Risk-Adjusted Ratios</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    sharpe = metrics.get('sharpe_ratio')
    st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "N/A")

with col2:
    sortino = metrics.get('sortino_ratio')
    st.metric("Sortino Ratio", f"{sortino:.2f}" if sortino is not None else "N/A")

with col3:
    calmar = metrics.get('calmar_ratio')
    st.metric("Calmar Ratio", f"{calmar:.2f}" if calmar is not None else "N/A")

with col4:
    omega = metrics.get('omega_ratio')
    st.metric("Omega Ratio", f"{omega:.2f}" if omega is not None else "N/A")

st.markdown("<h2>Risk Metrics</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    max_dd = metrics.get('max_drawdown')
    st.metric("Max Drawdown", f"{max_dd:.2f}%" if max_dd is not None else "N/A")

with col2:
    vol = metrics.get('volatility')
    st.metric("Volatility (Annual)", f"{vol:.2f}%" if vol is not None else "N/A")

with col3:
    mean_ret = metrics.get('mean_return')
    st.metric("Mean Daily Return", f"{mean_ret:.4f}%" if mean_ret is not None else "N/A")

st.markdown("<h2>Period Information</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.metric("Analysis Period", f"{days} days")

with col2:
    st.metric("Initial Capital", f"${FIXED_CAPITAL}")

# auto refresh every 5 minutes
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

time_diff = (datetime.now() - st.session_state.last_refresh).total_seconds()
if time_diff > 300:
    st.session_state.last_refresh = datetime.now()
    st.rerun()
