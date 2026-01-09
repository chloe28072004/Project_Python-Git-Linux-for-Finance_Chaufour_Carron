# Quantitative Finance Dashboard

A professional finance dashboard for backtesting trading strategies and portfolio analysis with real-time market data.

## Features

### Single Asset Analysis
- **4 Trading Strategies**:
  - Buy & Hold (baseline)
  - Momentum (SMA 20/50 crossover)
  - RSI (Relative Strength Index)
  - Mean Reversion (Bollinger Bands)

### Portfolio Management
- Multi-asset portfolio construction
- Dynamic weight allocation
- Correlation matrix analysis
- Performance tracking

### Performance Metrics
- **Returns**: Total return, mean return
- **Risk Metrics**: Volatility, maximum drawdown, VaR, Beta
- **Risk-Adjusted Ratios**: Sharpe, Sortino, Calmar, Omega

## Architecture

```
┌─────────────────┐         HTTP API         ┌──────────────────┐
│                 │  ◄─────────────────────►  │                  │
│  Frontend       │    (port 5000)            │  Backend API     │
│  (Streamlit)    │                           │  (Flask)         │
│  port 8501      │                           │                  │
└─────────────────┘                           └──────────────────┘
                                                       │
                                                       │
                                              ┌────────┴────────┐
                                              │                 │
                                          ┌───▼───┐       ┌────▼────┐
                                          │Quant A│       │Quant B  │
                                          │Single │       │Portfolio│
                                          │Asset  │       │Analysis │
                                          └───────┘       └─────────┘
```

## Project Structure

```
finance-dashboard/
├── backend/              # Flask API server
│   ├── api.py           # Main API endpoints
│   ├── config.py        # Configuration
│   ├── data_source.py   # Yahoo Finance data fetching
│   ├── quant_a.py       # Single asset analysis
│   └── quant_b.py       # Portfolio analysis
├── frontend/            # Streamlit interface
│   ├── app.py          # Home page
│   └── pages/
│       ├── 1_Single_Asset.py   # Single asset analysis page
│       └── 2_Portfolio.py      # Portfolio management page
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Start Backend (Terminal 1)
```bash
python backend/api.py
```
Backend runs on `http://localhost:5000`

### Start Frontend (Terminal 2)
```bash
streamlit run frontend/app.py
```
Frontend runs on `http://localhost:8501`

### Access the Dashboard
Open your browser and navigate to `http://localhost:8501`

## API Endpoints

### Single Asset Analysis
- `GET /api/quant-a/historical/{symbol}?days={days}` - Get historical data
- `GET /api/quant-a/metrics/{symbol}` - Get performance metrics

### Portfolio Analysis
- `POST /api/quant-b/portfolio` - Analyze portfolio
- `POST /api/quant-b/optimize` - Optimize portfolio weights

### Example API Call
```bash
# Get historical data for Apple
curl http://localhost:5000/api/quant-a/historical/AAPL?days=90

# Get performance metrics
curl http://localhost:5000/api/quant-a/metrics/AAPL
```

## Technical Implementation

### Trading Strategies

#### Momentum (SMA Crossover)
```
SMA_20 = (1/20) × Σ(price[i-19:i])
SMA_50 = (1/50) × Σ(price[i-49:i])
Position = 1 if SMA_20 > SMA_50, else 0
```

#### RSI (Relative Strength Index)
```
RS = avg_gain / avg_loss
RSI = 100 - (100 / (1 + RS))
Buy when RSI < 30, Sell when RSI > 70
```

#### Mean Reversion (Bollinger Bands)
```
Upper = SMA + (2 × σ)
Lower = SMA - (2 × σ)
Buy at lower band, sell at upper band
```

### Risk Metrics

#### Volatility (Annualized)
```
σ_annual = σ_daily × √252
```

#### Sharpe Ratio
```
Sharpe = (R_p - R_f) / σ_p
Annualized: Sharpe_annual = Sharpe_daily × √252
Risk-free rate: 2% annual
```

#### Maximum Drawdown
```
Peak_t = max(portfolio_value[0:t])
DD_t = (portfolio_value[t] - Peak_t) / Peak_t
Max DD = min(DD_t)
```

## Data Source

Real-time and historical market data fetched from Yahoo Finance API via `yfinance` library.

## Configuration

Edit `backend/config.py` to modify:
- API host and port
- Data refresh intervals
- Risk-free rate (default: 2%)
- Default strategy parameters

## Technologies

- **Backend**: Flask, pandas, numpy, yfinance
- **Frontend**: Streamlit, plotly
- **Data**: Yahoo Finance API

## Notes

- All portfolio values are normalized to $100 starting capital
- Strategies use daily closing prices
- Auto-refresh every 5 minutes
- No transaction costs or slippage included
- Risk-free rate: 2% annual (0.02/252 daily)

## Troubleshooting

**Backend not starting?**
- Check port 5000 is not in use
- Ensure all dependencies are installed

**Frontend shows "Backend API not running"?**
- Ensure backend is running first on port 5000
- Check backend terminal for errors

**No data displaying?**
- Verify internet connection (Yahoo Finance API)
- Check symbol is valid (e.g., AAPL, GOOGL)

## License

Academic project - Educational purposes only
