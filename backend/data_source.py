"""
Shared Data Source Module
Common functions for fetching data (API or web scraping)
Used by both Quant A and Quant B
"""

import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import time


def handle_errors(func):
    """
    Decorator for graceful error handling
    Ensures app doesn't crash on failed API calls
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return None
    return wrapper


@handle_errors
def fetch_current_price(symbol):
    """
    Fetch current price for a single asset using Yahoo Finance

    Args:
        symbol (str): Asset symbol (e.g., 'AAPL', 'BTC-USD', 'EURUSD=X')

    Returns:
        dict: Contains current_price, timestamp, and symbol info
    """
    ticker = yf.Ticker(symbol)

    # Get current price from info
    info = ticker.info
    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')

    if current_price is None:
        # Fallback: get last price from recent history
        hist = ticker.history(period="1d")
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]

    return {
        'symbol': symbol,
        'current_price': float(current_price) if current_price else None,
        'timestamp': datetime.now().isoformat(),
        'currency': info.get('currency', 'USD'),
        'name': info.get('longName', symbol)
    }


@handle_errors
def fetch_historical_data(symbol, period="1mo", interval="1d"):
    """
    Fetch historical data for a single asset

    Args:
        symbol (str): Asset symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        pandas.DataFrame: Historical data with columns [Open, High, Low, Close, Volume]
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)

    if data.empty:
        print(f"Warning: No data available for {symbol}")
        return pd.DataFrame()

    # Reset index to have Date as a column
    data = data.reset_index()

    return data


@handle_errors
def fetch_historical_data_range(symbol, start_date, end_date, interval="1d"):
    """
    Fetch historical data for a specific date range

    Args:
        symbol (str): Asset symbol
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        interval (str): Data interval

    Returns:
        pandas.DataFrame: Historical data
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=interval)

    if data.empty:
        print(f"Warning: No data available for {symbol} between {start_date} and {end_date}")
        return pd.DataFrame()

    data = data.reset_index()
    return data


@handle_errors
def fetch_multiple_assets(symbols, period="1mo", interval="1d"):
    """
    Fetch historical data for multiple assets at once

    Args:
        symbols (list): List of asset symbols
        period (str): Time period
        interval (str): Data interval

    Returns:
        dict: Dictionary with symbol as key and DataFrame as value
    """
    data_dict = {}

    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        data = fetch_historical_data(symbol, period=period, interval=interval)
        if data is not None and not data.empty:
            data_dict[symbol] = data
        else:
            print(f"Warning: Could not fetch data for {symbol}")
            data_dict[symbol] = pd.DataFrame()

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    return data_dict


@handle_errors
def get_asset_info(symbol):
    """
    Get detailed information about an asset

    Args:
        symbol (str): Asset symbol

    Returns:
        dict: Asset information (name, sector, market cap, etc.)
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info

    return {
        'symbol': symbol,
        'name': info.get('longName', symbol),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'market_cap': info.get('marketCap'),
        'currency': info.get('currency', 'USD'),
        'exchange': info.get('exchange', 'N/A'),
        'description': info.get('longBusinessSummary', 'N/A')
    }


# Alternative: Web scraping function (if API fails or for specific sites)
def scrape_from_web(symbol, source_url):
    """
    Scrape data from website using requests + regex/BeautifulSoup
    This is a backup method if yfinance fails

    Example sources:
    - https://www.investing.com
    - https://www.boursorama.com
    """
    # TODO: Implement if needed as backup
    print(f"Web scraping not implemented yet for {symbol} from {source_url}")
    return None


# Helper function to convert common symbols to Yahoo Finance format
def normalize_symbol(symbol):
    """
    Convert common symbol formats to Yahoo Finance format

    Examples:
        EUR/USD -> EURUSD=X
        BTC/USD -> BTC-USD
        Gold -> GC=F (Gold Futures)
    """
    symbol_map = {
        'EUR/USD': 'EURUSD=X',
        'BTC/USD': 'BTC-USD',
        'ETH/USD': 'ETH-USD',
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Oil': 'CL=F',
        'ENGI': 'ENGI.PA',  # Engie on Paris exchange
    }

    return symbol_map.get(symbol, symbol)
