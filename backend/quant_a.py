"""
QUANT A - Single Asset Analysis Backend
Handles data fetching, strategies, and metrics for single asset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_source import (
    fetch_current_price,
    fetch_historical_data,
    normalize_symbol
)


class QuantA:
    """Single Asset Analysis Engine"""

    def __init__(self, asset_symbol):
        self.asset_symbol = normalize_symbol(asset_symbol)
        self.data = None
        self.current_price = None
        self.strategy_results = None

    # data fetching

    def fetch_realtime_data(self):
        """
        Fetch real-time data from Yahoo Finance API
        """
        price_data = fetch_current_price(self.asset_symbol)
        if price_data:
            self.current_price = price_data.get('current_price')
            return price_data
        return None

    def get_historical_data(self, days=30):
        """
        Get historical price data

        Args:
            days (int): Number of days of historical data

        Returns:
            pandas.DataFrame: Historical data with Date, Open, High, Low, Close, Volume
        """
        # Convert days to period string for yfinance
        if days <= 7:
            period = f"{days}d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        elif days <= 365:
            period = "1y"
        else:
            period = "max"

        data = fetch_historical_data(self.asset_symbol, period=period)

        if data is not None and not data.empty:
            self.data = data
            return data
        return pd.DataFrame()

    def update_current_price(self):
        """Get latest price"""
        price_data = self.fetch_realtime_data()
        if price_data:
            self.current_price = price_data.get('current_price')
        return self.current_price

    def get_timestamp(self):
        """Get current timestamp"""
        return datetime.now().isoformat()

    # strategies

    def strategy_buy_and_hold(self, initial_capital=10000):
        """
        Buy and hold strategy
        Buys at first available price and holds until end

        Args:
            initial_capital (float): Starting capital

        Returns:
            dict: Strategy results with portfolio values and metrics
        """
        if self.data is None or self.data.empty:
            self.get_historical_data()

        if self.data is None or self.data.empty:
            return {'error': 'No data available'}

        prices = self.data['Close'].values
        dates = self.data['Date'].values

    
        num_shares = initial_capital / prices[0]

        
        portfolio_values = num_shares * prices

      
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital * 100

        self.strategy_results = {
            'strategy': 'buy_and_hold',
            'dates': dates.tolist(),
            'portfolio_values': portfolio_values.tolist(),
            'prices': prices.tolist(),
            'initial_capital': initial_capital,
            'final_value': float(portfolio_values[-1]),
            'total_return': float(total_return),
            'num_shares': float(num_shares)
        }

        return self.strategy_results

    def strategy_momentum(self, window=20, initial_capital=10000):
        """
        Momentum strategy
        Buy when price > moving average, sell when price < moving average

        Args:
            window (int): Moving average window
            initial_capital (float): Starting capital

        Returns:
            dict: Strategy results with portfolio values and metrics
        """
        if self.data is None or self.data.empty:
            self.get_historical_data()

        if self.data is None or self.data.empty:
            return {'error': 'No data available'}

        df = self.data.copy()

        # moving average
        df['MA'] = df['Close'].rolling(window=window).mean()

        # signals 1 = buy and 0 = sell
        df['Signal'] = 0
        df.loc[df['Close'] > df['MA'], 'Signal'] = 1

       
        df['Position'] = df['Signal'].shift(1).fillna(0)  # Lag signal by 1 day
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Position'] * df['Returns']

       
        df['Portfolio_Value'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()
        df['Portfolio_Value'].fillna(initial_capital, inplace=True)

      
        final_value = df['Portfolio_Value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100

        self.strategy_results = {
            'strategy': 'momentum',
            'window': window,
            'dates': df['Date'].tolist(),
            'portfolio_values': df['Portfolio_Value'].tolist(),
            'prices': df['Close'].tolist(),
            'signals': df['Signal'].tolist(),
            'moving_average': df['MA'].tolist(),
            'initial_capital': initial_capital,
            'final_value': float(final_value),
            'total_return': float(total_return)
        }

        return self.strategy_results

    # metrics

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calculate Sharpe ratio (annualized)

        Formula: Sharpe = (mean_daily_return - rf_daily) / std_daily_return × √252

        Args:
            returns (array-like): Daily returns series
            risk_free_rate (float): Annual risk-free rate (default 2%)

        Returns:
            float: Annualized Sharpe ratio
        """
        if len(returns) == 0:
            return None

        returns = np.array(returns)

     
        rf_daily = risk_free_rate / 252

      
        excess_returns = returns - rf_daily

       
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)  # Sample std

        if std_excess == 0:
            return 0

        # annualize 
        sharpe = (mean_excess / std_excess) * np.sqrt(252)

        return float(sharpe)

    def calculate_max_drawdown(self, prices):
        """
        Calculate maximum drawdown

        Args:
            prices (array-like): Price series

        Returns:
            float: Maximum drawdown as percentage
        """
        if len(prices) == 0:
            return None

        prices = np.array(prices)
        cumulative_max = np.maximum.accumulate(prices)
        drawdowns = (prices - cumulative_max) / cumulative_max * 100
        max_drawdown = np.min(drawdowns)

        return float(max_drawdown)

    def calculate_sortino_ratio(self, returns, risk_free_rate=0.02, target=0):
        """
        Calculate Sortino ratio (annualized, downside deviation only)

        Formula: Sortino = (mean_daily_return - rf_daily) / downside_std × √252

        Args:
            returns (array-like): Daily returns series
            risk_free_rate (float): Annual risk-free rate (default 2%)
            target (float): Target return threshold (default 0)

        Returns:
            float: Annualized Sortino ratio
        """
        if len(returns) == 0:
            return None

        returns = np.array(returns)

    
        rf_daily = risk_free_rate / 252

        
        excess_returns = returns - rf_daily

        downside_returns = returns[returns < target] - target

        if len(downside_returns) == 0:
            return 0

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return 0

        # annualize
        sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(252)

        return float(sortino)

    def calculate_calmar_ratio(self, returns, max_drawdown):
        """
        Calculate Calmar ratio (annualized return / |max drawdown|)

        Formula: Calmar = (mean_daily_return × 252) / |max_drawdown|

        Args:
            returns (array-like): Daily returns series
            max_drawdown (float): Maximum drawdown as percentage (negative value)

        Returns:
            float: Calmar ratio
        """
        if max_drawdown == 0 or max_drawdown >= 0:
            return 0

        # annualized return
        mean_daily_return = np.mean(returns)
        annual_return = mean_daily_return * 252

        calmar = annual_return / abs(max_drawdown / 100)

        return float(calmar)

    def calculate_omega_ratio(self, returns, threshold=0):
        """
        Calculate Omega ratio

        Args:
            returns (array-like): Returns series
            threshold (float): Threshold return (default 0)

        Returns:
            float: Omega ratio
        """
        returns = np.array(returns)

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        if np.sum(losses) == 0:
            return float('inf') if np.sum(gains) > 0 else 1.0

        omega = np.sum(gains) / np.sum(losses)

        return float(omega)

    def get_performance_metrics(self):
        """
        Get all performance metrics (NON-ANNUALIZED except volatility)

        Returns:
            dict: Dictionary of performance metrics
        """
        if self.data is None or self.data.empty:
            self.get_historical_data()

        if self.data is None or self.data.empty:
            return {
                'current_price': None,
                'sharpe_ratio': None,
                'sortino_ratio': None,
                'calmar_ratio': None,
                'omega_ratio': None,
                'max_drawdown': None,
                'total_return': None,
                'volatility': None,
                'mean_return': None
            }

       
        self.update_current_price()

      
        returns = self.data['Close'].pct_change().dropna()

       
        max_dd = self.calculate_max_drawdown(self.data['Close'].values)

        # ratios
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        calmar = self.calculate_calmar_ratio(returns, max_dd)
        omega = self.calculate_omega_ratio(returns)

        # non annualized total return
        first_price = self.data['Close'].iloc[0]
        last_price = self.data['Close'].iloc[-1]
        total_return = (last_price - first_price) / first_price * 100

        # non annualized mean return
        mean_return = returns.mean() * 100  

        # annualized volatility using same ddof as Sharpe/Sortino
        daily_vol = returns.std(ddof=1)  # sample std
        annual_vol = daily_vol * np.sqrt(252) * 100  # annualized %

        metrics = {
            'current_price': float(self.current_price) if self.current_price else None,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'omega_ratio': omega,
            'max_drawdown': max_dd,
            'total_return': float(total_return),
            'volatility': float(annual_vol),  # annualized vol
            'mean_return': float(mean_return)  # daily mean %
        }

        return metrics
