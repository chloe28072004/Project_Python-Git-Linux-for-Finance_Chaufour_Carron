"""
QUANT B - Multi-Asset Portfolio Backend
Handles portfolio data, optimization, and metrics for multiple assets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_source import (
    fetch_current_price,
    fetch_multiple_assets,
    normalize_symbol
)


class QuantB:
    """Multi-Asset Portfolio Engine"""

    def __init__(self, asset_symbols):
        self.asset_symbols = [normalize_symbol(s) for s in asset_symbols]
        self.data = {}  # Data for each asset
        self.weights = None
        self.portfolio_value = None
        self.current_prices = {}

    # data fetching

    def fetch_realtime_data(self):
        """
        Fetch real-time data for all assets
        """
        prices = {}
        for symbol in self.asset_symbols:
            price_data = fetch_current_price(symbol)
            if price_data:
                prices[symbol] = price_data.get('current_price')

        self.current_prices = prices
        return prices

    def get_historical_data(self, days=30):
        """
        Get historical data for all assets

        Args:
            days (int): Number of days of historical data

        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        # converting days to period
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

        data_dict = fetch_multiple_assets(self.asset_symbols, period=period)
        self.data = data_dict
        return data_dict

    def update_current_prices(self):
        """Get latest prices for all assets"""
        return self.fetch_realtime_data()

    # strategies

    def equal_weight_portfolio(self, initial_capital=10000):
        """
        Equal weight allocation - each asset gets same weight

        Args:
            initial_capital (float): Starting capital

        Returns:
            dict: Portfolio results with values over time
        """
        if not self.data or len(self.data) == 0:
            self.get_historical_data()

        n_assets = len(self.asset_symbols)
        weights = np.array([1.0 / n_assets] * n_assets)
        self.weights = weights

        return self._calculate_portfolio_performance(weights, initial_capital)

    def custom_weight_portfolio(self, weights, initial_capital=10000):
        """
        Custom weight allocation

        Args:
            weights (list): List of weights for each asset (must sum to 1)
            initial_capital (float): Starting capital

        Returns:
            dict: Portfolio results
        """
        if not self.data or len(self.data) == 0:
            self.get_historical_data()

        weights = np.array(weights)

        # normalize weights 
        weights = weights / weights.sum()
        self.weights = weights

        return self._calculate_portfolio_performance(weights, initial_capital)

    def _calculate_portfolio_performance(self, weights, initial_capital):
        """
        Calculate portfolio performance given weights

        Args:
            weights (np.array): Asset weights
            initial_capital (float): Starting capital

        Returns:
            dict: Portfolio performance data
        """
        # align all dataframes by date
        dfs = []
        valid_symbols = []

        for symbol in self.asset_symbols:
            if symbol in self.data and not self.data[symbol].empty:
                df = self.data[symbol][['Date', 'Close']].copy()
                df = df.rename(columns={'Close': symbol})
                df = df.set_index('Date')
                dfs.append(df)
                valid_symbols.append(symbol)

        if not dfs:
            return {'error': 'No valid data for portfolio calculation'}

       
        combined_df = pd.concat(dfs, axis=1, join='inner')
        combined_df = combined_df.dropna()

        if combined_df.empty:
            return {'error': 'No overlapping dates for assets'}

        # daily returns for each asset
        returns = combined_df.pct_change().dropna()

        portfolio_returns = (returns * weights).sum(axis=1)

        portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
        portfolio_value = pd.concat([pd.Series([initial_capital], index=[returns.index[0]]), portfolio_value])

        #  individual asset values 
        asset_values = {}
        for symbol in valid_symbols:
            asset_values[symbol] = (initial_capital * (1 + returns[symbol]).cumprod()).tolist()

        final_value = portfolio_value.iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100

        result = {
            'symbols': valid_symbols,
            'weights': weights.tolist(),
            'dates': portfolio_value.index.tolist(),
            'portfolio_values': portfolio_value.tolist(),
            'asset_values': asset_values,
            'asset_prices': combined_df.to_dict('list'),
            'initial_capital': initial_capital,
            'final_value': float(final_value),
            'total_return': float(total_return)
        }

        return result

    def rebalance_portfolio(self, frequency='monthly'):
        """
        Rebalance portfolio at specified frequency

        Args:
            frequency (str): Rebalancing frequency ('daily', 'weekly', 'monthly')

        Returns:
            dict: Rebalanced portfolio performance
        """
        return self.equal_weight_portfolio()

    # metrics

    def calculate_correlation_matrix(self):
        """
        Calculate correlation matrix between assets

        Returns:
            pandas.DataFrame: Correlation matrix
        """
        if not self.data or len(self.data) == 0:
            self.get_historical_data()

        # combine close prices for all assets
        dfs = []
        for symbol in self.asset_symbols:
            if symbol in self.data and not self.data[symbol].empty:
                df = self.data[symbol][['Date', 'Close']].copy()
                df = df.rename(columns={'Close': symbol})
                df = df.set_index('Date')
                dfs.append(df)

        if not dfs:
            return None

        combined_df = pd.concat(dfs, axis=1, join='inner')
        returns = combined_df.pct_change().dropna()

        correlation_matrix = returns.corr()
        return correlation_matrix

    def calculate_portfolio_volatility(self):
        """
        Calculate portfolio volatility (annualized standard deviation)

        Returns:
            float: Portfolio volatility as percentage
        """
        if not self.data or self.weights is None:
            return None

        # returns for all assets
        dfs = []
        for symbol in self.asset_symbols:
            if symbol in self.data and not self.data[symbol].empty:
                df = self.data[symbol][['Date', 'Close']].copy()
                df = df.rename(columns={'Close': symbol})
                df = df.set_index('Date')
                dfs.append(df)

        if not dfs:
            return None

        combined_df = pd.concat(dfs, axis=1, join='inner')
        returns = combined_df.pct_change().dropna()

        portfolio_returns = (returns * self.weights).sum(axis=1)

        # annualized volatility
        volatility = portfolio_returns.std() * np.sqrt(252) * 100

        return float(volatility)

    def calculate_diversification_ratio(self):
        """
        Calculate diversification ratio
        Ratio of weighted average volatilities to portfolio volatility

        Returns:
            float: Diversification ratio (>1 means diversification benefit)
        """
        if not self.data or self.weights is None:
            return None

        # returns for all assets
        dfs = []
        valid_symbols = []
        for symbol in self.asset_symbols:
            if symbol in self.data and not self.data[symbol].empty:
                df = self.data[symbol][['Date', 'Close']].copy()
                df = df.rename(columns={'Close': symbol})
                df = df.set_index('Date')
                dfs.append(df)
                valid_symbols.append(symbol)

        if not dfs:
            return None

        combined_df = pd.concat(dfs, axis=1, join='inner')
        returns = combined_df.pct_change().dropna()

        individual_vols = returns.std() * np.sqrt(252)

        weighted_vol = (self.weights * individual_vols).sum()

        portfolio_returns = (returns * self.weights).sum(axis=1)
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)

        if portfolio_vol == 0:
            return None

        diversification_ratio = weighted_vol / portfolio_vol

        return float(diversification_ratio)

    def get_portfolio_metrics(self):
        """
        Get all portfolio metrics

        Returns:
            dict: Dictionary of portfolio metrics
        """
        if not self.data or len(self.data) == 0:
            self.get_historical_data()

   
        self.update_current_prices()

        
        corr_matrix = self.calculate_correlation_matrix()

        # If we don't have weights yet we use equal weights
        if self.weights is None:
            self.equal_weight_portfolio()

        volatility = self.calculate_portfolio_volatility()
        div_ratio = self.calculate_diversification_ratio()

        #  portfolio return if we have weights
        portfolio_return = None
        if self.weights is not None:
            result = self._calculate_portfolio_performance(self.weights, 10000)
            if 'total_return' in result:
                portfolio_return = result['total_return']

        metrics = {
            'symbols': self.asset_symbols,
            'current_prices': self.current_prices,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'portfolio_value': None,  # need initial capital
            'portfolio_return': portfolio_return,
            'total_return': portfolio_return,  
            'portfolio_volatility': volatility,
            'diversification_ratio': div_ratio,
            'correlation_matrix': corr_matrix.to_dict() if corr_matrix is not None else None
        }

        return metrics
