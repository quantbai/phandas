"""
Layer-based backtesting engine for long-short strategies.

Implements quantile-based portfolio construction:
- Long top quantile assets (high factor values)
- Short bottom quantile assets (low factor values)
- Equal-weighted within each quantile
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
import logging

from .backtest import Portfolio, Backtester
from .core import Factor

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class LayerBacktester(Backtester):
    """
    Layer-based backtesting for quantile long-short strategies.
    
    Constructs portfolio by:
    1. Ranking assets by factor values each day
    2. Going long top quantile (e.g., top 20%)
    3. Going short bottom quantile (e.g., bottom 20%)
    4. Equal-weighted within each quantile
    5. 50% capital allocated to long, 50% to short (dollar-neutral)
    
    Differences from standard Backtester:
    - Standard: Uses all assets with factor-proportional weights
    - Layer: Only uses top/bottom quantiles with equal weights
    """
    
    def __init__(
        self,
        price_factor: 'Factor',
        strategy_factor: 'Factor',
        long_top_n: int = None,
        short_bottom_n: int = None,
        transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
        initial_capital: float = 100000
    ):
        """
        Initialize layer-based backtester.
        
        Parameters
        ----------
        price_factor : Factor
            Price factor for entry/exit prices
        strategy_factor : Factor
            Strategy factor for ranking
        long_top_n : int, optional
            Number of top-ranked assets to go long. If None, uses top 20%.
        short_bottom_n : int, optional
            Number of bottom-ranked assets to go short. If None, uses bottom 20%.
        transaction_cost : Union[float, Tuple[float, float]], default (0.0003, 0.0003)
            Transaction cost rate(s)
        initial_capital : float, default 100000
            Initial capital
            
        Notes
        -----
        Capital allocation:
        - 50% capital → equally split among long_top_n assets
        - 50% capital → equally split among short_bottom_n assets
        
        Example: long_top_n=2, short_bottom_n=4
        - Each long position: 50% / 2 = 25% capital
        - Each short position: 50% / 4 = 12.5% capital
        """
        super().__init__(price_factor, strategy_factor, transaction_cost, initial_capital)
        
        if long_top_n is not None and long_top_n < 1:
            raise ValueError("long_top_n must be >= 1")
        if short_bottom_n is not None and short_bottom_n < 1:
            raise ValueError("short_bottom_n must be >= 1")
        
        self.long_top_n = long_top_n
        self.short_bottom_n = short_bottom_n
    
    def _calculate_target_holdings(self, factors: pd.Series) -> pd.Series:
        """
        Calculate target holdings using fixed-count selection.
        
        Strategy:
        - Top N assets: 50% capital, equal-weighted longs
        - Bottom N assets: 50% capital, equal-weighted shorts
        - Middle: no position
        
        Daily rebalance: recalculates target holdings based on new factor rankings.
        Only trades the difference between current and target positions.
        """
        if len(factors) < 2:
            return pd.Series(0.0, index=factors.index)
        
        sorted_factors = factors.sort_values(ascending=False)
        n_assets = len(sorted_factors)
        
        # Use fixed counts or default to 20% of assets
        n_long = self.long_top_n if self.long_top_n is not None else max(1, int(n_assets * 0.2))
        n_short = self.short_bottom_n if self.short_bottom_n is not None else max(1, int(n_assets * 0.2))
        
        # Cap at available assets
        n_long = min(n_long, n_assets)
        n_short = min(n_short, n_assets)
        
        long_symbols = sorted_factors.index[:n_long]
        short_symbols = sorted_factors.index[-n_short:]
        
        target_holdings = pd.Series(0.0, index=factors.index)
        
        long_capital = self.portfolio.total_value * 0.5
        short_capital = self.portfolio.total_value * 0.5
        
        if n_long > 0:
            target_holdings.loc[long_symbols] = long_capital / n_long
        
        if n_short > 0:
            target_holdings.loc[short_symbols] = -short_capital / n_short
        
        return target_holdings
    
    def __repr__(self):
        """Professional representation."""
        history = self.portfolio.get_history_df()
        long_str = f"{self.long_top_n}" if self.long_top_n else "20%"
        short_str = f"{self.short_bottom_n}" if self.short_bottom_n else "20%"
        if not history.empty:
            days = len(history)
            start_date = history.index[0].strftime('%Y-%m-%d')
            end_date = history.index[-1].strftime('%Y-%m-%d')
            return (f"LayerBacktester(strategy={self.strategy_factor.name}, "
                   f"long={long_str}, short={short_str}, "
                   f"period={start_date} to {end_date}, days={days})")
        else:
            return (f"LayerBacktester(strategy={self.strategy_factor.name}, "
                   f"long={long_str}, short={short_str})")


def backtest_layer(
    price_factor: 'Factor',
    strategy_factor: 'Factor',
    long_top_n: int = None,
    short_bottom_n: int = None,
    transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
    initial_capital: float = 100000,
    auto_run: bool = True
) -> LayerBacktester:
    """
    Convenience function for layer-based backtesting.
    
    Parameters
    ----------
    price_factor : Factor
        Price factor for entry/exit prices
    strategy_factor : Factor
        Strategy factor for ranking
    long_top_n : int, optional
        Number of top-ranked assets to go long. If None, uses top 20%.
    short_bottom_n : int, optional
        Number of bottom-ranked assets to go short. If None, uses bottom 20%.
    transaction_cost : Union[float, Tuple[float, float]], default (0.0003, 0.0003)
        Transaction cost rate(s)
    initial_capital : float, default 100000
        Initial capital
    auto_run : bool, default True
        Automatically run backtest
        
    Returns
    -------
    LayerBacktester
        Backtester instance
        
    Examples
    --------
    >>> from phandas import Panel, backtest_layer
    >>> panel = Panel.from_csv('data.csv')
    >>> 
    >>> # Example 1: Symmetric (long 2, short 2)
    >>> bt = backtest_layer(
    ...     price_factor=panel['open'],
    ...     strategy_factor=my_factor,
    ...     long_top_n=2,
    ...     short_bottom_n=2
    ... )
    >>> 
    >>> # Example 2: Asymmetric (long 2, short 4)
    >>> bt = backtest_layer(
    ...     price_factor=panel['open'],
    ...     strategy_factor=my_factor,
    ...     long_top_n=2,
    ...     short_bottom_n=4
    ... )
    >>> bt.plot_equity()
    
    Notes
    -----
    Capital allocation (market-neutral):
    - 50% capital → equally split among long_top_n assets
    - 50% capital → equally split among short_bottom_n assets
    - Net exposure = 0%
    
    Example: long_top_n=2, short_bottom_n=4, initial_capital=100k
    - Each long position: 50k / 2 = 25k (25% of portfolio)
    - Each short position: 50k / 4 = 12.5k (12.5% of portfolio)
    
    Daily rebalance:
    1. Rank all assets by factor values
    2. Select top N for long, bottom N for short
    3. Calculate target holdings (equal-weight within each side)
    4. Execute trades = target - current holdings
    5. Only trades the difference (not full liquidation)
    
    Comparison with standard backtest():
    - standard: All assets, factor-proportional weights
    - layer: Only top/bottom N, equal weights
    """
    bt = LayerBacktester(price_factor, strategy_factor, long_top_n, short_bottom_n,
                        transaction_cost, initial_capital)
    
    if auto_run:
        bt.run().calculate_metrics()
    
    return bt
