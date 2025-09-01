"""
Professional backtesting engine for factor strategies.

Efficient implementation with clean API.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .core import Factor


logger = logging.getLogger(__name__)

# Configure matplotlib
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Backtester:
    """
    Professional backtesting engine with unified Factor format.
    
    Implements dollar-neutral factor strategies using Factor objects.
    """
    
    def __init__(
        self,
        price_factor: 'Factor',
        strategy_factor: 'Factor',
        transaction_cost: float = 0.001,
        initial_capital: float = 100000
    ):
        """
        Initialize backtester with unified Factor format.
        
        Parameters
        ----------
        price_factor : Factor
            Price factor for entry/exit prices (e.g., open price)
        strategy_factor : Factor
            Strategy factor for position weights
        transaction_cost : float, default 0.001
            Transaction cost per trade (0.1%)
        initial_capital : float, default 100000
            Initial capital for backtesting
        """
        from .core import Factor
        
        if not isinstance(price_factor, Factor):
            raise TypeError("price_factor must be a Factor object")
        if not isinstance(strategy_factor, Factor):
            raise TypeError("strategy_factor must be a Factor object")
        
        self.price_factor = price_factor
        self.strategy_factor = strategy_factor
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        
        # Results
        self.daily_pnl = pd.Series(dtype=float)
        self.equity_curve = pd.Series(dtype=float)
        self.metrics = {}
    
    def run(self) -> 'Backtester':
        """
        Execute backtest using Factor objects.
        
        Returns
        -------
        Backtester
            Self for method chaining
        """
        # Get all available dates from both factors
        price_dates = set(self.price_factor.data['timestamp'])
        strategy_dates = set(self.strategy_factor.data['timestamp'])
        common_dates = sorted(price_dates & strategy_dates)
        
        if len(common_dates) < 2:
            raise ValueError("Insufficient overlapping dates for backtesting")
        
        # Find first valid date with complete data
        start_idx = self._find_start_date(common_dates)
        if start_idx >= len(common_dates) - 1:
            raise ValueError("Insufficient data for backtesting")
        
        # Run backtest
        pnl_data = []
        equity = self.initial_capital
        
        for i in range(start_idx + 1, len(common_dates) - 1):  # Need next day for exit price
            current_date = common_dates[i]
            prev_date = common_dates[i - 1]
            next_date = common_dates[i + 1]
            
            try:
                # Get strategy factors from previous date (T-1)
                prev_strategy = self._get_factor_data(self.strategy_factor, prev_date)
                
                # Get entry price from current date (T) and exit price from next date (T+1)
                entry_prices = self._get_factor_data(self.price_factor, current_date)
                exit_prices = self._get_factor_data(self.price_factor, next_date)
                
                # Find common symbols with valid data
                common_symbols = (set(prev_strategy.index) & 
                                set(entry_prices.index) & 
                                set(exit_prices.index))
                
                if len(common_symbols) == 0:
                    continue
                
                # Filter to common symbols
                strategy_values = prev_strategy.loc[list(common_symbols)]
                entry_price_values = entry_prices.loc[list(common_symbols)]
                exit_price_values = exit_prices.loc[list(common_symbols)]
                
                # Remove any remaining NaN values
                valid_mask = (strategy_values.notna() & 
                            entry_price_values.notna() & 
                            exit_price_values.notna())
                
                if not valid_mask.any():
                    continue
                
                strategy_values = strategy_values[valid_mask]
                entry_price_values = entry_price_values[valid_mask]
                exit_price_values = exit_price_values[valid_mask]
                
                # Calculate positions and returns
                positions = self._calculate_positions(strategy_values)
                daily_return = self._calculate_return(positions, entry_price_values, exit_price_values)
                daily_pnl = equity * daily_return
                
                equity += daily_pnl
                pnl_data.append({
                    'date': current_date,
                    'pnl': daily_pnl,
                    'equity': equity,
                    'return': daily_return
                })
                
            except Exception as e:
                logger.warning(f"Error on {current_date}: {e}")
                continue
        
        # Store results
        if pnl_data:
            df = pd.DataFrame(pnl_data)
            self.daily_pnl = pd.Series(df['pnl'].values, index=df['date'])
            self.equity_curve = pd.Series(df['equity'].values, index=df['date'])
            
            # Add initial equity
            start_equity = pd.Series([self.initial_capital], 
                                   index=[df['date'].iloc[0] - pd.DateOffset(days=1)])
            self.equity_curve = pd.concat([start_equity, self.equity_curve])
        
        return self
    
    def calculate_metrics(self, risk_free_rate: float = 0.0) -> 'Backtester':
        """
        Calculate performance metrics.
        
        Parameters
        ----------
        risk_free_rate : float, default 0.0
            Risk-free rate for Sharpe ratio
            
        Returns
        -------
        Backtester
            Self for method chaining
        """
        if self.equity_curve.empty:
            self.metrics = {}
            return self
        
        # Basic metrics
        total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        days = max(days, 1)
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # Risk metrics
        daily_returns = self.equity_curve.pct_change().dropna()
        if not daily_returns.empty:
            annual_vol = daily_returns.std() * np.sqrt(365)
            sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        else:
            annual_vol = 0
            sharpe = 0
        
        # Drawdown
        rolling_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve / rolling_max - 1)
        max_drawdown = drawdown.min()
        
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar
        }
        
        return self
    
    def _get_factor_data(self, factor: 'Factor', date) -> pd.Series:
        """Extract factor data for a specific date."""
        date_data = factor.data[factor.data['timestamp'] == date]
        return date_data.set_index('symbol')['factor']
    
    def _find_start_date(self, dates) -> int:
        """Find first date with complete data in both factors."""
        for i, date in enumerate(dates):
            strategy_data = self._get_factor_data(self.strategy_factor, date)
            price_data = self._get_factor_data(self.price_factor, date)
            
            # Check if we have valid data for both factors
            if (len(strategy_data) > 0 and len(price_data) > 0 and
                not strategy_data.isna().all() and not price_data.isna().all()):
                return i
        raise ValueError("No valid factor data found")
    
    def _calculate_positions(self, factors: pd.Series) -> pd.Series:
        """Calculate dollar-neutral positions from factors."""
        # Demean factors
        demeaned = factors - factors.mean()
        
        # Normalize by sum of absolute values
        abs_sum = np.abs(demeaned).sum()
        
        # Check if factors are essentially constant (no meaningful signal)
        if abs_sum == 0 or abs_sum < 1e-10:  # Very small threshold for numerical precision
            return pd.Series(0.0, index=factors.index)
        
        return demeaned / abs_sum
    
    def _calculate_return(self, positions: pd.Series, entry_prices: pd.Series, exit_prices: pd.Series) -> float:
        """Calculate daily strategy return using entry and exit prices."""
        # Symbol returns from entry to exit prices
        symbol_returns = (exit_prices / entry_prices) - 1
        
        # Portfolio return
        portfolio_return = (positions * symbol_returns).sum()
        
        # Apply transaction cost
        return portfolio_return - self.transaction_cost
    
    def plot_equity(self, figsize: tuple = (12, 6)) -> 'Backtester':
        """
        Plot equity curve.
        
        Parameters
        ----------
        figsize : tuple, default (12, 6)
            Figure size
            
        Returns
        -------
        Backtester
            Self for method chaining
        """
        if self.equity_curve.empty:
            logger.warning("No equity data to plot")
            return self
        
        fig, ax = plt.subplots(figsize=figsize)
        self.equity_curve.plot(ax=ax, color='blue', linewidth=1.5)
        ax.set_title(f'Equity Curve ({self.strategy_factor.name})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity Value')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return self
    
    def summary(self) -> None:
        """Print performance summary."""
        if not self.metrics:
            print("No metrics available. Run calculate_metrics() first.")
            return
        
        print(f"\nBacktest Results: {self.strategy_factor.name}")
        print("=" * 50)
        
        if not self.equity_curve.empty:
            start = self.equity_curve.index[0].strftime('%Y-%m-%d')
            end = self.equity_curve.index[-1].strftime('%Y-%m-%d')
            print(f"Period: {start} to {end}")
            print(f"Initial Capital: ${self.initial_capital:,.0f}")
            print(f"Final Capital: ${self.equity_curve.iloc[-1]:,.0f}")
        
        print("\nPerformance Metrics:")
        print("-" * 30)
        for name, value in self.metrics.items():
            if 'return' in name or 'volatility' in name or 'drawdown' in name:
                print(f"{name.replace('_', ' ').title()}: {value:.2%}")
            else:
                print(f"{name.replace('_', ' ').title()}: {value:.3f}")
        
        # Assessment
        total_ret = self.metrics.get('total_return', 0)
        sharpe = self.metrics.get('sharpe_ratio', 0)
        
        print("\nAssessment:")
        status = "Profitable" if total_ret > 0 else "Loss-making"
        if sharpe > 1:
            risk_assessment = "Excellent"
        elif sharpe > 0.5:
            risk_assessment = "Good"
        elif sharpe > 0:
            risk_assessment = "Fair"
        else:
            risk_assessment = "Poor"
        
        print(f"Return: {status}")
        print(f"Risk-adjusted: {risk_assessment}")
        print("=" * 50)
    
    def __repr__(self):
        """Professional representation of Backtester."""
        if hasattr(self, 'daily_pnl') and not self.daily_pnl.empty:
            # After backtest
            days = len(self.daily_pnl)
            start_date = self.daily_pnl.index[0].strftime('%Y-%m-%d')
            end_date = self.daily_pnl.index[-1].strftime('%Y-%m-%d')
            return (f"Backtester(strategy={self.strategy_factor.name}, "
                   f"period={start_date} to {end_date}, days={days})")
        else:
            # Before backtest
            return (f"Backtester(strategy={self.strategy_factor.name}, "
                   f"price={self.price_factor.name}, cost={self.transaction_cost:.3%})")


def backtest(
    price_factor: 'Factor',
    strategy_factor: 'Factor',
    transaction_cost: float = 0.001,
    initial_capital: float = 100000,
    auto_run: bool = True
) -> Backtester:
    """
    Convenience function for quick backtesting with unified Factor format.
    
    Parameters
    ----------
    price_factor : Factor
        Price factor for entry/exit prices (e.g., open price)
    strategy_factor : Factor
        Strategy factor for position weights
    transaction_cost : float, default 0.001
        Transaction cost per trade
    initial_capital : float, default 100000
        Initial capital
    auto_run : bool, default True
        Automatically run backtest and calculate metrics
        
    Returns
    -------
    Backtester
        Backtester instance
    """
    bt = Backtester(price_factor, strategy_factor, transaction_cost, initial_capital)
    
    if auto_run:
        bt.run().calculate_metrics()
    
    return bt
