"""
Professional backtesting engine for factor strategies.

Optimized for 3-column Factor format [timestamp, symbol, factor].
Supports dollar-neutral portfolio-based backtesting with dynamic rebalancing,
transaction cost modeling, and comprehensive performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Union, Tuple
import logging

if TYPE_CHECKING:
    from .core import Factor


logger = logging.getLogger(__name__)

# Configure matplotlib with fallback fonts for Chinese/English support
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class Portfolio:
    """Manages trading portfolio state, positions, and trade execution."""
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.holdings = {}
        self.total_value = initial_capital
        
        self.history = []
        self.trade_log = []

    def update_market_value(self, date, prices: pd.Series):
        """Mark-to-market portfolio valuation."""
        holdings_value = 0.0
        self.holdings.clear()
        
        prices_dict = prices.to_dict()
        for symbol, qty in self.positions.items():
            if symbol in prices_dict:
                value = qty * prices_dict[symbol]
                self.holdings[symbol] = value
                holdings_value += value
        
        previous_total_value = self.total_value
        self.total_value = self.cash + holdings_value
        
        self.history.append({
            'date': date,
            'total_value': self.total_value,
            'cash': self.cash,
            'holdings_value': holdings_value,
            'daily_pnl': self.total_value - previous_total_value,
        })

    def execute_trade(self, symbol: str, quantity: float, price: float, 
                     transaction_cost_rates: Union[float, Tuple[float, float]], 
                     trade_date: pd.Timestamp):
        """Execute trade with transaction costs (separate buy/sell rates)."""
        if isinstance(transaction_cost_rates, (list, tuple)):
            buy_cost_rate = transaction_cost_rates[0]
            sell_cost_rate = transaction_cost_rates[1]
        else:
            buy_cost_rate = sell_cost_rate = transaction_cost_rates
            
        trade_value = quantity * price
        cost = abs(trade_value) * (buy_cost_rate if quantity > 0 else sell_cost_rate)
        
        self.cash -= (trade_value + cost)
        
        new_quantity = self.positions.get(symbol, 0.0) + quantity
        
        if abs(new_quantity) < 1e-10:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_quantity
        
        self.trade_log.append({
            'date': trade_date,
            'symbol': symbol,
            'trade_value': trade_value,
            'cost': cost
        })
    
    def get_history_df(self) -> pd.DataFrame:
        """Return portfolio history as DataFrame."""
        if not self.history:
            return pd.DataFrame()
        df = pd.DataFrame(self.history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df

    def get_trade_log_df(self) -> pd.DataFrame:
        """Return trade log as DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        df = pd.DataFrame(self.trade_log)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df


class Backtester:
    """
    Backtesting engine with 3-column Factor format [timestamp, symbol, factor].
    
    Optimized for fast backtesting with dynamic rebalancing and metrics calculation.
    """
    
    def __init__(
        self,
        price_factor: 'Factor',
        strategy_factor: 'Factor',
        transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
        initial_capital: float = 100000
    ):
        """
        Parameters
        ----------
        price_factor : Factor
            Price factor for entry/exit prices
        strategy_factor : Factor
            Strategy factor for position weights
        transaction_cost : Union[float, Tuple[float, float]]
            Transaction cost rate(s) for buy/sell
        initial_capital : float
            Initial capital for backtesting
        """
        self.price_factor = price_factor
        self.strategy_factor = strategy_factor
        
        if isinstance(transaction_cost, (list, tuple)):
            self.transaction_cost_rates = tuple(transaction_cost)
        else:
            self.transaction_cost_rates = (transaction_cost, transaction_cost)
        
        self.portfolio = Portfolio(initial_capital)
        self.metrics = {}
        
        self._price_cache = self._build_date_cache(price_factor)
        self._strategy_cache = self._build_date_cache(strategy_factor)
    
    def run(self) -> 'Backtester':
        """Execute backtest using 3-column Factor format. Returns self for chaining."""
        price_dates = set(self.price_factor.data['timestamp'].unique())
        strategy_dates = set(self.strategy_factor.data['timestamp'].unique())
        common_dates = sorted(price_dates & strategy_dates)
        
        if len(common_dates) < 2:
            raise ValueError("Insufficient overlapping dates for backtesting")
        
        start_idx = self._find_start_date(common_dates)
        if start_idx >= len(common_dates):
            raise ValueError("Insufficient data for backtesting")
            
        initial_date = common_dates[start_idx] - pd.DateOffset(days=1)
        self.portfolio.history.append({
            'date': initial_date,
            'total_value': self.portfolio.initial_capital,
            'cash': self.portfolio.initial_capital,
            'holdings_value': 0,
        })

        for i in range(start_idx, len(common_dates)):
            current_date = common_dates[i]
            prev_date = common_dates[i - 1] if i > 0 else None
            
            try:
                current_prices = self._get_factor_data(self.price_factor, current_date)
                if current_prices.empty:
                    continue
                
                self.portfolio.update_market_value(current_date, current_prices)
                
                if not prev_date:
                    continue
                
                strategy_factors = self._get_factor_data(self.strategy_factor, prev_date)
                target_holdings = self._calculate_target_holdings(strategy_factors)
                orders = self._generate_orders(target_holdings, current_prices)
                
                for symbol, quantity in orders.items():
                    if symbol in current_prices.index:
                        price = current_prices.loc[symbol]
                        self.portfolio.execute_trade(symbol, quantity, price, 
                                                    self.transaction_cost_rates, current_date)
                
            except Exception as e:
                logger.warning(f"Error on {current_date}: {e}")
                continue
        
        return self
    
    def calculate_metrics(self, risk_free_rate: float = 0.0) -> 'Backtester':
        """Calculate performance metrics. Returns self for chaining."""
        history = self.portfolio.get_history_df()
        if history.empty or len(history) < 2:
            self.metrics = {}
            return self
        
        equity_curve = history['total_value']
        
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        days = max(days, 1)
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        daily_returns = equity_curve.pct_change(fill_method=None).dropna()
        if not daily_returns.empty:
            annual_vol = daily_returns.std() * np.sqrt(365)
            sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        else:
            annual_vol = 0
            sharpe = 0
        
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve / rolling_max - 1)
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
    
    # ==================== Internal Methods ====================
    
    def _build_date_cache(self, factor: 'Factor') -> dict:
        """Preprocess factor data into date-indexed cache for O(1) lookup."""
        cache = {}
        for date, group in factor.data.groupby('timestamp', sort=False):
            cache[date] = group.set_index('symbol')['factor'].dropna()
        return cache
    
    def _get_factor_data(self, factor: 'Factor', date) -> pd.Series:
        """Extract factor data for specific date using cache."""
        if date is None:
            return pd.Series(dtype=float)
        
        if factor is self.price_factor:
            return self._price_cache.get(date, pd.Series(dtype=float))
        else:
            return self._strategy_cache.get(date, pd.Series(dtype=float))
    
    def _find_start_date(self, dates) -> int:
        """Find first date with complete data in both factors."""
        for i, date in enumerate(dates):
            if i == 0:
                continue
            prev_date = dates[i-1]
            
            strategy_data = self._get_factor_data(self.strategy_factor, prev_date)
            price_data = self._get_factor_data(self.price_factor, date)
            
            if not strategy_data.empty and not price_data.empty:
                return i
        raise ValueError("No valid start date found with overlapping data")
    
    def _calculate_target_holdings(self, factors: pd.Series) -> pd.Series:
        """Calculate target dollar-neutral holdings from factors."""
        demeaned = factors - factors.mean()
        abs_sum = np.abs(demeaned).sum()
        
        if abs_sum < 1e-10:
            return pd.Series(0.0, index=factors.index)
        
        weights = demeaned / abs_sum
        return weights * self.portfolio.total_value

    def _generate_orders(self, target_holdings: pd.Series, prices: pd.Series) -> pd.Series:
        """Generate trade orders based on target holdings and current prices."""
        current_holdings = self.portfolio.holdings
        
        all_symbols = set(target_holdings.index) | set(current_holdings.keys())
        
        trade_quantities = {}
        prices_dict = prices.to_dict()
        
        for symbol in all_symbols:
            if symbol not in prices_dict:
                continue
            
            target_value = target_holdings.get(symbol, 0)
            current_value = current_holdings.get(symbol, 0)
            trade_value = target_value - current_value
            
            if abs(trade_value) > 1e-10:
                trade_quantities[symbol] = trade_value / prices_dict[symbol]
        
        return pd.Series(trade_quantities)
    
    # ==================== Visualization ====================
    
    def plot_equity(self, figsize: tuple = (12, 7), show_summary: bool = True) -> 'Backtester':
        """Plot equity curve with drawdown and turnover subplots."""
        history = self.portfolio.get_history_df()
        if history.empty:
            logger.warning("No equity data to plot")
            return self
        
        equity_curve = history['total_value']
        equity_norm = equity_curve / equity_curve.iloc[0]
        rolling_max = equity_norm.cummax()
        drawdown = equity_norm / rolling_max - 1.0
        
        turnover_df = self.get_daily_turnover_df()
        
        plt.style.use('default')
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        ax = fig.add_subplot(gs[0, 0])
        ax_dd = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_to = fig.add_subplot(gs[2, 0], sharex=ax)
        
        ax.set_facecolor('#fcfcfc')
        y_min = equity_curve.min()
        layers = [
            (0.22, '#2563eb'),
            (0.12, '#3b82f6'),
            (0.06, '#60a5fa'),
            (0.03, '#93c5fd'),
            (0.015, '#dbeafe'),
        ]
        for alpha, color in layers:
            ax.fill_between(equity_curve.index, y_min, equity_curve,
                           alpha=alpha, color=color, interpolate=True)
        ax.plot(equity_curve.index, equity_curve, color='#1e40af', linewidth=1.05, alpha=0.95)
        
        ax.set_title(f'Equity Curve ({self.strategy_factor.name})', 
                    fontsize=12.5, fontweight='400', color='#1f2937', pad=14)
        ax.set_ylabel('Equity Value', fontsize=10.5, color='#6b7280')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.grid(True, alpha=0.15, color='#e5e7eb', linestyle='-', linewidth=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=9.5, colors='#6b7280', 
                      width=0.5, length=3)
        
        if show_summary:
            summary_text = self.summary()
            ax.text(0.015, 0.98, summary_text, transform=ax.transAxes, fontsize=9.5, 
                    verticalalignment='top', horizontalalignment='left', color='#374151',
                    bbox=dict(boxstyle='round,pad=0.75', facecolor='white', 
                             edgecolor='#e5e7eb', alpha=0.96, linewidth=1))
        
        ax_dd.set_facecolor('#ffffff')
        ax_dd.fill_between(drawdown.index, 0, drawdown, color='#ef4444', alpha=0.35, step='pre')
        ax_dd.plot(drawdown.index, drawdown, color='#991b1b', linewidth=0.8)
        ax_dd.set_ylabel('Drawdown', fontsize=10, color='#6b7280')
        ax_dd.grid(True, alpha=0.12, color='#e5e7eb', linestyle='-', linewidth=0.4)
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax_dd.spines['top'].set_visible(False)
        ax_dd.spines['right'].set_visible(False)
        ax_dd.tick_params(axis='both', which='major', labelsize=9.0, colors='#6b7280', 
                          width=0.5, length=3)
        
        ax_to.set_facecolor('#ffffff')
        if not turnover_df.empty:
            ax_to.plot(turnover_df.index, turnover_df['turnover'], color='#10b981', 
                      linewidth=0.8, alpha=0.8)
            ax_to.set_ylabel('Turnover', fontsize=10, color='#6b7280')
            ax_to.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
            ax_to.grid(True, alpha=0.12, color='#e5e7eb', linestyle='-', linewidth=0.4)
        else:
            ax_to.text(0.5, 0.5, 'No Turnover Data', transform=ax_to.transAxes, 
                      ha='center', va='center', fontsize=10, color='#9ca3af')
        
        ax_to.set_xlabel('Date', fontsize=10.5, color='#6b7280')
        ax_to.spines['top'].set_visible(False)
        ax_to.spines['right'].set_visible(False)
        ax_to.tick_params(axis='both', which='major', labelsize=9.0, colors='#6b7280', 
                          width=0.5, length=3)
        
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax_dd.get_xticklabels(), visible=False)
        
        plt.show()
        return self
    
    def summary(self) -> str:
        """Return concise performance summary for plotting."""
        if not self.metrics:
            return "No metrics available."
        
        equity_curve = self.portfolio.get_history_df()['total_value']
        summary_lines = [f"Strategy: {self.strategy_factor.name}"]
        
        if not equity_curve.empty:
            start = equity_curve.index[0].strftime('%Y-%m-%d')
            end = equity_curve.index[-1].strftime('%Y-%m-%d')
            
            turnover_df = self.get_daily_turnover_df()
            avg_turnover = turnover_df['turnover'].mean() * 365 if not turnover_df.empty else 0
            
            summary_lines.extend([
                f"Period: {start} to {end}",
                f"Total Return: {self.metrics.get('total_return', 0):.2%}",
                f"Annual Return: {self.metrics.get('annual_return', 0):.2%}",
                f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}",
                f"Max Drawdown: {self.metrics.get('max_drawdown', 0):.2%}",
                f"Avg. Annual Turnover: {avg_turnover:.2%}"
            ])
        
        return "\n".join(summary_lines)

    def print_summary(self) -> 'Backtester':
        """Print performance summary to console."""
        print(self.summary())
        return self

    def get_daily_turnover_df(self) -> pd.DataFrame:
        """
        Calculate daily portfolio turnover.
        
        Turnover = total absolute dollar trades / daily NAV
        """
        trade_log_df = self.portfolio.get_trade_log_df()
        history_df = self.portfolio.get_history_df()
        
        if trade_log_df.empty or history_df.empty:
            return pd.DataFrame()
            
        daily_trade_value = trade_log_df['trade_value'].abs().groupby(level='date').sum()
        
        daily_nav = history_df['total_value']
        
        combined = pd.DataFrame({
            'daily_trade_value': daily_trade_value,
            'daily_nav': daily_nav
        }).dropna()
        
        if combined.empty:
            return pd.DataFrame()
            
        combined['turnover'] = combined['daily_trade_value'] / combined['daily_nav']
        return combined[['turnover']]
    
    
    def __repr__(self):
        """String representation of Backtester."""
        history = self.portfolio.get_history_df()
        if not history.empty:
            days = len(history)
            start_date = history.index[0].strftime('%Y-%m-%d')
            end_date = history.index[-1].strftime('%Y-%m-%d')
            return (f"Backtester(strategy={self.strategy_factor.name}, "
                   f"period={start_date} to {end_date}, days={days})")
        else:
            return (f"Backtester(strategy={self.strategy_factor.name}, "
                   f"price={self.price_factor.name}, cost={self.transaction_cost_rates[0]:.3%})")


def backtest(
    price_factor: 'Factor',
    strategy_factor: 'Factor',
    transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
    initial_capital: float = 100000,
    auto_run: bool = True
) -> Backtester:
    """
    Convenience function for quick backtesting.
    
    Parameters
    ----------
    price_factor : Factor
    strategy_factor : Factor
    transaction_cost : Union[float, Tuple[float, float]]
        Single value or (buy_cost_rate, sell_cost_rate)
    initial_capital : float
    auto_run : bool
        Automatically run backtest and calculate metrics
    """
    bt = Backtester(price_factor, strategy_factor, transaction_cost, initial_capital)
    
    if auto_run:
        bt.run().calculate_metrics()
    
    return bt