"""Backtesting engine for factor strategies."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Union, Tuple, Dict, List, Optional
import logging
from scipy.stats import linregress, skew, kurtosis, norm

if TYPE_CHECKING:
    from .core import Factor

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class Portfolio:
    """Portfolio state and trade execution."""
    def __init__(self, initial_capital: float = 1000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.holdings = {}
        self.total_value = initial_capital
        self.history = []
        self.trade_log = []

    def update_market_value(self, date, prices: pd.Series):
        """Mark-to-market valuation."""
        holdings_value = 0.0
        self.holdings.clear()
        
        prices_dict = prices.to_dict()
        for symbol, qty in self.positions.items():
            if symbol in prices_dict:
                value = qty * prices_dict[symbol]
                self.holdings[symbol] = value
                holdings_value += value
        
        self.total_value = self.cash + holdings_value
        self.history.append({'date': date, 'total_value': self.total_value})

    def execute_trade(self, symbol: str, quantity: float, price: float, 
                     transaction_cost_rates: Union[float, Tuple[float, float]], 
                     trade_date: pd.Timestamp):
        """Execute trade with separate buy/sell cost rates."""
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
    
    def _build_datetime_df(self, data_list: list) -> pd.DataFrame:
        """Convert list of dicts to DataFrame with datetime index."""
        if not data_list:
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')
    
    def get_history_df(self) -> pd.DataFrame:
        """Return portfolio history as DataFrame."""
        return self._build_datetime_df(self.history)

    def get_trade_log_df(self) -> pd.DataFrame:
        """Return trade log as DataFrame."""
        return self._build_datetime_df(self.trade_log)


class Backtester:
    """Backtesting engine for factor strategies."""
    
    def __init__(
        self,
        price_factor: 'Factor',
        strategy_factor: 'Factor',
        transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
        initial_capital: float = 1000,
        full_rebalance: bool = False,
        neutralization: str = "market"
    ):
        """
        Parameters
        ----------
        price_factor : Factor
            Price data for entry/exit
        strategy_factor : Factor
            Position weights
        transaction_cost : float or (buy_rate, sell_rate)
            Transaction cost rates
        initial_capital : float
        full_rebalance : bool
            If True: liquidate all daily (turnover ~200%)
            If False: incremental rebalancing (turnover ~20-50%)
        neutralization : str
            "market": normalize to dollar-neutral weights (default)
            "none": use strategy_factor directly (must be dollar-neutral)
        """
        self.price_factor = price_factor
        self.strategy_factor = strategy_factor
        self.full_rebalance = full_rebalance
        self.neutralization = neutralization.lower()
        
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
                
                if self.full_rebalance:
                    for symbol in list(self.portfolio.positions.keys()):
                        if symbol in current_prices.index:
                            quantity = -self.portfolio.positions[symbol]
                            price = current_prices.loc[symbol]
                            self.portfolio.execute_trade(symbol, quantity, price,
                                                        self.transaction_cost_rates, current_date)
                    
                    self.portfolio.update_market_value(current_date, current_prices)
                
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
        
        t = np.arange(len(equity_curve))
        slope, intercept, r_value, _, _ = linregress(t, equity_curve.values)
        linearity = r_value ** 2
        
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(365)
            sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        else:
            sortino = 0
        
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        drawdown_periods = self._identify_drawdown_periods(equity_curve)
        
        var_95 = daily_returns.quantile(0.05) if not daily_returns.empty else 0
        cvar = daily_returns[daily_returns <= var_95].mean() if not daily_returns.empty and (daily_returns <= var_95).any() else 0
        
        psr = self._calculate_psr(daily_returns) if not daily_returns.empty else 0
        
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'linearity': linearity,
            'drawdown_periods': drawdown_periods,
            'var_95': var_95,
            'cvar': cvar,
            'psr': psr,
        }
        
        return self
    
    def _calculate_psr(self, daily_returns: pd.Series, sr_benchmark: float = 0.0) -> float:
        """Probabilistic Sharpe Ratio: statistical confidence (0-1) of beating benchmark Sharpe."""
        if len(daily_returns) < 2:
            return 0.0
        
        sr_obs = (daily_returns.mean() * 365) / (daily_returns.std() * np.sqrt(365)) \
                 if daily_returns.std() > 0 else 0
        
        g3 = skew(daily_returns)
        g4 = kurtosis(daily_returns, fisher=False)
        T = len(daily_returns)
        adjustment = np.sqrt(1 - g3 * sr_obs + ((g4 - 1) / 4) * sr_obs ** 2)
        psr_stat = (sr_obs - sr_benchmark) / adjustment * np.sqrt(T / 365)    
        psr = norm.cdf(psr_stat)
        return float(np.clip(psr, 0.0, 1.0))

    def _identify_drawdown_periods(self, equity_curve: pd.Series) -> List[Dict]:
        """Identify drawdown periods (start/end dates and depth)."""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve / rolling_max - 1)
        
        in_drawdown = False
        periods = []
        start_date = None
        
        for date, dd_value in drawdown.items():
            if dd_value < -1e-6:
                if not in_drawdown:
                    in_drawdown = True
                    start_date = date
            else:
                if in_drawdown:
                    end_date = date
                    duration = (end_date - start_date).days
                    depth = drawdown.loc[start_date:end_date].min()
                    
                    periods.append({
                        'start': start_date.strftime('%Y-%m-%d'),
                        'end': end_date.strftime('%Y-%m-%d'),
                        'depth': depth,
                        'duration_days': duration,
                    })
                    in_drawdown = False
        
        if in_drawdown:
            end_date = equity_curve.index[-1]
            duration = (end_date - start_date).days
            depth = drawdown.loc[start_date:end_date].min()
            periods.append({
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'depth': depth,
                'duration_days': duration,
            })
        
        return sorted(periods, key=lambda x: x['depth'])

    def _build_date_cache(self, factor: 'Factor') -> dict:
        """Cache factor data by date (only complete cross-sections)."""
        cache = {}
        for date, group in factor.data.groupby('timestamp', sort=False):
            series = group.set_index('symbol')['factor']
            if not series.isna().any():
                cache[date] = series
        return cache
    
    def _get_factor_data(self, factor: 'Factor', date) -> pd.Series:
        """Get factor data for date from cache."""
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
        """Calculate target dollar-neutral holdings."""
        if self.neutralization == "none":
            return factors * self.portfolio.total_value
        
        demeaned = factors - factors.mean()
        abs_sum = np.abs(demeaned).sum()
        
        if abs_sum < 1e-10:
            return pd.Series(0.0, index=factors.index)
        
        weights = demeaned / abs_sum
        return weights * self.portfolio.total_value
    
    def _generate_orders(self, target_holdings: pd.Series, prices: pd.Series) -> pd.Series:
        """Generate trade orders based on target vs current holdings."""
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
    
    def get_daily_returns(self) -> pd.Series:
        """Daily returns series."""
        history = self.portfolio.get_history_df()
        if history.empty or len(history) < 2:
            return pd.Series(dtype=float)
        equity = history['total_value']
        return equity.pct_change(fill_method=None).dropna()

    def get_equity_curve(self) -> pd.Series:
        """Equity curve normalized to 1.0."""
        history = self.portfolio.get_history_df()
        if history.empty:
            return pd.Series(dtype=float)
        equity = history['total_value']
        return equity / equity.iloc[0]

    def summary(self) -> str:
        """Return concise performance summary."""
        if not self.metrics:
            return "No metrics available."
        
        equity_curve = self.portfolio.get_history_df()['total_value']
        summary_lines = [f"Strategy: {self.strategy_factor.name}"]
        
        if not equity_curve.empty:
            start = equity_curve.index[0].strftime('%Y-%m-%d')
            end = equity_curve.index[-1].strftime('%Y-%m-%d')
            
            turnover_df = self.get_daily_turnover_df()
            avg_turnover = turnover_df['turnover'].mean() * 365 if not turnover_df.empty else 0
            
            psr = self.metrics.get('psr', 0)
            psr_label = '[High]' if psr > 0.75 else '[Medium]' if psr > 0.50 else '[Low]'
            
            summary_lines.extend([
                f"Period: {start} to {end}",
                f"Total Return: {self.metrics.get('total_return', 0):.2%}",
                f"Annual Return: {self.metrics.get('annual_return', 0):.2%}",
                f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}",
                f"PSR: {psr:.1%} {psr_label}",
                f"Sortino Ratio: {self.metrics.get('sortino_ratio', 0):.2f}",
                f"Calmar Ratio: {self.metrics.get('calmar_ratio', 0):.2f}",
                f"Linearity: {self.metrics.get('linearity', 0):.4f}",
                f"Max Drawdown: {self.metrics.get('max_drawdown', 0):.2%}",
                f"VaR 95%: {self.metrics.get('var_95', 0):.2%}",
                f"CVaR: {self.metrics.get('cvar', 0):.2%}",
                f"Avg. Annual Turnover: {avg_turnover:.2%}"
            ])
        
        return "\n".join(summary_lines)

    def print_summary(self) -> 'Backtester':
        """Print performance summary. Returns self for chaining."""
        print(self.summary())
        return self
    
    def print_drawdown_periods(self, top_n: int = 5) -> 'Backtester':
        """Print top N drawdown periods (sorted by depth). Returns self for chaining."""
        drawdown_periods = self.metrics.get('drawdown_periods', [])
        
        if not drawdown_periods:
            print("\nNo significant drawdown periods detected.")
            return self
        
        periods_to_show = drawdown_periods[:top_n]
        total_periods = len(drawdown_periods)
        
        print(f"\nTop {min(top_n, total_periods)} Drawdown Periods (sorted by depth):")
        print("-" * 70)
        for i, period in enumerate(periods_to_show, 1):
            print(f"{i}. {period['start']} → {period['end']} | "
                  f"Depth: {period['depth']:.2%} | Duration: {period['duration_days']} days")
        
        if total_periods > top_n:
            print(f"\n(Showing {top_n} of {total_periods} total drawdown periods)")
        
        return self
    
    def _calculate_benchmark_equity(self) -> pd.Series:
        """Equal-weight benchmark: buy all assets on first trading date, hold."""
        history = self.portfolio.get_history_df()
        if history.empty or len(history) < 2:
            return pd.Series(dtype=float)
        
        first_date = history.index[1]
        if first_date not in self._price_cache:
            return pd.Series(dtype=float)
        
        prices_first = self._price_cache[first_date]
        if prices_first.empty:
            return pd.Series(dtype=float)
        
        alloc_per_asset = self.portfolio.initial_capital / len(prices_first)
        holdings = {s: alloc_per_asset / prices_first[s] for s in prices_first.index}
        
        values, dates = [], []
        for date in sorted(self._price_cache.keys()):
            if date < first_date:
                continue
            prices = self._price_cache[date]
            if prices.empty:
                continue
            values.append(sum(holdings[s] * prices[s] for s in holdings if s in prices.index))
            dates.append(date)
        
        return pd.Series(values, index=pd.DatetimeIndex(dates))
    
    def plot_equity(self, figsize: tuple = (12, 7), show_summary: bool = True, show_benchmark: bool = True) -> 'Backtester':
        """Plot equity curve with drawdown, turnover subplots and optional benchmark.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        show_summary : bool
            Show summary statistics
        show_benchmark : bool
            Show equal-weight benchmark comparison
        """
        history = self.portfolio.get_history_df()
        if history.empty:
            logger.warning("No equity data to plot")
            return self
        
        equity_curve = history['total_value']
        equity_norm = equity_curve / equity_curve.iloc[0]
        rolling_max = equity_norm.cummax()
        drawdown = equity_norm / rolling_max - 1.0
        
        benchmark_series = None
        benchmark_norm = None
        if show_benchmark:
            benchmark_series = self._calculate_benchmark_equity()
            if not benchmark_series.empty and len(benchmark_series) > 0:
                benchmark_norm = benchmark_series / benchmark_series.iloc[0]
        
        turnover_df = self.get_daily_turnover_df()
        
        plt.style.use('default')
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        ax = fig.add_subplot(gs[0, 0])
        ax_dd = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_to = fig.add_subplot(gs[2, 0], sharex=ax)
        
        ax.set_facecolor('#fcfcfc')
        y_min = equity_curve.min()
        self._plot_equity_line(ax, equity_curve, y_min)
        
        if benchmark_norm is not None and len(benchmark_norm) > 0:
            benchmark_abs = benchmark_norm * self.portfolio.initial_capital
            y_min = min(y_min, benchmark_abs.min())
            ax.plot(benchmark_norm.index, benchmark_abs, color='#f97316', linewidth=1.2, 
                   alpha=0.8, linestyle='--')
        
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
        self._plot_drawdown(ax_dd, drawdown)
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
    
    def _plot_equity_line(self, ax, equity_series: pd.Series, y_min: float):
        """Plot equity curve with layered fill."""
        layers = [
            (0.22, '#2563eb'),
            (0.12, '#3b82f6'),
            (0.06, '#60a5fa'),
            (0.03, '#93c5fd'),
            (0.015, '#dbeafe'),
        ]
        for alpha, color in layers:
            ax.fill_between(equity_series.index, y_min, equity_series,
                           alpha=alpha, color=color, interpolate=True)
        ax.plot(equity_series.index, equity_series, color='#1e40af', linewidth=1.05, alpha=0.95)
    
    def _plot_drawdown(self, ax, drawdown_series: pd.Series):
        """Plot drawdown area."""
        ax.fill_between(drawdown_series.index, 0, drawdown_series, 
                       color='#ef4444', alpha=0.35, step='pre')
        ax.plot(drawdown_series.index, drawdown_series, color='#991b1b', linewidth=0.8)
    
    
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

    def __add__(self, other: 'Backtester') -> 'CombinedBacktester':
        """Combine two strategies with equal weights (50-50 split)."""
        return CombinedBacktester([self, other], weights=[0.5, 0.5])

    def __mul__(self, weight: float) -> 'CombinedBacktester':
        """Weight single strategy: bt * 0.7 for 70% allocation."""
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
        return CombinedBacktester([self], weights=[weight])


class CombinedBacktester:
    """Multi-strategy portfolio combiner."""

    def __init__(self, backtests: List[Backtester], weights: List[float]):
        """
        Combine multiple backtesting results.

        Parameters
        ----------
        backtests : List[Backtester]
            List of Backtester objects
        weights : List[float]
            Portfolio weights (must sum to 1.0)
        """
        if len(backtests) != len(weights):
            raise ValueError("Number of weights must match number of backtests")
        if not np.isclose(sum(weights), 1.0, atol=1e-10):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights):.6f}")
        
        self.backtests = backtests
        self.weights = np.array(weights)
        self.metrics = {}
        self.calculate_metrics()

    def calculate_metrics(self, risk_free_rate: float = 0.0) -> 'CombinedBacktester':
        """Calculate portfolio-level performance metrics."""
        port_returns = self.get_portfolio_returns()
        if port_returns.empty or len(port_returns) < 2:
            self.metrics = {}
            return self
        
        total_return = (1 + port_returns).prod() - 1
        
        days = len(port_returns)
        if days > 1:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0
        
        annual_vol = port_returns.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        equity = (1 + port_returns).cumprod()
        rolling_max = equity.expanding().max()
        drawdown = equity / rolling_max - 1
        max_drawdown = drawdown.min()
        
        t = np.arange(len(equity))
        slope, intercept, r_value, _, _ = linregress(t, equity.values)
        linearity = r_value ** 2
        
        downside_returns = port_returns[port_returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(252)
            sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        else:
            sortino = 0
        
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        drawdown_periods = self._identify_drawdown_periods(equity)
        
        var_95 = port_returns.quantile(0.05) if not port_returns.empty else 0
        cvar = port_returns[port_returns <= var_95].mean() if not port_returns.empty and (port_returns <= var_95).any() else 0
        
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'linearity': linearity,
            'drawdown_periods': drawdown_periods,
            'var_95': var_95,
            'cvar': cvar,
        }
        
        return self

    def _get_aligned_returns(self) -> pd.DataFrame:
        """Get aligned daily returns for all strategies."""
        returns_dict = {}
        for i, bt in enumerate(self.backtests):
            returns_dict[f'strategy_{i}'] = bt.get_daily_returns()
        
        df = pd.DataFrame(returns_dict)
        return df.dropna(how='all')

    def _identify_drawdown_periods(self, equity_series: pd.Series) -> List[Dict]:
        """Identify drawdown periods (start/end dates and depth)."""
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series / rolling_max - 1)
        
        in_drawdown = False
        periods = []
        start_date = None
        
        for i, (date, dd_value) in enumerate(drawdown.items()):
            if dd_value < -1e-6:
                if not in_drawdown:
                    in_drawdown = True
                    start_date = i
            else:
                if in_drawdown:
                    end_date = i
                    start_label = drawdown.index[start_date].strftime('%Y-%m-%d')
                    end_label = drawdown.index[end_date].strftime('%Y-%m-%d')
                    duration = (drawdown.index[end_date] - drawdown.index[start_date]).days
                    depth = drawdown.iloc[start_date:end_date + 1].min()
                    
                    periods.append({
                        'start': start_label,
                        'end': end_label,
                        'depth': depth,
                        'duration_days': duration,
                    })
                    in_drawdown = False
        
        if in_drawdown:
            end_date = len(drawdown) - 1
            start_label = drawdown.index[start_date].strftime('%Y-%m-%d')
            end_label = drawdown.index[end_date].strftime('%Y-%m-%d')
            duration = (drawdown.index[end_date] - drawdown.index[start_date]).days
            depth = drawdown.iloc[start_date:end_date + 1].min()
            periods.append({
                'start': start_label,
                'end': end_label,
                'depth': depth,
                'duration_days': duration,
            })
        
        return sorted(periods, key=lambda x: x['depth'])

    def get_portfolio_returns(self) -> pd.Series:
        """Weighted portfolio daily returns."""
        returns_df = self._get_aligned_returns()
        if returns_df.empty:
            return pd.Series(dtype=float)
        
        portfolio_returns = (returns_df * self.weights).sum(axis=1)
        return portfolio_returns

    def get_portfolio_equity(self) -> pd.Series:
        """Portfolio equity curve."""
        port_returns = self.get_portfolio_returns()
        if port_returns.empty:
            return pd.Series(dtype=float)
        
        equity = (1 + port_returns).cumprod()
        initial_capital = sum([bt.portfolio.initial_capital * w 
                              for bt, w in zip(self.backtests, self.weights)])
        return equity * initial_capital

    def correlation_matrix(self) -> pd.DataFrame:
        """Correlation matrix of strategy returns."""
        returns_df = self._get_aligned_returns()
        if returns_df.empty or len(returns_df) < 2:
            logger.warning("Insufficient data for correlation calculation")
            return pd.DataFrame()
        
        corr = returns_df.corr()
        
        rename_map = {f'strategy_{i}': self.backtests[i].strategy_factor.name 
                     for i in range(len(self.backtests))}
        corr = corr.rename(index=rename_map, columns=rename_map)
        
        return corr

    def print_correlation_matrix(self) -> 'CombinedBacktester':
        """Print correlation matrix. Returns self for chaining."""
        corr = self.correlation_matrix()
        if not corr.empty:
            print("\nCorrelation Matrix:")
            print(corr.to_string(max_cols=None, max_rows=None, float_format=lambda x: f'{x:.6f}'))
        return self

    def summary(self, compact: bool = False) -> str:
        """Portfolio performance summary."""
        lines = []
        
        strategy_names = ", ".join([bt.strategy_factor.name for bt in self.backtests])
        lines.append(f"Strategy: {strategy_names}")
        
        equity = self.get_portfolio_equity()
        if not equity.empty and len(equity) > 0:
            start_date = equity.index[0].strftime('%Y-%m-%d')
            end_date = equity.index[-1].strftime('%Y-%m-%d')
            lines.append(f"Period: {start_date} to {end_date}")
        
        if self.metrics:
            lines.append(f"Total Return: {self.metrics.get('total_return', 0):.2%}")
            lines.append(f"Annual Return: {self.metrics.get('annual_return', 0):.2%}")
            lines.append(f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
            lines.append(f"Sortino Ratio: {self.metrics.get('sortino_ratio', 0):.2f}")
            lines.append(f"Calmar Ratio: {self.metrics.get('calmar_ratio', 0):.2f}")
            lines.append(f"Linearity: {self.metrics.get('linearity', 0):.4f}")
            lines.append(f"Max Drawdown: {self.metrics.get('max_drawdown', 0):.2%}")
            lines.append(f"VaR 95%: {self.metrics.get('var_95', 0):.2%}")
            lines.append(f"CVaR: {self.metrics.get('cvar', 0):.2%}")
        
        lines.append("")
        lines.append("Strategy Weights:")
        for bt, w in zip(self.backtests, self.weights):
            lines.append(f"  {bt.strategy_factor.name}: {w*100:.1f}%")
        
        corr = self.correlation_matrix()
        if not corr.empty and len(corr) > 1:
            lines.append("")
            lines.append("Correlation Matrix:")
            corr_str = corr.to_string(max_cols=None, max_rows=None, float_format=lambda x: f'{x:.4f}')
            lines.extend(corr_str.split('\n'))
        
        return "\n".join(lines)

    def print_summary(self) -> 'CombinedBacktester':
        """Print portfolio summary. Returns self for chaining."""
        print(self.summary())
        return self
    
    def print_drawdown_periods(self, top_n: int = 5) -> 'CombinedBacktester':
        """Print top N drawdown periods. Returns self for chaining."""
        drawdown_periods = self.metrics.get('drawdown_periods', [])
        
        if not drawdown_periods:
            print("\nNo significant drawdown periods detected.")
            return self
        
        periods_to_show = drawdown_periods[:top_n]
        total_periods = len(drawdown_periods)
        
        print(f"\nTop {min(top_n, total_periods)} Drawdown Periods (sorted by depth):")
        print("-" * 70)
        for i, period in enumerate(periods_to_show, 1):
            print(f"{i}. {period['start']} → {period['end']} | "
                  f"Depth: {period['depth']:.2%} | Duration: {period['duration_days']} days")
        
        if total_periods > top_n:
            print(f"\n(Showing {top_n} of {total_periods} total drawdown periods)")
        
        return self
    
    def _plot_equity_line(self, ax, equity_series: pd.Series, y_min: float):
        """Plot equity curve with layered fill."""
        layers = [
            (0.22, '#2563eb'),
            (0.12, '#3b82f6'),
            (0.06, '#60a5fa'),
            (0.03, '#93c5fd'),
            (0.015, '#dbeafe'),
        ]
        for alpha, color in layers:
            ax.fill_between(equity_series.index, y_min, equity_series,
                           alpha=alpha, color=color, interpolate=True)
        ax.plot(equity_series.index, equity_series, color='#1e40af', linewidth=1.05, alpha=0.95)
    
    def _plot_drawdown(self, ax, drawdown_series: pd.Series):
        """Plot drawdown area."""
        ax.fill_between(drawdown_series.index, 0, drawdown_series, 
                       color='#ef4444', alpha=0.35, step='pre')
        ax.plot(drawdown_series.index, drawdown_series, color='#991b1b', linewidth=0.8)
    
    def plot_equity(self, figsize: tuple = (12, 7), show_summary: bool = True) -> 'CombinedBacktester':
        """Plot combined portfolio equity curve with drawdown."""
        equity = self.get_portfolio_equity()
        if equity.empty or len(equity) < 2:
            logger.warning("No equity data to plot")
            return self
        
        equity_norm = equity / equity.iloc[0]
        rolling_max = equity_norm.cummax()
        drawdown = equity_norm / rolling_max - 1.0
        
        plt.style.use('default')
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax = fig.add_subplot(gs[0, 0])
        ax_dd = fig.add_subplot(gs[1, 0], sharex=ax)
        
        ax.set_facecolor('#fcfcfc')
        y_min = equity.min()
        self._plot_equity_line(ax, equity, y_min)
        
        strategy_names = ", ".join([bt.strategy_factor.name for bt in self.backtests])
        ax.set_title(f'Combined Portfolio Equity ({strategy_names})', 
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
        self._plot_drawdown(ax_dd, drawdown)
        ax_dd.set_ylabel('Drawdown', fontsize=10, color='#6b7280')
        ax_dd.set_xlabel('Date', fontsize=10.5, color='#6b7280')
        ax_dd.grid(True, alpha=0.12, color='#e5e7eb', linestyle='-', linewidth=0.4)
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax_dd.spines['top'].set_visible(False)
        ax_dd.spines['right'].set_visible(False)
        ax_dd.tick_params(axis='both', which='major', labelsize=9.0, colors='#6b7280', 
                          width=0.5, length=3)
        
        plt.setp(ax.get_xticklabels(), visible=False)
        
        plt.show()
        return self

    def __add__(self, other: Union[Backtester, 'CombinedBacktester']) -> 'CombinedBacktester':
        """Add another strategy to portfolio."""
        if isinstance(other, Backtester):
            combined_backtests = self.backtests + [other]
            n_total = len(combined_backtests)
            new_weights = np.concatenate([self.weights * (len(self.backtests) - 1) / len(self.backtests), 
                                          [1.0 / n_total]])
            new_weights = new_weights / new_weights.sum()
        elif isinstance(other, CombinedBacktester):
            combined_backtests = self.backtests + other.backtests
            new_weights = np.concatenate([self.weights, other.weights])
            new_weights = new_weights / new_weights.sum()
        else:
            raise TypeError("Can only add Backtester or CombinedBacktester")
        
        result = CombinedBacktester(combined_backtests, new_weights)
        return result

    def __mul__(self, scalar: float) -> 'CombinedBacktester':
        """Scale portfolio weight."""
        if not 0 <= scalar <= 1:
            raise ValueError("Scalar must be between 0 and 1")
        return CombinedBacktester(self.backtests, self.weights * scalar / self.weights.sum())

    def __repr__(self) -> str:
        """String representation."""
        names = [bt.strategy_factor.name for bt in self.backtests]
        return f"CombinedBacktester({', '.join(names)}, weights={list(np.round(self.weights, 3))})"


def backtest(
    price_factor: 'Factor',
    strategy_factor: 'Factor',
    transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
    initial_capital: float = 1000,
    full_rebalance: bool = False,
    neutralization: str = "market",
    auto_run: bool = True
) -> Backtester:
    """
    Quick backtesting convenience function.
    
    Parameters
    ----------
    price_factor : Factor
    strategy_factor : Factor
    transaction_cost : float or (buy_rate, sell_rate)
    initial_capital : float
    full_rebalance : bool
        If True: liquidate all daily (turnover ~200%)
        If False: incremental rebalancing (turnover ~20-50%)
    neutralization : str
        If "market": normalize to dollar-neutral weights (default)
        If "none": use strategy_factor directly (must be dollar-neutral)
    auto_run : bool
        Automatically run backtest and calculate metrics
    """
    bt = Backtester(price_factor, strategy_factor, transaction_cost, initial_capital, 
                   full_rebalance, neutralization)
    
    if auto_run:
        bt.run().calculate_metrics()
    
    return bt