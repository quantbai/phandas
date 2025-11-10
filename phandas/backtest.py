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

_DATE_FORMAT = '%Y-%m-%d'

_PLOT_COLORS = {
    'equity_fill': [(0.22, '#2563eb'), (0.12, '#3b82f6'), (0.06, '#60a5fa'), (0.03, '#93c5fd'), (0.015, '#dbeafe')],
    'equity_line': '#1e40af',
    'benchmark_line': '#f97316',
    'drawdown_fill': '#ef4444',
    'drawdown_line': '#991b1b',
    'background': '#fcfcfc',
    'white': '#ffffff',
    'text': '#1f2937',
    'text_light': '#6b7280',
    'text_muted': '#9ca3af',
    'text_info': '#374151',
    'grid': '#e5e7eb',
    'turnover_line': '#10b981',
}

_PLOT_STYLES = {
    'title_size': 12.5,
    'ylabel_size': 10.5,
    'xlabel_size': 10.5,
    'label_size': 9.5,
    'small_label_size': 9.0,
    'grid_alpha': 0.15,
    'grid_width': 0.4,
    'grid_alpha_secondary': 0.12,
    'spine_width': 0.5,
    'tick_length': 3,
    'linewidth': 1.05,
    'benchmark_linewidth': 1.2,
    'benchmark_alpha': 0.8,
    'thin_linewidth': 0.8,
    'line_alpha': 0.95,
    'box_alpha': 0.96,
    'fill_alpha': 0.35,
}


def _plot_equity_line(ax, equity_series: pd.Series, y_min: float) -> None:
    """Layered equity curve visualization."""
    for alpha, color in _PLOT_COLORS['equity_fill']:
        ax.fill_between(equity_series.index, y_min, equity_series, alpha=alpha, color=color, interpolate=True)
    ax.plot(equity_series.index, equity_series, color=_PLOT_COLORS['equity_line'], 
           linewidth=_PLOT_STYLES['linewidth'], alpha=_PLOT_STYLES['line_alpha'])


def _plot_drawdown(ax, drawdown_series: pd.Series) -> None:
    """Drawdown visualization."""
    ax.fill_between(drawdown_series.index, 0, drawdown_series, 
                   color=_PLOT_COLORS['drawdown_fill'], alpha=_PLOT_STYLES['fill_alpha'], step='pre')
    ax.plot(drawdown_series.index, drawdown_series, color=_PLOT_COLORS['drawdown_line'], 
           linewidth=_PLOT_STYLES['thin_linewidth'])


def _identify_drawdown_periods(equity_series: pd.Series) -> List[Dict]:
    """Identify drawdown periods with depth and duration."""
    rolling_max = equity_series.expanding().max()
    drawdown = equity_series / rolling_max - 1
    
    in_drawdown = False
    periods = []
    start_idx = None
    
    for i, (date, dd_value) in enumerate(drawdown.items()):
        if dd_value < -1e-6:
            if not in_drawdown:
                in_drawdown = True
                start_idx = i
        else:
            if in_drawdown:
                end_idx = i
                periods.append({
                    'start': drawdown.index[start_idx].strftime(_DATE_FORMAT),
                    'end': drawdown.index[end_idx].strftime(_DATE_FORMAT),
                    'depth': drawdown.iloc[start_idx:end_idx + 1].min(),
                    'duration_days': (drawdown.index[end_idx] - drawdown.index[start_idx]).days,
                })
                in_drawdown = False
    
    if in_drawdown:
        end_idx = len(drawdown) - 1
        periods.append({
            'start': drawdown.index[start_idx].strftime(_DATE_FORMAT),
            'end': drawdown.index[end_idx].strftime(_DATE_FORMAT),
            'depth': drawdown.iloc[start_idx:end_idx + 1].min(),
            'duration_days': (drawdown.index[end_idx] - drawdown.index[start_idx]).days,
        })
    
    return sorted(periods, key=lambda x: x['depth'])


def _calculate_performance_metrics(returns: pd.Series, risk_free_rate: float = 0.0, 
                                   annualization_factor: float = 365.0) -> Dict:
    """Calculate Sharpe, Sortino, Calmar, linearity, and risk metrics."""
    if returns.empty or len(returns) < 2:
        return {}
    
    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1
    days = len(returns)
    annual_return = (1 + total_return) ** (annualization_factor / days) - 1 if days > 1 else 0
    annual_vol = returns.std() * np.sqrt(annualization_factor)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    rolling_max = equity.expanding().max()
    drawdown = equity / rolling_max - 1
    max_drawdown = drawdown.min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    t = np.arange(len(equity))
    r_value = linregress(t, equity.values)[2]
    linearity = r_value ** 2
    
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(annualization_factor) if len(downside_returns) > 0 else 0
    sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    var_95 = returns.quantile(0.05)
    cvar = returns[returns <= var_95].mean() if (returns <= var_95).any() else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'linearity': linearity,
        'drawdown_periods': _identify_drawdown_periods(equity),
        'var_95': var_95,
        'cvar': cvar,
    }


class Portfolio:
    """Portfolio state with trade execution and valuation."""
    def __init__(self, initial_capital: float = 1000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.holdings = {}
        self.total_value = initial_capital
        self.history = []
        self.trade_log = []

    def update_market_value(self, date, prices: pd.Series):
        """Update holdings value and total portfolio value."""
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
        """Execute trade with asymmetric buy/sell costs."""
        if isinstance(transaction_cost_rates, (list, tuple)):
            buy_cost_rate, sell_cost_rate = transaction_cost_rates
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
        """DataFrame with datetime index."""
        if not data_list:
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')
    
    def get_history_df(self) -> pd.DataFrame:
        """Portfolio value history."""
        return self._build_datetime_df(self.history)

    def get_trade_log_df(self) -> pd.DataFrame:
        """Trade execution log."""
        return self._build_datetime_df(self.trade_log)


class Backtester:
    """Factor strategy backtesting engine."""
    
    def __init__(
        self,
        price_factor: 'Factor',
        strategy_factor: 'Factor',
        transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
        initial_capital: float = 1000,
        full_rebalance: bool = False,
        neutralization: str = "market"
    ):
        """Initialize backtester."""
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
        """Execute backtest and return self for method chaining."""
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
                            self.portfolio.execute_trade(
                                symbol, -self.portfolio.positions[symbol], 
                                current_prices.loc[symbol],
                                self.transaction_cost_rates, current_date)
                    self.portfolio.update_market_value(current_date, current_prices)
                
                for symbol, quantity in self._generate_orders(target_holdings, current_prices).items():
                    if symbol in current_prices.index:
                        self.portfolio.execute_trade(symbol, quantity, current_prices.loc[symbol], 
                                                    self.transaction_cost_rates, current_date)
            except Exception as e:
                logger.warning(f"Error on {current_date}: {e}")
                continue
        
        return self
    
    def calculate_metrics(self, risk_free_rate: float = 0.0) -> 'Backtester':
        """Calculate performance metrics and return self for chaining."""
        history = self.portfolio.get_history_df()
        if history.empty or len(history) < 2:
            self.metrics = {}
            return self
        
        equity_curve = history['total_value']
        daily_returns = equity_curve.pct_change(fill_method=None).dropna()
        
        self.metrics = _calculate_performance_metrics(daily_returns, risk_free_rate, annualization_factor=365.0)
        psr = self._calculate_psr(daily_returns) if not daily_returns.empty else 0
        self.metrics['psr'] = psr
        
        return self
    
    def _calculate_psr(self, daily_returns: pd.Series, sr_benchmark: float = 0.0) -> float:
        """Probabilistic Sharpe Ratio."""
        if len(daily_returns) < 2:
            return 0.0
        
        std = daily_returns.std()
        sr_obs = (daily_returns.mean() * 365) / (std * np.sqrt(365)) if std > 0 else 0
        
        T = len(daily_returns)
        adjustment = np.sqrt(1 - skew(daily_returns) * sr_obs + 
                           ((kurtosis(daily_returns, fisher=False) - 1) / 4) * sr_obs ** 2)
        psr_stat = (sr_obs - sr_benchmark) / adjustment * np.sqrt(T / 365)
        psr = norm.cdf(psr_stat)
        return float(np.clip(psr, 0.0, 1.0))


    def _build_date_cache(self, factor: 'Factor') -> dict:
        """Cache factor data by date."""
        cache = {}
        for date, group in factor.data.groupby('timestamp', sort=False):
            series = group.set_index('symbol')['factor']
            if not series.isna().any():
                cache[date] = series
        return cache
    
    def _get_factor_data(self, factor: 'Factor', date) -> pd.Series:
        """Retrieve cached factor data for date."""
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
        """Normalize factors to dollar-neutral target holdings."""
        if self.neutralization == "none":
            return factors * self.portfolio.total_value
        
        demeaned = factors - factors.mean()
        abs_sum = np.abs(demeaned).sum()
        if abs_sum < 1e-10:
            return pd.Series(0.0, index=factors.index)
        
        return (demeaned / abs_sum) * self.portfolio.total_value
    
    def _generate_orders(self, target_holdings: pd.Series, prices: pd.Series) -> pd.Series:
        """Generate trade orders from target vs current holdings."""
        current_holdings = self.portfolio.holdings
        all_symbols = set(target_holdings.index) | set(current_holdings.keys())
        trade_quantities = {}
        prices_dict = prices.to_dict()
        
        for symbol in all_symbols:
            if symbol not in prices_dict:
                continue
            trade_value = target_holdings.get(symbol, 0) - current_holdings.get(symbol, 0)
            if abs(trade_value) > 1e-10:
                trade_quantities[symbol] = trade_value / prices_dict[symbol]
        
        return pd.Series(trade_quantities)
    
    def get_daily_returns(self) -> pd.Series:
        """Portfolio daily returns."""
        history = self.portfolio.get_history_df()
        if history.empty or len(history) < 2:
            return pd.Series(dtype=float)
        equity = history['total_value']
        return equity.pct_change(fill_method=None).dropna()

    def get_equity_curve(self) -> pd.Series:
        """Equity curve normalized to 1.0 at start."""
        history = self.portfolio.get_history_df()
        if history.empty:
            return pd.Series(dtype=float)
        equity = history['total_value']
        return equity / equity.iloc[0]

    def summary(self) -> str:
        """Performance summary string."""
        if not self.metrics:
            return "No metrics available."
        
        equity_curve = self.portfolio.get_history_df()['total_value']
        summary_lines = [f"Strategy: {self.strategy_factor.name}"]
        
        if not equity_curve.empty:
            start = equity_curve.index[0].strftime(_DATE_FORMAT)
            end = equity_curve.index[-1].strftime(_DATE_FORMAT)
            
            turnover_df = self.get_daily_turnover_df()
            avg_turnover = turnover_df['turnover'].mean() * 365 if not turnover_df.empty else 0
            
            summary_lines.extend([
                f"Period: {start} to {end}",
                f"Total Return: {self.metrics.get('total_return', 0):.2%}",
                f"Annual Return: {self.metrics.get('annual_return', 0):.2%}",
                f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}",
                f"PSR: {self.metrics.get('psr', 0):.1%}",
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
        """Print summary and return self for chaining."""
        print(self.summary())
        return self
    
    def print_drawdown_periods(self, top_n: int = 5) -> 'Backtester':
        """Print top N drawdown periods by depth."""
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
        """Equal-weight benchmark: buy all assets on first date, hold."""
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
        """Plot equity, drawdown, turnover, summary, and benchmark."""
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
        
        ax.set_facecolor(_PLOT_COLORS['background'])
        y_min = equity_curve.min()
        _plot_equity_line(ax, equity_curve, y_min)
        
        if benchmark_norm is not None and len(benchmark_norm) > 0:
            benchmark_abs = benchmark_norm * self.portfolio.initial_capital
            y_min = min(y_min, benchmark_abs.min())
            ax.plot(benchmark_norm.index, benchmark_abs, color=_PLOT_COLORS['benchmark_line'], 
                   linewidth=_PLOT_STYLES['benchmark_linewidth'], alpha=_PLOT_STYLES['benchmark_alpha'], linestyle='--')
        
        ax.set_title(f'Equity Curve ({self.strategy_factor.name})', 
                    fontsize=_PLOT_STYLES['title_size'], fontweight='400', color=_PLOT_COLORS['text'], pad=14)
        ax.set_ylabel('Equity Value', fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_light'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.grid(True, alpha=_PLOT_STYLES['grid_alpha'], color=_PLOT_COLORS['grid'], linestyle='-', linewidth=_PLOT_STYLES['grid_width'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=_PLOT_STYLES['label_size'], colors=_PLOT_COLORS['text_light'], 
                      width=_PLOT_STYLES['spine_width'], length=_PLOT_STYLES['tick_length'])
        
        if show_summary:
            summary_text = self.summary()
            ax.text(0.015, 0.98, summary_text, transform=ax.transAxes, fontsize=_PLOT_STYLES['label_size'], 
                    verticalalignment='top', horizontalalignment='left', color=_PLOT_COLORS['text_info'],
                    bbox=dict(boxstyle='round,pad=0.75', facecolor=_PLOT_COLORS['white'], 
                             edgecolor=_PLOT_COLORS['grid'], alpha=_PLOT_STYLES['box_alpha'], linewidth=1))
        
        ax_dd.set_facecolor(_PLOT_COLORS['white'])
        _plot_drawdown(ax_dd, drawdown)
        ax_dd.set_ylabel('Drawdown', fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_light'])
        ax_dd.grid(True, alpha=_PLOT_STYLES['grid_alpha_secondary'], color=_PLOT_COLORS['grid'], linestyle='-', linewidth=_PLOT_STYLES['grid_width'])
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax_dd.spines['top'].set_visible(False)
        ax_dd.spines['right'].set_visible(False)
        ax_dd.tick_params(axis='both', which='major', labelsize=_PLOT_STYLES['small_label_size'], colors=_PLOT_COLORS['text_light'], 
                          width=_PLOT_STYLES['spine_width'], length=_PLOT_STYLES['tick_length'])
        
        ax_to.set_facecolor(_PLOT_COLORS['white'])
        if not turnover_df.empty:
            ax_to.plot(turnover_df.index, turnover_df['turnover'], color=_PLOT_COLORS['turnover_line'], 
                      linewidth=_PLOT_STYLES['thin_linewidth'], alpha=_PLOT_STYLES['line_alpha'])
            ax_to.set_ylabel('Turnover', fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_light'])
            ax_to.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
            ax_to.grid(True, alpha=_PLOT_STYLES['grid_alpha_secondary'], color=_PLOT_COLORS['grid'], linestyle='-', linewidth=_PLOT_STYLES['grid_width'])
        else:
            ax_to.text(0.5, 0.5, 'No Turnover Data', transform=ax_to.transAxes, 
                      ha='center', va='center', fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_muted'])
        
        ax_to.set_xlabel('Date', fontsize=_PLOT_STYLES['xlabel_size'], color=_PLOT_COLORS['text_light'])
        ax_to.spines['top'].set_visible(False)
        ax_to.spines['right'].set_visible(False)
        ax_to.tick_params(axis='both', which='major', labelsize=_PLOT_STYLES['small_label_size'], colors=_PLOT_COLORS['text_light'], 
                          width=_PLOT_STYLES['spine_width'], length=_PLOT_STYLES['tick_length'])
        
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax_dd.get_xticklabels(), visible=False)
        
        plt.show()
        return self
    
    def get_daily_turnover_df(self) -> pd.DataFrame:
        """Daily turnover (|trades| / NAV)."""
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
    
    def calculate_ic(self, periods: Union[int, List[int]] = 1) -> Dict:
        """Calculate IC and RankIC metrics for given forward periods."""
        from scipy.stats import rankdata, pearsonr
        
        if isinstance(periods, int):
            periods = [periods]
        
        strategy_data = self.strategy_factor.data.copy().sort_values(['symbol', 'timestamp'])
        price_data = self.price_factor.data.copy().sort_values(['symbol', 'timestamp'])
        
        if strategy_data.empty or price_data.empty:
            return {}
        
        results = {}
        unique_dates = sorted(strategy_data['timestamp'].unique())
        
        for period in periods:
            ic_values, rankic_values = [], []
            
            for i, date in enumerate(unique_dates):
                if i + period >= len(unique_dates):
                    break
                
                forward_date = unique_dates[i + period]
                
                factor_t = strategy_data[strategy_data['timestamp'] == date][['symbol', 'factor']].set_index('symbol')
                price_t = price_data[price_data['timestamp'] == date][['symbol', 'factor']].set_index('symbol')
                price_f = price_data[price_data['timestamp'] == forward_date][['symbol', 'factor']].set_index('symbol')
                
                symbols = factor_t.index.intersection(price_t.index).intersection(price_f.index)
                if len(symbols) < 2:
                    continue
                
                factor_vals = factor_t.loc[symbols, 'factor'].values
                returns = (price_f.loc[symbols, 'factor'].values - price_t.loc[symbols, 'factor'].values) / price_t.loc[symbols, 'factor'].values
                
                valid = ~(np.isnan(factor_vals) | np.isnan(returns))
                if valid.sum() < 2:
                    continue
                
                factor_vals = factor_vals[valid]
                returns = returns[valid]
                
                ic, _ = pearsonr(factor_vals, returns)
                ic_values.append(ic)
                
                factor_rank = rankdata(factor_vals)
                returns_rank = rankdata(returns)
                rankic, _ = pearsonr(factor_rank, returns_rank)
                rankic_values.append(rankic)
            
            if ic_values:
                ic_mean = np.nanmean(ic_values)
                ic_std = np.nanstd(ic_values)
                rankic_mean = np.nanmean(rankic_values)
                rankic_std = np.nanstd(rankic_values)
                
                results[period] = {
                    'ic_mean': ic_mean,
                    'ic_std': ic_std,
                    'icir': (ic_mean / ic_std * np.sqrt(365)) if ic_std > 0 else 0,
                    'rankic_mean': rankic_mean,
                    'rankic_std': rankic_std,
                    'rankicir': (rankic_mean / rankic_std * np.sqrt(365)) if rankic_std > 0 else 0,
                    'n': len(ic_values)
                }
        
        return results
    
    def print_ic(self, periods: Union[int, List[int]] = None) -> 'Backtester':
        """Print RankIC statistics for multiple periods."""
        if periods is None:
            periods = [1, 5, 10, 20]
        elif isinstance(periods, int):
            periods = [periods]
        
        ic_results = self.calculate_ic(periods)
        if not ic_results:
            logger.warning("No IC data available")
            return self
        
        print(f"\nIC Analysis ({self.strategy_factor.name})")
        print("-" * 80)
        print(f"{'Period':<10} {'RankIC Mean':<16} {'RankIC Std':<16} {'RankICIR':<16} {'Samples':<10}")
        print("-" * 80)
        
        for period in sorted(ic_results.keys()):
            metrics = ic_results[period]
            print(f"{period}d{' '*7} {metrics['rankic_mean']:>13.4f}  "
                  f"{metrics['rankic_std']:>13.4f}  {metrics['rankicir']:>13.4f}  {metrics['n']:>8}")
        
        print("-" * 80)
        return self

    def __repr__(self):
        """String representation."""
        history = self.portfolio.get_history_df()
        if not history.empty:
            days = len(history)
            start_date = history.index[0].strftime(_DATE_FORMAT)
            end_date = history.index[-1].strftime(_DATE_FORMAT)
            return (f"Backtester(strategy={self.strategy_factor.name}, "
                   f"period={start_date} to {end_date}, days={days})")
        else:
            return (f"Backtester(strategy={self.strategy_factor.name}, "
                   f"price={self.price_factor.name}, cost={self.transaction_cost_rates[0]:.3%})")

    def __add__(self, other: 'Backtester') -> 'CombinedBacktester':
        """Combine two strategies with equal 50-50 weights."""
        return CombinedBacktester([self, other], weights=[0.5, 0.5])

    def __mul__(self, weight: float) -> 'CombinedBacktester':
        """Apply portfolio weight (0-1)."""
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
        return CombinedBacktester([self], weights=[weight])


class CombinedBacktester:
    """Multi-strategy portfolio combining."""

    def __init__(self, backtests: List[Backtester], weights: List[float]):
        """Combine backtests with equal or custom weights."""
        if len(backtests) != len(weights):
            raise ValueError("Number of weights must match number of backtests")
        if not np.isclose(sum(weights), 1.0, atol=1e-10):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights):.6f}")
        
        self.backtests = backtests
        self.weights = np.array(weights)
        self.metrics = {}
        self.calculate_metrics()

    def calculate_metrics(self, risk_free_rate: float = 0.0) -> 'CombinedBacktester':
        """Calculate combined portfolio metrics."""
        port_returns = self.get_portfolio_returns()
        self.metrics = _calculate_performance_metrics(port_returns, risk_free_rate, annualization_factor=252.0)
        return self

    def _get_aligned_returns(self) -> pd.DataFrame:
        """Align returns across strategies."""
        returns_dict = {}
        for i, bt in enumerate(self.backtests):
            returns_dict[f'strategy_{i}'] = bt.get_daily_returns()
        
        df = pd.DataFrame(returns_dict)
        return df.dropna(how='all')

    def get_portfolio_returns(self) -> pd.Series:
        """Weighted returns."""
        returns_df = self._get_aligned_returns()
        if returns_df.empty:
            return pd.Series(dtype=float)
        
        portfolio_returns = (returns_df * self.weights).sum(axis=1)
        return portfolio_returns

    def get_portfolio_equity(self) -> pd.Series:
        """Weighted equity curve."""
        port_returns = self.get_portfolio_returns()
        if port_returns.empty:
            return pd.Series(dtype=float)
        
        equity = (1 + port_returns).cumprod()
        initial_capital = sum([bt.portfolio.initial_capital * w 
                              for bt, w in zip(self.backtests, self.weights)])
        return equity * initial_capital

    def correlation_matrix(self) -> pd.DataFrame:
        """Strategy correlation matrix."""
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
        """Print correlation matrix."""
        corr = self.correlation_matrix()
        if not corr.empty:
            print("\nCorrelation Matrix:")
            print(corr.to_string(max_cols=None, max_rows=None, float_format=lambda x: f'{x:.6f}'))
        return self

    def summary(self) -> str:
        """Portfolio summary string."""
        lines = []
        
        strategy_names = ", ".join([bt.strategy_factor.name for bt in self.backtests])
        lines.append(f"Strategy: {strategy_names}")
        
        equity = self.get_portfolio_equity()
        if not equity.empty and len(equity) > 0:
            start_date = equity.index[0].strftime(_DATE_FORMAT)
            end_date = equity.index[-1].strftime(_DATE_FORMAT)
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
        """Print summary and return self for chaining."""
        print(self.summary())
        return self
    
    def print_drawdown_periods(self, top_n: int = 5) -> 'CombinedBacktester':
        """Print top N drawdown periods by depth."""
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
    
    def plot_equity(self, figsize: tuple = (12, 7), show_summary: bool = True) -> 'CombinedBacktester':
        """Plot portfolio equity, drawdown, and summary."""
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
        
        ax.set_facecolor(_PLOT_COLORS['background'])
        y_min = equity.min()
        _plot_equity_line(ax, equity, y_min)
        
        strategy_names = ", ".join([bt.strategy_factor.name for bt in self.backtests])
        ax.set_title(f'Combined Portfolio Equity ({strategy_names})', 
                    fontsize=_PLOT_STYLES['title_size'], fontweight='400', color=_PLOT_COLORS['text'], pad=14)
        ax.set_ylabel('Equity Value', fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_light'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.grid(True, alpha=_PLOT_STYLES['grid_alpha'], color=_PLOT_COLORS['grid'], linestyle='-', linewidth=_PLOT_STYLES['grid_width'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=_PLOT_STYLES['label_size'], colors=_PLOT_COLORS['text_light'], 
                      width=_PLOT_STYLES['spine_width'], length=_PLOT_STYLES['tick_length'])
        
        if show_summary:
            summary_text = self.summary()
            ax.text(0.015, 0.98, summary_text, transform=ax.transAxes, fontsize=_PLOT_STYLES['label_size'], 
                    verticalalignment='top', horizontalalignment='left', color=_PLOT_COLORS['text_info'],
                    bbox=dict(boxstyle='round,pad=0.75', facecolor=_PLOT_COLORS['white'], 
                             edgecolor=_PLOT_COLORS['grid'], alpha=_PLOT_STYLES['box_alpha'], linewidth=1))
        
        ax_dd.set_facecolor(_PLOT_COLORS['white'])
        _plot_drawdown(ax_dd, drawdown)
        ax_dd.set_ylabel('Drawdown', fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_light'])
        ax_dd.set_xlabel('Date', fontsize=_PLOT_STYLES['xlabel_size'], color=_PLOT_COLORS['text_light'])
        ax_dd.grid(True, alpha=_PLOT_STYLES['grid_alpha_secondary'], color=_PLOT_COLORS['grid'], linestyle='-', linewidth=_PLOT_STYLES['grid_width'])
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax_dd.spines['top'].set_visible(False)
        ax_dd.spines['right'].set_visible(False)
        ax_dd.tick_params(axis='both', which='major', labelsize=_PLOT_STYLES['small_label_size'], colors=_PLOT_COLORS['text_light'], 
                          width=_PLOT_STYLES['spine_width'], length=_PLOT_STYLES['tick_length'])
        
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
        """Scale portfolio allocation."""
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
    """Quick backtesting convenience function."""
    bt = Backtester(price_factor, strategy_factor, transaction_cost, initial_capital, 
                   full_rebalance, neutralization)
    
    if auto_run:
        bt.run().calculate_metrics()
    
    return bt