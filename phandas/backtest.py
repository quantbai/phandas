"""Backtesting engine for factor strategies."""

import warnings
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING, Union, Tuple, Dict, List, Optional
from scipy.stats import linregress

if TYPE_CHECKING:
    from .core import Factor

from .plot import BacktestPlotter, _DATE_FORMAT
from .console import print, console


def _identify_drawdown_periods(equity_series: pd.Series) -> List[Dict]:
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


def _calculate_performance_metrics(returns: pd.Series, risk_free_rate: float = 0.03, 
                                   annualization_factor: float = 365.0) -> Dict:
    if returns.empty or len(returns) < 2:
        return {}
    
    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1
    if hasattr(returns.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(returns.index):
        days = (returns.index[-1] - returns.index[0]).days
    else:
        days = len(returns)
    
    annual_return = (1 + total_return) ** (annualization_factor / days) - 1 if days > 0 else 0
    annual_vol = returns.std() * np.sqrt(annualization_factor)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    rolling_max = equity.expanding().max()
    drawdown = equity / rolling_max - 1
    max_drawdown = drawdown.min()
    
    t = np.arange(len(equity))
    r_value = linregress(t, equity.values)[2]
    linearity = r_value ** 2
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'linearity': linearity,
        'drawdown_periods': _identify_drawdown_periods(equity),
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
        if not data_list:
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')
    
    def get_history_df(self) -> pd.DataFrame:
        return self._build_datetime_df(self.history)

    def get_trade_log_df(self) -> pd.DataFrame:
        return self._build_datetime_df(self.trade_log)


class Backtester:
    """Factor strategy backtesting engine."""
    
    def __init__(
        self,
        entry_price_factor: 'Factor',
        strategy_factor: 'Factor',
        transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
        initial_capital: float = 1000,
        full_rebalance: bool = False
    ):
        self.entry_price_factor = entry_price_factor
        self.strategy_factor = strategy_factor.signal()
        self.full_rebalance = full_rebalance
        
        if isinstance(transaction_cost, (list, tuple)):
            self.transaction_cost_rates = tuple(transaction_cost)
        else:
            self.transaction_cost_rates = (transaction_cost, transaction_cost)
        
        self.portfolio = Portfolio(initial_capital)
        self.metrics = {}
        
        self._price_cache = self._build_date_cache(entry_price_factor)
        self._strategy_cache = self._build_date_cache(self.strategy_factor)
    
    def run(self) -> 'Backtester':
        price_dates = set(self.entry_price_factor.data['timestamp'].unique())
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
        
        last_valid_prices = {}

        for i in range(start_idx, len(common_dates)):
            current_date = common_dates[i]
            prev_date = common_dates[i - 1] if i > 0 else None
            
            try:
                current_prices = self._get_factor_data(self.entry_price_factor, current_date)
                valid_prices = current_prices.dropna()
                
                if valid_prices.empty:
                    continue
                
                for symbol, price in valid_prices.items():
                    last_valid_prices[symbol] = price
                
                nan_symbols = set(current_prices.index) - set(valid_prices.index)
                if nan_symbols:
                    for symbol in nan_symbols:
                        if symbol in self.portfolio.positions and symbol in last_valid_prices:
                            self.portfolio.execute_trade(
                                symbol, -self.portfolio.positions[symbol],
                                last_valid_prices[symbol],
                                self.transaction_cost_rates, current_date)
                
                self.portfolio.update_market_value(current_date, valid_prices)
                if not prev_date:
                    continue
                
                strategy_factors = self._get_factor_data(self.strategy_factor, prev_date)
                
                if nan_symbols:
                    strategy_factors = self._resignal_excluding(strategy_factors, nan_symbols)
                
                target_holdings = self._calculate_target_holdings(strategy_factors, prev_date)
                
                if self.full_rebalance:
                    for symbol in list(self.portfolio.positions.keys()):
                        if symbol in valid_prices.index:
                            self.portfolio.execute_trade(
                                symbol, -self.portfolio.positions[symbol], 
                                valid_prices.loc[symbol],
                                self.transaction_cost_rates, current_date)
                    self.portfolio.update_market_value(current_date, valid_prices)
                
                for symbol, quantity in self._generate_orders(target_holdings, valid_prices).items():
                    if symbol in valid_prices.index:
                        self.portfolio.execute_trade(symbol, quantity, valid_prices.loc[symbol], 
                                                    self.transaction_cost_rates, current_date)
            except Exception as e:
                warnings.warn(f"Error on {current_date}: {e}")
                continue
        
        return self
    
    def _resignal_excluding(self, factors: pd.Series, exclude_symbols: set) -> pd.Series:
        """Re-calculate signal weights excluding specified symbols.
        
        When a symbol has no price (NaN), its factor is set to 0 and weights
        are redistributed among remaining symbols to maintain dollar-neutral.
        """
        adjusted = factors.copy()
        for symbol in exclude_symbols:
            if symbol in adjusted.index:
                adjusted.loc[symbol] = 0.0
        
        valid_data = adjusted[adjusted != 0]
        if len(valid_data) < 2:
            return pd.Series(0.0, index=factors.index)
        
        demeaned = valid_data - valid_data.mean()
        abs_sum = np.abs(demeaned).sum()
        
        if abs_sum < 1e-10:
            return pd.Series(0.0, index=factors.index)
        
        result = pd.Series(0.0, index=factors.index)
        result[valid_data.index] = demeaned / abs_sum
        return result
    
    def calculate_metrics(self, risk_free_rate: float = 0.03) -> 'Backtester':
        history = self.portfolio.get_history_df()
        if history.empty or len(history) < 2:
            self.metrics = {}
            return self
        
        equity_curve = history['total_value']
        daily_returns = equity_curve.pct_change().dropna()
        
        self.metrics = _calculate_performance_metrics(daily_returns, risk_free_rate, annualization_factor=365)
        
        return self


    def _build_date_cache(self, factor: 'Factor') -> dict:
        """Build date-indexed cache for fast factor data lookup."""
        cache = {}
        for date in factor.data['timestamp'].unique():
            group = factor.data[factor.data['timestamp'] == date]
            cache[date] = group.set_index('symbol')['factor']
        return cache
    
    def _get_factor_data(self, factor: 'Factor', date) -> pd.Series:
        if date is None:
            return pd.Series(dtype=float)
        
        if factor is self.entry_price_factor:
            return self._price_cache.get(date, pd.Series(dtype=float))
        else:
            return self._strategy_cache.get(date, pd.Series(dtype=float))
    
    def _find_start_date(self, dates) -> int:
        for i, date in enumerate(dates):
            if i == 0:
                continue
            prev_date = dates[i - 1]
            
            strategy_data = self._get_factor_data(self.strategy_factor, prev_date)
            price_data = self._get_factor_data(self.entry_price_factor, date)
            
            if strategy_data.empty or price_data.empty:
                continue
            
            if (strategy_data.abs() > 1e-10).any():
                return i
        
        raise ValueError("No valid start date found with non-zero signal")
    
    def _calculate_target_holdings(self, factors: pd.Series, date=None) -> pd.Series:
        return factors * self.portfolio.total_value
    
    def _generate_orders(self, target_holdings: pd.Series, prices: pd.Series) -> pd.Series:
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

    def summary(self) -> str:
        if not self.metrics:
            return "Backtester(no metrics available)"
        
        equity_curve = self.portfolio.get_history_df()['total_value']
        if equity_curve.empty:
            return "Backtester(no data)"
        
        start = equity_curve.index[0].strftime(_DATE_FORMAT)
        end = equity_curve.index[-1].strftime(_DATE_FORMAT)
        name = self.strategy_factor.name
        
        turnover_df = self.turnover
        avg_turnover = turnover_df['turnover'].mean() * 365 if not turnover_df.empty else 0
        
        m = self.metrics
        lines = [
            f"Backtester(strategy='{name}', period={start} to {end})",
            f"  total_return:   {m.get('total_return', 0):>8.2%}    annual_return:  {m.get('annual_return', 0):>8.2%}",
            f"  sharpe_ratio:   {m.get('sharpe_ratio', 0):>8.2f}    max_drawdown:   {m.get('max_drawdown', 0):>8.2%}",
            f"  linearity:      {m.get('linearity', 0):>8.4f}    turnover:       {avg_turnover:>8.2%}",
        ]
        
        return "\n".join(lines)
    
    def print_summary(self) -> 'Backtester':
        if not self.metrics:
            print("Backtester(no metrics available)")
            return self
        
        equity_curve = self.portfolio.get_history_df()['total_value']
        if equity_curve.empty:
            print("Backtester(no data)")
            return self
        
        start = equity_curve.index[0].strftime(_DATE_FORMAT)
        end = equity_curve.index[-1].strftime(_DATE_FORMAT)
        name = self.strategy_factor.name
        
        turnover_df = self.turnover
        avg_turnover = turnover_df['turnover'].mean() * 365 if not turnover_df.empty else 0
        
        m = self.metrics
        print(f"Backtester(strategy='{name}', period={start} to {end})")
        print(f"  total_return:   {m.get('total_return', 0):>8.2%}    annual_return:  {m.get('annual_return', 0):>8.2%}")
        print(f"  sharpe_ratio:   {m.get('sharpe_ratio', 0):>8.2f}    max_drawdown:   {m.get('max_drawdown', 0):>8.2%}")
        print(f"  linearity:      {m.get('linearity', 0):>8.4f}    turnover:       {avg_turnover:>8.2%}")
        
        return self
    
    def print_drawdown_periods(self, top_n: int = 5) -> 'Backtester':
        drawdown_periods = self.metrics.get('drawdown_periods', [])
        
        if not drawdown_periods:
            print("Drawdown Periods: none detected")
            return self
        
        periods_to_show = drawdown_periods[:top_n]
        total_periods = len(drawdown_periods)
        
        print(f"Drawdown Periods (top {min(top_n, total_periods)}):")
        for i, period in enumerate(periods_to_show, 1):
            print(f"  {i}. {period['start']} to {period['end']}    "
                  f"depth={period['depth']:.2%}    duration={period['duration_days']}d")
        
        if total_periods > top_n:
            print(f"  (showing {top_n} of {total_periods} periods)")
        
        return self
    
    def _calculate_benchmark_equity(self) -> pd.Series:
        history = self.portfolio.get_history_df()
        if history.empty or len(history) < 2:
            return pd.Series(dtype=float)
        
        first_date = history.index[1]
        prices_first = self._price_cache.get(first_date)
        if prices_first is None or prices_first.empty:
            return pd.Series(dtype=float)
        
        valid_prices_first = prices_first.dropna()
        if valid_prices_first.empty:
            return pd.Series(dtype=float)
        
        alloc_per_asset = self.portfolio.initial_capital / len(valid_prices_first)
        holdings = {s: alloc_per_asset / valid_prices_first[s] for s in valid_prices_first.index}
        
        values, dates = [], []
        for date in sorted(self._price_cache.keys()):
            if date < first_date:
                continue
            prices = self._price_cache[date]
            if prices.empty:
                continue
            valid_prices = prices.dropna()
            portfolio_value = sum(holdings[s] * valid_prices[s] for s in holdings if s in valid_prices.index)
            if portfolio_value > 0:
                values.append(portfolio_value)
                dates.append(date)
        
        return pd.Series(values, index=pd.DatetimeIndex(dates))
    
    def plot_equity(self, figsize: tuple = (14, 8), show_summary: bool = True, show_benchmark: bool = True) -> 'Backtester':
        plotter = BacktestPlotter(self)
        plotter.plot_equity(figsize, show_summary, show_benchmark)
        return self
    
    @property
    def returns(self) -> pd.Series:
        history = self.portfolio.get_history_df()
        if history.empty or len(history) < 2:
            return pd.Series(dtype=float)
        return history['total_value'].pct_change().dropna()
    
    @property
    def equity(self) -> pd.Series:
        history = self.portfolio.get_history_df()
        return history['total_value'] if not history.empty else pd.Series(dtype=float)
    
    @property
    def trades(self) -> pd.DataFrame:
        return self.portfolio.get_trade_log_df()
    
    @property
    def turnover(self) -> pd.DataFrame:
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
    
    @property
    def drawdown(self) -> pd.Series:
        equity = self.equity
        if equity.empty:
            return pd.Series(dtype=float)
        return equity / equity.cummax() - 1
    
    def to_dict(self) -> dict:
        equity = self.equity
        return {
            'strategy': self.strategy_factor.name,
            'period': {
                'start': equity.index[0].strftime(_DATE_FORMAT) if not equity.empty else None,
                'end': equity.index[-1].strftime(_DATE_FORMAT) if not equity.empty else None,
            },
            'metrics': self.metrics,
            'returns': self.returns.to_dict() if not self.returns.empty else {},
            'equity': equity.to_dict() if not equity.empty else {},
        }
    
    def __repr__(self):
        history = self.portfolio.get_history_df()
        if not history.empty:
            days = len(history)
            start_date = history.index[0].strftime(_DATE_FORMAT)
            end_date = history.index[-1].strftime(_DATE_FORMAT)
            return (f"Backtester(strategy={self.strategy_factor.name}, "
                   f"period={start_date} to {end_date}, days={days})")
        else:
            return (f"Backtester(strategy={self.strategy_factor.name}, "
                   f"entry_price={self.entry_price_factor.name}, cost={self.transaction_cost_rates[0]:.3%})")


def backtest(
    entry_price_factor: 'Factor',
    strategy_factor: 'Factor',
    transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
    initial_capital: float = 1000,
    full_rebalance: bool = False,
    auto_run: bool = True
) -> Backtester:
    bt = Backtester(entry_price_factor, strategy_factor, transaction_cost, initial_capital, 
                   full_rebalance)
    
    if auto_run:
        bt.run().calculate_metrics()
    
    return bt