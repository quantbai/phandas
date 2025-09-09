"""
Professional backtesting engine for factor strategies.

Efficient implementation with clean API.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Union, Tuple
import logging

if TYPE_CHECKING:
    from .core import Factor


logger = logging.getLogger(__name__)

# Configure matplotlib
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Portfolio:
    """
    Manages the state of a trading portfolio.
    
    Tracks cash, positions, and historical performance.
    """
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = pd.Series(dtype=float)  # symbol -> quantity
        self.holdings = pd.Series(dtype=float)   # symbol -> market value
        self.total_value = initial_capital
        
        self.history = []
        self.trade_log = [] # 新增：記錄詳細交易日誌

    def update_market_value(self, date, prices: pd.Series):
        """
        Update holdings and total portfolio value based on new prices.
        This is the "mark-to-market" step.
        """
        common_symbols = self.positions.index.intersection(prices.index)
        
        self.holdings = self.positions.loc[common_symbols] * prices.loc[common_symbols]
        
        previous_total_value = self.total_value
        self.total_value = self.cash + self.holdings.sum()
        
        # 計算每日 PnL
        daily_pnl = self.total_value - previous_total_value

        # 計算多頭和空頭持倉市值
        long_holdings_value = self.holdings[self.holdings > 0].sum()
        short_holdings_value = self.holdings[self.holdings < 0].sum()
        
        self.history.append({
            'date': date,
            'total_value': self.total_value,
            'cash': self.cash,
            'holdings_value': self.holdings.sum(),
            'daily_pnl': daily_pnl, # 新增：記錄每日 PnL
            'long_holdings_value': long_holdings_value, # 新增：記錄多頭持倉市值
            'short_holdings_value': short_holdings_value, # 新增：記錄空頭持倉市值
        })

    def execute_trade(self, symbol: str, quantity: float, price: float, transaction_cost_rates: Union[float, Tuple[float, float]], trade_date: pd.Timestamp):
        """
        Executes a single trade and updates portfolio state.
        
        Parameters
        ----------
        symbol : str
            The asset symbol.
        quantity : float
            Number of shares to trade. Positive for buy, negative for sell.
        price : float
            Execution price per share.
        transaction_cost_rates : Union[float, Tuple[float, float]]
            Transaction cost. Can be a single float (e.g., 0.001) for both buy/sell,
            or a tuple (buy_cost_rate, sell_cost_rate) for separate rates.
        """
        # 處理交易成本可以是單一費率或 (買入費率, 賣出費率) 的情況
        if isinstance(transaction_cost_rates, (list, tuple)):
            buy_cost_rate = transaction_cost_rates[0]
            sell_cost_rate = transaction_cost_rates[1]
        else:
            buy_cost_rate = transaction_cost_rates
            sell_cost_rate = transaction_cost_rates
            
        trade_value = quantity * price
        
        if quantity > 0: # 買入
            cost = abs(trade_value) * buy_cost_rate
        else: # 賣出
            cost = abs(trade_value) * sell_cost_rate
        
        self.cash -= (trade_value + cost)
        
        current_quantity = self.positions.get(symbol, 0.0)
        new_quantity = current_quantity + quantity
        
        # 記錄交易日誌
        self.trade_log.append({
            'date': trade_date,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'trade_value': trade_value,
            'cost': cost
        })

        if np.isclose(new_quantity, 0):
            if symbol in self.positions.index:
                self.positions = self.positions.drop(symbol)
        else:
            self.positions.loc[symbol] = new_quantity
            
    def get_history_df(self) -> pd.DataFrame:
        """Return the portfolio history as a DataFrame."""
        if not self.history:
            return pd.DataFrame()
        df = pd.DataFrame(self.history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df

    def get_trade_log_df(self) -> pd.DataFrame:
        """Return the trade log as a DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        df = pd.DataFrame(self.trade_log)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df


class Backtester:
    """
    Professional backtesting engine with unified Factor format.
    
    Implements dollar-neutral factor strategies using a portfolio-based approach.
    """
    
    def __init__(
        self,
        price_factor: 'Factor',
        strategy_factor: 'Factor',
        transaction_cost: Union[float, Tuple[float, float]] = 0.001,
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
        transaction_cost : Union[float, Tuple[float, float]], default 0.001
            Transaction cost. Can be a single float (e.g., 0.001) for both buy/sell,
            or a tuple (buy_cost_rate, sell_cost_rate) for separate rates.
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
        
        # 確保 transaction_cost 始終是 (buy_rate, sell_rate) 形式的 tuple
        if isinstance(transaction_cost, (list, tuple)):
            self.transaction_cost_rates = tuple(transaction_cost)
        else:
            self.transaction_cost_rates = (transaction_cost, transaction_cost)
        
        self.portfolio = Portfolio(initial_capital)
        
        self.metrics = {}
    
    def run(self) -> 'Backtester':
        """
        Execute backtest using Factor objects.
        
        Returns
        -------
        Backtester
            Self for method chaining
        """
        price_dates = set(self.price_factor.data['timestamp'])
        strategy_dates = set(self.strategy_factor.data['timestamp'])
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
                        self.portfolio.execute_trade(symbol, quantity, price, self.transaction_cost_rates, current_date)
                
            except Exception as e:
                logger.warning(f"Error on {current_date}: {e}")
                continue
        
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
        history = self.portfolio.get_history_df()
        if history.empty or len(history) < 2:
            self.metrics = {}
            return self
        
        equity_curve = history['total_value']
        
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        days = max(days, 1)
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        daily_returns = equity_curve.pct_change().dropna()
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
    
    def _get_factor_data(self, factor: 'Factor', date) -> pd.Series:
        """Extract factor data for a specific date."""
        date_data = factor.data[factor.data['timestamp'] == date]
        if date_data.empty:
            return pd.Series(dtype=float)
        return date_data.set_index('symbol')['factor'].dropna()
    
    def _find_start_date(self, dates) -> int:
        """Find first date with complete data in both factors."""
        for i, date in enumerate(dates):
            if i == 0: continue
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
        
        all_symbols = target_holdings.index.union(current_holdings.index)
        
        trade_values = target_holdings.reindex(all_symbols, fill_value=0) - \
                       current_holdings.reindex(all_symbols, fill_value=0)
                       
        valid_prices = prices.reindex(trade_values.index)
        trade_quantities = trade_values / valid_prices
        
        return trade_quantities.dropna().loc[lambda x: ~np.isclose(x, 0)]
    
    def plot_equity(self, figsize: tuple = (12, 6), summary_text: str = None) -> 'Backtester':
        """
        Plot equity curve.
        
        Parameters
        ----------
        figsize : tuple, default (12, 6)
            Figure size
        summary_text : str, optional
            Summary text to display on the plot.
            
        Returns
        -------
        Backtester
            Self for method chaining
        """
        history = self.portfolio.get_history_df()
        if history.empty:
            logger.warning("No equity data to plot")
            return self
        
        equity_curve = history['total_value']
        
        fig, ax = plt.subplots(figsize=figsize)
        equity_curve.plot(ax=ax, color='blue', linewidth=1.5)
        ax.set_title(f'Equity Curve ({self.strategy_factor.name})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity Value')
        ax.grid(True, alpha=0.3)
        
        if summary_text:
            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10, 
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            
        plt.tight_layout()
        plt.show()
        
        return self
    
    def summary(self) -> str:
        """Return concise performance summary as a string for plotting."""
        if not self.metrics:
            return "No metrics available."
        
        equity_curve = self.portfolio.get_history_df()['total_value']
        
        summary_lines = []
        summary_lines.append(f"Strategy: {self.strategy_factor.name}")
        
        if not equity_curve.empty:
            start = equity_curve.index[0].strftime('%Y-%m-%d')
            end = equity_curve.index[-1].strftime('%Y-%m-%d')
            summary_lines.append(f"Period: {start} to {end}")
            summary_lines.append(f"Total Return: {self.metrics.get('total_return', 0):.2%}")
            summary_lines.append(f"Annual Return: {self.metrics.get('annual_return', 0):.2%}")
            summary_lines.append(f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
            summary_lines.append(f"Max Drawdown: {self.metrics.get('max_drawdown', 0):.2%}")
        
        return "\n".join(summary_lines)

    def calculate_total_transaction_cost(self) -> float:
        """
        Calculate the total transaction cost incurred during the backtest.
        
        Returns
        -------
        float
            Total transaction cost.
        """
        trade_log_df = self.portfolio.get_trade_log_df()
        if trade_log_df.empty:
            return 0.0
        return trade_log_df['cost'].sum()

    def __repr__(self):
        """Professional representation of Backtester."""
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
    transaction_cost: Union[float, Tuple[float, float]] = 0.001,
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
    transaction_cost : Union[float, Tuple[float, float]], default 0.001
        Transaction cost. Can be a single float (e.g., 0.001) for both buy/sell,
        or a tuple (buy_cost_rate, sell_cost_rate) for separate rates.
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
