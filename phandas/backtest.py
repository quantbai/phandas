"""
Professional backtesting engine for factor strategies.

Implements dollar-neutral portfolio-based backtesting with:
- Dynamic position rebalancing
- Transaction cost modeling (separate buy/sell rates)
- Real-time portfolio valuation
- Comprehensive performance metrics
- Detailed trade logging

Optimized for 3-column Factor format [timestamp, symbol, factor]

Classes:
    Portfolio: Portfolio state management and trade execution
    Backtester: Main backtesting engine with metrics calculation
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
    """
    Manages the state of a trading portfolio.
    
    Tracks cash, positions, historical equity curve, and detailed trade log.
    Supports real-time mark-to-market valuation and trade execution.
    """
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = pd.Series(dtype=float)
        self.holdings = pd.Series(dtype=float)
        self.total_value = initial_capital
        
        self.history = []
        self.trade_log = []

    def update_market_value(self, date, prices: pd.Series):
        """Update holdings and total portfolio value based on new prices."""
        common_symbols = self.positions.index.intersection(prices.index)
        
        self.holdings = self.positions.loc[common_symbols] * prices.loc[common_symbols]
        
        previous_total_value = self.total_value
        self.total_value = self.cash + self.holdings.sum()
        
        daily_pnl = self.total_value - previous_total_value
        long_holdings_value = self.holdings[self.holdings > 0].sum()
        short_holdings_value = self.holdings[self.holdings < 0].sum()
        
        self.history.append({
            'date': date,
            'total_value': self.total_value,
            'cash': self.cash,
            'holdings_value': self.holdings.sum(),
            'daily_pnl': daily_pnl,
            'long_holdings_value': long_holdings_value,
            'short_holdings_value': short_holdings_value,
        })

    def execute_trade(self, symbol: str, quantity: float, price: float, 
                     transaction_cost_rates: Union[float, Tuple[float, float]], 
                     trade_date: pd.Timestamp):
        """Execute a single trade and update portfolio state."""
        if isinstance(transaction_cost_rates, (list, tuple)):
            buy_cost_rate = transaction_cost_rates[0]
            sell_cost_rate = transaction_cost_rates[1]
        else:
            buy_cost_rate = transaction_cost_rates
            sell_cost_rate = transaction_cost_rates
            
        trade_value = quantity * price
        
        if quantity > 0:
            cost = abs(trade_value) * buy_cost_rate
        else:
            cost = abs(trade_value) * sell_cost_rate
        
        self.cash -= (trade_value + cost)
        
        current_quantity = self.positions.get(symbol, 0.0)
        new_quantity = current_quantity + quantity
        
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
    Professional backtesting engine with 3-column Factor format.
    
    Optimized for fast backtesting with [timestamp, symbol, factor] format.
    """
    
    def __init__(
        self,
        price_factor: 'Factor',
        strategy_factor: 'Factor',
        transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
        initial_capital: float = 100000
    ):
        """
        Initialize backtester with 3-column Factor format.
        
        Parameters
        ----------
        price_factor : Factor
            Price factor for entry/exit prices (e.g., open price)
        strategy_factor : Factor
            Strategy factor for position weights
        transaction_cost : Union[float, Tuple[float, float]], default 0.001
            Transaction cost rate(s)
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
        
        if isinstance(transaction_cost, (list, tuple)):
            self.transaction_cost_rates = tuple(transaction_cost)
        else:
            self.transaction_cost_rates = (transaction_cost, transaction_cost)
        
        self.portfolio = Portfolio(initial_capital)
        
        self.metrics = {}
        self.detailed_history = []
    
    def run(self) -> 'Backtester':
        """
        Execute backtest using 3-column Factor format.
        
        Returns
        -------
        Backtester
            Self for method chaining
        """
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
                
                current_strategy_factors = self._get_factor_data(self.strategy_factor, current_date)
                prev_strategy_factors = self._get_factor_data(self.strategy_factor, prev_date) if prev_date else pd.Series(dtype=float)
                
                old_holdings = self.portfolio.holdings.copy()
                old_total_value = self.portfolio.total_value
                
                self.portfolio.update_market_value(current_date, current_prices)
                
                if not prev_date:
                    continue
                
                strategy_factors = self._get_factor_data(self.strategy_factor, prev_date)
                
                target_holdings = self._calculate_target_holdings(strategy_factors)
                
                orders = self._generate_orders(target_holdings, current_prices)
                
                trade_pnl_by_symbol = {}
                trade_value_by_symbol = {}
                for symbol, quantity in orders.items():
                    if symbol in current_prices.index:
                        price = current_prices.loc[symbol]
                        self.portfolio.execute_trade(symbol, quantity, price, 
                                                    self.transaction_cost_rates, current_date)
                        
                        trade_value = quantity * price
                        trade_value_by_symbol[symbol] = trade_value
                        cost = abs(trade_value) * (self.transaction_cost_rates[0] if quantity > 0 
                                                   else self.transaction_cost_rates[1])
                        trade_pnl_by_symbol[symbol] = -abs(cost)
                
                all_symbols = set(current_prices.index) | set(current_strategy_factors.index) | \
                             set(prev_strategy_factors.index) | set(self.portfolio.holdings.index) | \
                             set(target_holdings.index)
                
                for symbol in all_symbols:
                    current_price = current_prices.get(symbol, np.nan)
                    current_factor = current_strategy_factors.get(symbol, np.nan)
                    prev_factor = prev_strategy_factors.get(symbol, np.nan)
                    factor_change = current_factor - prev_factor if not (pd.isna(current_factor) or 
                                                                         pd.isna(prev_factor)) else np.nan
                    
                    current_holding_value = self.portfolio.holdings.get(symbol, 0.0)
                    old_holding_value = old_holdings.get(symbol, 0.0)
                    holding_pnl = current_holding_value - old_holding_value if not pd.isna(current_price) else 0.0
                    trade_pnl = trade_pnl_by_symbol.get(symbol, 0.0)
                    
                    total_symbol_pnl = holding_pnl + trade_pnl
                    position_qty = self.portfolio.positions.get(symbol, 0.0)
                    target_holding_value = target_holdings.get(symbol, 0.0)
                    trade_value = trade_value_by_symbol.get(symbol, 0.0)

                    self.detailed_history.append({
                        'timestamp': current_date,
                        'symbol': symbol,
                        'price': current_price,
                        'current_factor': current_factor,
                        'prev_factor': prev_factor,
                        'factor_change': factor_change,
                        'opening_value': old_holding_value,
                        'target_holding_value': target_holding_value,
                        'trade_value': trade_value,
                        'position_qty': position_qty,
                        'holding_value': current_holding_value,
                        'holding_pnl': holding_pnl,
                        'trade_pnl': trade_pnl,
                        'total_pnl': total_symbol_pnl,
                        'portfolio_total_value': self.portfolio.total_value,
                        'portfolio_daily_pnl': self.portfolio.total_value - old_total_value,
                    })
                
            except Exception as e:
                logger.warning(f"Error on {current_date}: {e}")
                continue
        
        return self
    
    def calculate_metrics(self, risk_free_rate: float = 0.0) -> 'Backtester':
        """Calculate performance metrics."""
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
    
    # ==================== Internal Helper Methods ====================
    
    def _get_factor_data(self, factor: 'Factor', date) -> pd.Series:
        """
        Extract factor data for a specific date from 3-column format.
        
        OPTIMIZED: Direct DataFrame query without MultiIndex overhead.
        """
        if date is None:
            return pd.Series(dtype=float)
        
        try:
            date_data = factor.data[factor.data['timestamp'] == date]
            
            if date_data.empty:
                return pd.Series(dtype=float)
            
            result = date_data.set_index('symbol')['factor'].dropna()
            return result
            
        except Exception:
            return pd.Series(dtype=float)
    
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
        
        all_symbols = target_holdings.index.union(current_holdings.index)
        
        trade_values = target_holdings.reindex(all_symbols, fill_value=0) - \
                       current_holdings.reindex(all_symbols, fill_value=0)
                       
        valid_prices = prices.reindex(trade_values.index)
        trade_quantities = trade_values / valid_prices
        
        return trade_quantities.dropna().loc[lambda x: ~np.isclose(x, 0)]
    
    # ==================== Visualization and Reporting ====================
    
    def plot_equity(self, figsize: tuple = (12, 7), show_summary: bool = True) -> 'Backtester':
        """
        Plot equity curve with professional layout: equity (top) + drawdown (bottom) + turnover (third).
        """
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
        
        # Drawdown subplot
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
        
        # Turnover subplot
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
        
        # Hide x-axis labels for upper plots
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax_dd.get_xticklabels(), visible=False)
        
        plt.show()
        return self
    
    def summary(self) -> str:
        """Return concise performance summary as a string for plotting."""
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

    def calculate_total_transaction_cost(self) -> float:
        """Calculate the total transaction cost incurred during the backtest."""
        trade_log_df = self.portfolio.get_trade_log_df()
        if trade_log_df.empty:
            return 0.0
        return trade_log_df['cost'].sum()

    def get_daily_turnover_df(self) -> pd.DataFrame:
        """
        Calculate daily portfolio turnover.
        
        Turnover is defined as the total absolute dollar value of trades 
        (buys + sells) divided by the total portfolio value (NAV).
        
        Returns
        -------
        pd.DataFrame
            DataFrame with 'turnover' column, indexed by date.
        """
        trade_log_df = self.portfolio.get_trade_log_df()
        history_df = self.portfolio.get_history_df()
        
        if trade_log_df.empty or history_df.empty:
            return pd.DataFrame()
            
        # trade_value is positive for buy, negative for sell
        daily_trade_value = trade_log_df['trade_value'].abs().groupby(level='date').sum()
        
        # Total portfolio value (NAV) at the end of the day (used as denominator)
        daily_nav = history_df['total_value']
        
        combined = pd.DataFrame({
            'daily_trade_value': daily_trade_value,
            'daily_nav': daily_nav
        }).dropna()
        
        if combined.empty:
            return pd.DataFrame()
            
        combined['turnover'] = combined['daily_trade_value'] / combined['daily_nav']
        return combined[['turnover']]

    def show_daily_returns(self, top_days: int = 20, show_summary: bool = True) -> None:
        """Display daily returns by symbol in a clean format."""
        if not self.detailed_history:
            print("No detailed data available. Run backtest first.")
            return
        
        df = pd.DataFrame(self.detailed_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['pnl_pct'] = df['total_pnl'] / df['portfolio_total_value'] * 100
        
        print(f"\n{self.strategy_factor.name} - Daily Returns by Symbol")
        print("=" * 60)
        
        for symbol in sorted(df['symbol'].unique()):
            symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
            
            print(f"\n{symbol}")
            print("-" * 35)
            
            recent_data = symbol_data.tail(top_days)
            for _, row in recent_data.iterrows():
                date = row['timestamp'].strftime('%Y-%m-%d')
                pnl_pct = row['pnl_pct']
                total_pnl = row['total_pnl']
                
                if pd.notna(pnl_pct):
                    print(f"{date}: {pnl_pct:+6.2f}% (${total_pnl:+8.2f})")
                else:
                    print(f"{date}: {'N/A':>6s}% (${total_pnl:+8.2f})")
            
            if len(symbol_data) > top_days:
                print(f"... ({len(symbol_data)-top_days} more days)")
        
        if show_summary:
            print(f"\nSymbol Performance Summary")
            print("-" * 40)
            for symbol in sorted(df['symbol'].unique()):
                symbol_data = df[df['symbol'] == symbol]
                total_pnl = symbol_data['total_pnl'].sum()
                total_return_pct = symbol_data['pnl_pct'].sum()
                avg_daily_pct = symbol_data['pnl_pct'].mean()
                best_day_pct = symbol_data['pnl_pct'].max()
                worst_day_pct = symbol_data['pnl_pct'].min()
                trading_days = len(symbol_data)
                
                print(f"{symbol:>6}: ${total_pnl:+8.1f} | {total_return_pct:+6.2f}% | "
                      f"日均{avg_daily_pct:+.2f}% | 最佳{best_day_pct:+.2f}% | "
                      f"最糟{worst_day_pct:+.2f}% | {trading_days}天")

    def get_symbol_daily_returns(self) -> pd.DataFrame:
        """Get daily returns data structured by symbol."""
        if not self.detailed_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.detailed_history)
        df['pnl_pct'] = df['total_pnl'] / df['portfolio_total_value'] * 100
        
        return df[['timestamp', 'symbol', 'pnl_pct', 'total_pnl', 'factor_change', 
                  'position_qty', 'holding_value']].sort_values(['symbol', 'timestamp'])

    def get_detailed_data(self) -> pd.DataFrame:
        """Get comprehensive daily data per symbol with MultiIndex (timestamp, symbol)."""
        if not self.detailed_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.detailed_history)
        df['pnl_pct'] = df['total_pnl'] / df['portfolio_total_value'] * 100
        df['portfolio_pnl_pct'] = df['portfolio_daily_pnl'] / df['portfolio_total_value'] * 100
        df = df.set_index(['timestamp', 'symbol']).sort_index()
        
        return df

    def export_detailed_data(self, filename: str = None) -> str:
        """Export comprehensive backtest data to CSV file."""
        if not filename:
            strategy_name = self.strategy_factor.name.replace(' ', '_').replace('+', 'plus')
            filename = f"backtest_detailed_{strategy_name}.csv"
        
        detailed_df = self.get_detailed_data()
        
        if detailed_df.empty:
            raise ValueError("No detailed data available. Make sure to run the backtest first.")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            f.write(f"# Backtest Detailed Export\n")
            f.write(f"# Strategy: {self.strategy_factor.name}\n")
            f.write(f"# Price Factor: {self.price_factor.name}\n")
            f.write(f"# Initial Capital: {self.portfolio.initial_capital:,.0f}\n")
            f.write(f"# Transaction Cost: {self.transaction_cost_rates[0]:.4f}, {self.transaction_cost_rates[1]:.4f}\n")
            if self.metrics:
                f.write(f"# Total Return: {self.metrics.get('total_return', 0):.2%}\n")
                f.write(f"# Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.3f}\n")
            f.write(f"# Columns: timestamp,symbol,price,current_factor,prev_factor,factor_change,position_qty,holding_value,holding_pnl,trade_pnl,total_pnl,pnl_pct,portfolio_total_value,portfolio_daily_pnl,portfolio_pnl_pct\n")
            f.write(f"#\n")
        
        detailed_df.to_csv(filename, mode='a')
        
        print(f"Detailed backtest data exported to: {filename}")
        print(f"Records: {len(detailed_df):,} rows")
        print(f"Date range: {detailed_df.index.get_level_values('timestamp').min()} to {detailed_df.index.get_level_values('timestamp').max()}")
        print(f"Symbols: {', '.join(sorted(detailed_df.index.get_level_values('symbol').unique()))}")
        
        return filename

    def __add__(self, other: 'Backtester') -> 'Backtester':
        """Combine two backtest results with equal-weight capital allocation."""
        if not isinstance(other, Backtester):
            raise TypeError("Can only combine Backtester objects")
        
        hist1 = self.portfolio.get_history_df()
        hist2 = other.portfolio.get_history_df()
        
        if hist1.empty or hist2.empty:
            raise ValueError("Both backtests must have historical data")
        
        all_dates = sorted(hist1.index.union(hist2.index))
        if len(all_dates) < 2:
            raise ValueError("Insufficient total dates for combination")
        
        combined = Backtester(
            self.price_factor, 
            self.strategy_factor,
            self.transaction_cost_rates[0],
            self.portfolio.initial_capital + other.portfolio.initial_capital
        )
        
        combined.portfolio.history = []
        cumulative_value = self.portfolio.initial_capital + other.portfolio.initial_capital
        
        for date in all_dates:
            has_hist1 = date in hist1.index
            has_hist2 = date in hist2.index
            total_daily_pnl = 0
            total_cash = 0
            total_holdings = 0
            
            if has_hist1:
                if 'daily_pnl' in hist1.columns:
                    pnl1 = hist1.loc[date, 'daily_pnl']
                    if not pd.isna(pnl1):
                        total_daily_pnl += pnl1
                total_cash += hist1.loc[date, 'cash']
                total_holdings += hist1.loc[date, 'holdings_value']
            
            if has_hist2:
                if 'daily_pnl' in hist2.columns:
                    pnl2 = hist2.loc[date, 'daily_pnl']
                    if not pd.isna(pnl2):
                        total_daily_pnl += pnl2
                total_cash += hist2.loc[date, 'cash']
                total_holdings += hist2.loc[date, 'holdings_value']
            
            cumulative_value += total_daily_pnl
            
            combined.portfolio.history.append({
                'date': date,
                'total_value': cumulative_value,
                'daily_pnl': total_daily_pnl,
                'cash': total_cash,
                'holdings_value': total_holdings,
            })
        
        if combined.portfolio.history:
            combined.portfolio.total_value = combined.portfolio.history[-1]['total_value']
        
        combined.calculate_metrics()
        combined.strategy_factor.name = f"{self.strategy_factor.name} + {other.strategy_factor.name}"
        combined.detailed_history = self.detailed_history + other.detailed_history
        combined.detailed_history.sort(key=lambda x: (x['timestamp'], x['symbol']))
        
        return combined

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
    transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
    initial_capital: float = 100000,
    auto_run: bool = True
) -> Backtester:
    """
    Convenience function for quick backtesting with 3-column Factor format.
    
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