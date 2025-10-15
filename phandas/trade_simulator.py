"""
Trade replay simulator for visualizing day-by-day factor strategy execution.

Leverages the Backtester engine to run a short, detailed backtest 
and prints the results in a concise, human-readable terminal output.
"""

import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.rule import Rule
from rich import box

from .backtest import Backtester
from .core import Factor

if TYPE_CHECKING:
    from .core import Factor


def simulate_trade_replay(
    price_factor: Factor,
    strategy_factor: Factor,
    n_days: int = 7,
    initial_capital: float = 100000,
    transaction_cost: float = 0.0003
):
    """
    Simulates and prints the day-by-day trade execution of a factor strategy.
    
    Parameters
    ----------
    price_factor : Factor
        Price factor (e.g., Close price).
    strategy_factor : Factor
        Factor used for calculating strategy weights.
    n_days : int, default 7
        Number of most recent days to simulate and display.
    initial_capital : float, default 100000
        Starting capital.
    transaction_cost : float, default 0.0003
        Per-trade transaction cost rate.
    """
    
    all_dates = sorted(price_factor.data['timestamp'].unique())
    
    if len(all_dates) < n_days + 1:
        print(f"Warning: Only {len(all_dates)} days available, less than requested {n_days}. Using all available data.")
        start_date_idx = 0
    else:
        # We need n_days of trading + 1 day lookback for the factor
        start_date_idx = len(all_dates) - n_days - 1 

    start_date = all_dates[start_date_idx]
    
    price_data_truncated = price_factor.data[
        price_factor.data['timestamp'] >= start_date
    ].copy()
    strategy_data_truncated = strategy_factor.data[
        strategy_factor.data['timestamp'] >= start_date
    ].copy()
    
    temp_price_factor = Factor(
        name=f"Price_{price_factor.name}", 
        data=price_data_truncated
    )
    temp_strategy_factor = Factor(
        name=f"Strategy_{strategy_factor.name}", 
        data=strategy_data_truncated
    )
    
    bt = Backtester(temp_price_factor, temp_strategy_factor, transaction_cost, initial_capital)
    bt.run().calculate_metrics()
    
    history_df = bt.portfolio.get_history_df()
    detailed_df = bt.get_detailed_data().reset_index()
    
    if history_df.empty:
        print("Simulation failed or returned no data.")
        return

    _print_replay(history_df, detailed_df, n_days, strategy_factor.name, bt)


def _print_replay(history_df, detailed_df, n_days, strategy_name, bt):
    """Helper to format and print the trade replay using rich."""
    
    console = Console()

    console.print(Panel(
        f"[bold cyan]{strategy_name}[/bold cyan]",
        title=f"N-Day Trade Replay Simulation ({n_days} days)",
        expand=False,
        border_style="bold blue"
    ))
    
    # Initial state (one day before the first trading day)
    initial_date = history_df.index[0] - pd.DateOffset(days=1)
    initial_cash = history_df['cash'].iloc[0]
    initial_nav = history_df['total_value'].iloc[0]

    initial_panel = Panel(
        f"Capital: [bold green]${initial_cash:,.2f}[/bold green]\n"
        f"NAV:     [bold green]${initial_nav:,.2f}[/bold green]",
        title=f"Initial State ({initial_date.strftime('%Y-%m-%d')})",
        border_style="green",
        expand=False
    )
    console.print(initial_panel)

    # Loop through each trading day
    for i in range(1, len(history_df)):
        date = history_df.index[i]
        
        day_history = history_df.iloc[i]
        prev_day_history = history_df.iloc[i - 1]
        
        day_pnl = day_history['daily_pnl']
        day_total_value = day_history['total_value']
        
        prev_day_total_value = prev_day_history['total_value']
        pnl_percent = (day_pnl / prev_day_total_value * 100) if prev_day_total_value != 0 else 0

        pnl_color = "green" if day_pnl >= 0 else "red"
        pnl_sign = "+" if day_pnl >= 0 else ""
        
        console.print(Rule(f"[bold]DAY {i} ({date.strftime('%Y-%m-%d')})[/bold]"))
        
        summary_text = Text.from_markup(
            f"NAV End: [bold]${day_total_value:,.2f}[/] | PnL: [{pnl_color}]{pnl_sign}${day_pnl:,.2f} ({pnl_sign}{pnl_percent:.2f}%)[/{pnl_color}]"
        )
        console.print(Align.center(summary_text))
        
        day_details = detailed_df[detailed_df['timestamp'] == date]
        
        total_holding_pnl = day_details['holding_pnl'].sum()
        total_trade_pnl = day_details['trade_pnl'].sum()
        total_opening_value = day_details['opening_value'].sum()
        total_trade_value = day_details['trade_value'].sum()
        total_holding_value = day_details['holding_value'].sum()

        total_opening_value_gross = day_details['opening_value'].abs().sum()
        total_trade_value_gross = day_details['trade_value'].abs().sum()
        total_long_holdings = day_details[day_details['holding_value'] > 0]['holding_value'].sum()
        total_short_holdings = day_details[day_details['holding_value'] < 0]['holding_value'].sum()

        table = Table(
            show_header=True, 
            header_style="bold magenta", 
            box=box.SQUARE, 
            title="[dim]Positions & PnL[/dim]",
            show_footer=True
        )
        
        table.add_column("Symbol", style="cyan", no_wrap=True, footer=Text("TOTALS", style="bold"))
        
        total_h_pnl_text = Text(f"${total_holding_pnl:+.2f}", style="bold green" if total_holding_pnl >= 0 else "bold red")
        table.add_column("持倉損益 (U)", justify="right", footer=total_h_pnl_text)

        total_opening_value_text = Text(f"${total_opening_value_gross:,.2f}", style="bold")
        table.add_column("原始持倉 (U)", justify="right", footer=total_opening_value_text)
        
        table.add_column("昨日因子", justify="right")
        table.add_column("目標持倉 (U)", justify="right")
        
        total_trade_value_text = Text(f"${total_trade_value_gross:,.2f}", style="bold")
        table.add_column("交易指令 (U)", justify="right", footer=total_trade_value_text)

        total_t_pnl_text = Text(f"${total_trade_pnl:,.2f}", style="bold red")
        table.add_column("交易成本 (U)", justify="right", footer=total_t_pnl_text)

        total_long_holdings_text = Text(f"${total_long_holdings:,.2f}", style="bold green")
        table.add_column("多頭持倉 (U)", justify="right", footer=total_long_holdings_text)

        total_short_holdings_text = Text(f"${total_short_holdings:,.2f}", style="bold red")
        table.add_column("空頭持倉 (U)", justify="right", footer=total_short_holdings_text)
        
        for _, row in day_details.iterrows():
            if np.isclose(row['opening_value'], 0.0) and np.isclose(row['trade_value'], 0.0) and np.isclose(row['holding_value'], 0.0):
                continue

            h_pnl = row['holding_pnl']
            h_pnl_text = Text(f"${h_pnl:+.2f}", style="green" if h_pnl >= 0 else "red")
            
            trade_value = row['trade_value']
            trade_text = f"${trade_value:,.2f}"
            if trade_value > 0:
                trade_text = Text(trade_text, style="green")
            elif trade_value < 0:
                trade_text = Text(trade_text, style="red")

            t_pnl = row['trade_pnl']
            t_pnl_text = Text(f"${t_pnl:,.2f}", style="red" if t_pnl < 0 else "")

            long_holding_value = row['holding_value'] if row['holding_value'] > 0 else 0.0
            short_holding_value = row['holding_value'] if row['holding_value'] < 0 else 0.0

            table.add_row(
                row['symbol'],
                h_pnl_text,
                f"${row['opening_value']:,.2f}",
                f"{row['prev_factor']:.4f}",
                f"${row['target_holding_value']:,.2f}",
                trade_text,
                t_pnl_text,
                f"${long_holding_value:,.2f}",
                f"${short_holding_value:,.2f}",
            )
        
        console.print(table)
        console.print()

    # Final Summary
    metrics = bt.metrics
    total_return = metrics.get('total_return', 0)
    annual_return = metrics.get('annual_return', 0)
    volatility = metrics.get('annual_volatility', 0)
    sharpe = metrics.get('sharpe_ratio', 0)
    mdd = metrics.get('max_drawdown', 0)
    calmar = metrics.get('calmar_ratio', 0)

    turnover_df = bt.get_daily_turnover_df()
    avg_annual_turnover = (turnover_df['turnover'].mean() * 365) if not turnover_df.empty else 0

    summary_content = (
        f"Total Return:         [bold {'green' if total_return >= 0 else 'red'}]{total_return: >8.2%}[/]\n"
        f"Annual Return:        [bold {'green' if annual_return >= 0 else 'red'}]{annual_return: >8.2%}[/]\n"
        f"Annual Volatility:    [bold yellow]{volatility: >8.2%}[/]\n"
        f"Sharpe Ratio:         [bold cyan]{sharpe: >8.2f}[/]\n"
        f"Max Drawdown:         [bold red]{mdd: >8.2%}[/]\n"
        f"Calmar Ratio:         [bold cyan]{calmar: >8.2f}[/]\n"
        f"Avg. Annual Turnover: [bold yellow]{avg_annual_turnover: >8.2%}[/]"
    )
    
    summary_panel = Panel(
        summary_content,
        title="Replay Completed",
        border_style="bold blue",
        expand=False
    )
    console.print(summary_panel)
