"""
Trade replay simulator for visualizing day-by-day factor strategy execution.

Leverages the Backtester engine to run a short, detailed backtest 
and prints the results in a concise, human-readable terminal output.
"""

import pandas as pd
import numpy as np
from typing import TYPE_CHECKING
from dataclasses import dataclass

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


@dataclass
class DayMetrics:
    """單日交易指標"""
    total_holding_pnl: float
    total_trade_pnl: float
    total_opening_value_gross: float
    total_trade_value_gross: float
    total_long_holdings: float
    total_short_holdings: float


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
    
    # 驗證資料充足性
    if len(all_dates) < n_days + 1:
        print(f"Warning: Only {len(all_dates)} days available, less than requested {n_days}. Using all available data.")
        start_date_idx = 0
    else:
        # 需要 n_days + 1 天的資料（包含前一天的初始狀態）
        start_date_idx = len(all_dates) - n_days - 1 
    
    start_date = all_dates[start_date_idx]
    
    # 創建截斷的因子資料
    temp_price_factor = _create_truncated_factor(price_factor, start_date, "Price")
    temp_strategy_factor = _create_truncated_factor(strategy_factor, start_date, "Strategy")
    
    # 執行回測
    bt = Backtester(temp_price_factor, temp_strategy_factor, transaction_cost, initial_capital)
    bt.run().calculate_metrics()
    
    history_df = bt.portfolio.get_history_df()
    detailed_df = bt.get_detailed_data().reset_index()
    
    if history_df.empty:
        print("Simulation failed or returned no data.")
        return

    _print_replay(history_df, detailed_df, n_days, strategy_factor.name, bt)


def _create_truncated_factor(factor: Factor, start_date, prefix: str) -> Factor:
    """創建截斷後的因子"""
    truncated_data = factor.data[factor.data['timestamp'] >= start_date].copy()
    return Factor(name=f"{prefix}_{factor.name}", data=truncated_data)


def _calculate_day_metrics(day_details: pd.DataFrame) -> DayMetrics:
    """計算單日彙總指標"""
    return DayMetrics(
        total_holding_pnl=day_details['holding_pnl'].sum(),
        total_trade_pnl=day_details['trade_pnl'].sum(),
        total_opening_value_gross=day_details['opening_value'].abs().sum(),
        total_trade_value_gross=day_details['trade_value'].abs().sum(),
        total_long_holdings=day_details[day_details['holding_value'] > 0]['holding_value'].sum(),
        total_short_holdings=day_details[day_details['holding_value'] < 0]['holding_value'].sum()
    )


def _get_colored_value(value: float, positive_color: str = "green", negative_color: str = "red") -> Text:
    """根據正負值返回帶顏色的 Text"""
    style = positive_color if value >= 0 else negative_color
    sign = "+" if value >= 0 else ""
    return Text(f"${value:,.2f}", style=style) if sign == "" else Text(f"{sign}${value:,.2f}", style=style)


def _should_skip_row(row: pd.Series) -> bool:
    """判斷是否應跳過該行（所有值接近 0）"""
    return (np.isclose(row['opening_value'], 0.0) and 
            np.isclose(row['trade_value'], 0.0) and 
            np.isclose(row['holding_value'], 0.0))


def _create_position_table(day_details: pd.DataFrame, metrics: DayMetrics) -> Table:
    """創建持倉與損益表格"""
    table = Table(
        show_header=True, 
        header_style="bold magenta", 
        box=box.SQUARE, 
        title="[dim]Positions & PnL[/dim]",
        show_footer=True
    )
    
    # 定義欄位
    table.add_column("Symbol", style="cyan", no_wrap=True, footer=Text("TOTALS", style="bold"))
    table.add_column("Holding PnL", justify="right", 
                    footer=_get_colored_value(metrics.total_holding_pnl))
    table.add_column("Opening Pos", justify="right", 
                    footer=Text(f"${metrics.total_opening_value_gross:,.2f}", style="bold"))
    table.add_column("Prev Factor", justify="right")
    table.add_column("Target Pos", justify="right")
    table.add_column("Trade Value", justify="right", 
                    footer=Text(f"${metrics.total_trade_value_gross:,.2f}", style="bold"))
    table.add_column("Trade Cost", justify="right", 
                    footer=Text(f"${metrics.total_trade_pnl:,.2f}", style="bold red"))
    table.add_column("Long Pos", justify="right", 
                    footer=Text(f"${metrics.total_long_holdings:,.2f}", style="bold green"))
    table.add_column("Short Pos", justify="right", 
                    footer=Text(f"${metrics.total_short_holdings:,.2f}", style="bold red"))
    
    # 填充資料行
    for _, row in day_details.iterrows():
        if _should_skip_row(row):
            continue
        
        long_value = max(row['holding_value'], 0.0)
        short_value = min(row['holding_value'], 0.0)
        
        table.add_row(
            row['symbol'],
            _get_colored_value(row['holding_pnl']),
            f"${row['opening_value']:,.2f}",
            f"{row['prev_factor']:.4f}",
            f"${row['target_holding_value']:,.2f}",
            _get_colored_value(row['trade_value']),
            Text(f"${row['trade_pnl']:,.2f}", style="red" if row['trade_pnl'] < 0 else ""),
            f"${long_value:,.2f}",
            f"${short_value:,.2f}",
        )
    
    return table


def _print_initial_state(console: Console, history_df: pd.DataFrame):
    """印出初始狀態"""
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


def _print_day_summary(console: Console, day_idx: int, date, day_pnl: float, 
                      day_total_value: float, prev_total_value: float):
    """印出單日摘要"""
    pnl_percent = (day_pnl / prev_total_value * 100) if prev_total_value != 0 else 0
    pnl_color = "green" if day_pnl >= 0 else "red"
    pnl_sign = "+" if day_pnl >= 0 else ""
    
    console.print(Rule(f"[bold]DAY {day_idx} ({date.strftime('%Y-%m-%d')})[/bold]"))
    
    summary_text = Text.from_markup(
        f"NAV End: [bold]${day_total_value:,.2f}[/] | "
        f"PnL: [{pnl_color}]{pnl_sign}${day_pnl:,.2f} ({pnl_sign}{pnl_percent:.2f}%)[/{pnl_color}]"
    )
    console.print(Align.center(summary_text))


def _print_final_metrics(console: Console, bt: Backtester):
    """印出最終績效指標"""
    metrics = bt.metrics
    turnover_df = bt.get_daily_turnover_df()
    avg_annual_turnover = (turnover_df['turnover'].mean() * 365) if not turnover_df.empty else 0

    def format_metric(value, is_positive_good=True):
        if value >= 0:
            color = "green" if is_positive_good else "red"
        else:
            color = "red" if is_positive_good else "green"
        return f"[bold {color}]{value: >8.2%}[/]" if isinstance(value, float) and abs(value) < 100 else f"[bold {color}]{value: >8.2f}[/]"

    summary_content = (
        f"Total Return:         {format_metric(metrics.get('total_return', 0))}\n"
        f"Annual Return:        {format_metric(metrics.get('annual_return', 0))}\n"
        f"Annual Volatility:    [bold yellow]{metrics.get('annual_volatility', 0): >8.2%}[/]\n"
        f"Sharpe Ratio:         [bold cyan]{metrics.get('sharpe_ratio', 0): >8.2f}[/]\n"
        f"Max Drawdown:         {format_metric(metrics.get('max_drawdown', 0), is_positive_good=False)}\n"
        f"Calmar Ratio:         [bold cyan]{metrics.get('calmar_ratio', 0): >8.2f}[/]\n"
        f"Avg. Annual Turnover: [bold yellow]{avg_annual_turnover: >8.2%}[/]"
    )
    
    summary_panel = Panel(
        summary_content,
        title="Replay Completed",
        border_style="bold blue",
        expand=False
    )
    console.print(summary_panel)


def _print_replay(history_df, detailed_df, n_days, strategy_name, bt):
    """Helper to format and print the trade replay using rich."""
    
    console = Console()

    console.print(Panel(
        f"[bold cyan]{strategy_name}[/bold cyan]",
        title=f"N-Day Trade Replay Simulation ({n_days} days)",
        expand=False,
        border_style="bold blue"
    ))
    
    _print_initial_state(console, history_df)

    # 預先按日期分組，避免重複過濾
    detailed_by_date = detailed_df.groupby('timestamp')

    for i in range(1, len(history_df)):
        date = history_df.index[i]
        day_history = history_df.iloc[i]
        prev_day_history = history_df.iloc[i - 1]
        
        # 印出單日摘要
        _print_day_summary(
            console, i, date, 
            day_history['daily_pnl'], 
            day_history['total_value'],
            prev_day_history['total_value']
        )
        
        # 獲取當日詳細資料
        try:
            day_details = detailed_by_date.get_group(date)
        except KeyError:
            console.print("[yellow]No position details for this day[/yellow]\n")
            continue
        
        # 計算指標並創建表格
        metrics = _calculate_day_metrics(day_details)
        table = _create_position_table(day_details, metrics)
        
        console.print(table)
        console.print() 

    _print_final_metrics(console, bt)
