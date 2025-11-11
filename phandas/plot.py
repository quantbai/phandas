"""Plotting utilities for backtesting results and factor analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .backtest import Backtester, CombinedBacktester
    from .core import Factor

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

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


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


class BacktestPlotter:
    """Plotter for Backtester and CombinedBacktester results."""
    
    def __init__(self, backtester: 'Backtester'):
        """Initialize with a Backtester instance."""
        self.bt = backtester
    
    def plot_equity(self, figsize: tuple = (12, 7), show_summary: bool = True, 
                   show_benchmark: bool = True) -> None:
        """Plot equity, drawdown, turnover, summary, and benchmark."""
        history = self.bt.portfolio.get_history_df()
        if history.empty:
            return
        
        equity_curve = history['total_value']
        equity_norm = equity_curve / equity_curve.iloc[0]
        rolling_max = equity_norm.cummax()
        drawdown = equity_norm / rolling_max - 1.0
        
        benchmark_series = None
        benchmark_norm = None
        if show_benchmark:
            benchmark_series = self.bt._calculate_benchmark_equity()
            if not benchmark_series.empty and len(benchmark_series) > 0:
                benchmark_norm = benchmark_series / benchmark_series.iloc[0]
        
        turnover_df = self.bt.get_daily_turnover_df()
        
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
            benchmark_abs = benchmark_norm * self.bt.portfolio.initial_capital
            y_min = min(y_min, benchmark_abs.min())
            ax.plot(benchmark_norm.index, benchmark_abs, color=_PLOT_COLORS['benchmark_line'], 
                   linewidth=_PLOT_STYLES['benchmark_linewidth'], alpha=_PLOT_STYLES['benchmark_alpha'], linestyle='--')
        
        ax.set_title(f'Equity Curve ({self.bt.strategy_factor.name})', 
                    fontsize=_PLOT_STYLES['title_size'], fontweight='400', color=_PLOT_COLORS['text'], pad=14)
        ax.set_ylabel('Equity Value', fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_light'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.grid(True, alpha=_PLOT_STYLES['grid_alpha'], color=_PLOT_COLORS['grid'], linestyle='-', linewidth=_PLOT_STYLES['grid_width'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=_PLOT_STYLES['label_size'], colors=_PLOT_COLORS['text_light'], 
                      width=_PLOT_STYLES['spine_width'], length=_PLOT_STYLES['tick_length'])
        
        if show_summary:
            summary_text = self.bt.summary()
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


class CombinedBacktestPlotter:
    """Plotter for CombinedBacktester results."""
    
    def __init__(self, combined_bt: 'CombinedBacktester'):
        """Initialize with a CombinedBacktester instance."""
        self.cbt = combined_bt
    
    def plot_equity(self, figsize: tuple = (12, 7), show_summary: bool = True) -> None:
        """Plot portfolio equity, drawdown, and summary."""
        equity = self.cbt.get_portfolio_equity()
        if equity.empty or len(equity) < 2:
            return
        
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
        
        strategy_names = ", ".join([bt.strategy_factor.name for bt in self.cbt.backtests])
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
            summary_text = self.cbt.summary()
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


class FactorPlotter:
    """Plotter for Factor data visualization."""
    
    def __init__(self, factor: 'Factor'):
        """Initialize with a Factor instance."""
        self.factor = factor
    
    def plot(self, symbol: Optional[str] = None, figsize: tuple = (12, 6), 
             title: Optional[str] = None) -> None:
        """Plot factor over time. symbol=None plots all symbols in subgrid."""
        if symbol is None:
            self._plot_all_symbols(figsize, title)
        else:
            self._plot_single_symbol(symbol, figsize, title)
    
    def _plot_single_symbol(self, symbol: str, figsize: tuple, title: Optional[str]) -> None:
        """Plot factor values for single symbol."""
        data = self.factor.data[self.factor.data['symbol'] == symbol].copy()
        if data.empty:
            print(f"No data found for symbol: {symbol}")
            return
        
        data = data.sort_values('timestamp')
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#fcfcfc')
        
        ax.plot(data['timestamp'], data['factor'], color='#2563eb', linewidth=1.2, alpha=0.8)
        ax.fill_between(data['timestamp'], data['factor'], alpha=0.15, color='#2563eb')
        
        plot_title = title or f'{self.factor.name} ({symbol})'
        ax.set_title(plot_title, fontsize=12.5, fontweight='400', color='#1f2937', pad=14)
        ax.set_xlabel('Date', fontsize=10.5, color='#6b7280')
        ax.set_ylabel('Factor Value', fontsize=10.5, color='#6b7280')
        ax.grid(True, alpha=0.15, color='#e5e7eb', linestyle='-', linewidth=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=9.5, colors='#6b7280', 
                      width=0.5, length=3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_all_symbols(self, figsize: tuple, title: Optional[str]) -> None:
        """Plot factor values for all symbols."""
        symbols = sorted(self.factor.data['symbol'].unique())
        n_symbols = len(symbols)
        
        if n_symbols == 0:
            print("No data to plot")
            return
        
        n_cols = min(3, n_symbols)
        n_rows = (n_symbols + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
        if n_symbols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten() if n_symbols > 1 else np.array([axes])
        
        colors = ['#2563eb', '#059669', '#dc2626', '#7c3aed', '#f59e0b', '#06b6d4']
        
        for idx, symbol in enumerate(symbols):
            ax = axes[idx]
            data = self.factor.data[self.factor.data['symbol'] == symbol].copy()
            data = data.sort_values('timestamp')
            
            color = colors[idx % len(colors)]
            ax.plot(data['timestamp'], data['factor'], color=color, linewidth=1.2, alpha=0.8)
            ax.fill_between(data['timestamp'], data['factor'], alpha=0.15, color=color)
            
            ax.set_title(symbol, fontsize=11, fontweight='500', color='#1f2937')
            ax.set_xlabel('Date', fontsize=9.5, color='#6b7280')
            ax.set_ylabel('Factor Value', fontsize=9.5, color='#6b7280')
            ax.grid(True, alpha=0.12, color='#e5e7eb', linestyle='-', linewidth=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=8.5, colors='#6b7280',
                          width=0.5, length=3)
            ax.set_facecolor('#fcfcfc')
            
            dates = data['timestamp'].values
            n_dates = len(dates)
            if n_dates > 2:
                tick_indices = [0, n_dates // 2, n_dates - 1]
                ax.set_xticks([dates[i] for i in tick_indices])
        
        for idx in range(n_symbols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.show()

