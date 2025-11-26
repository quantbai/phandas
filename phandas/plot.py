"""Plotting utilities for backtesting results and factor analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .backtest import Backtester, CombinedBacktester
    from .core import Factor

logger = logging.getLogger(__name__)

_DATE_FORMAT = '%Y-%m-%d'

_PLOT_COLORS = {
    'equity_fill': [(0.22, '#2563eb'), (0.12, '#3b82f6'), (0.06, '#60a5fa'), (0.03, '#93c5fd'), (0.015, '#dbeafe')],
    'equity_line': '#1e40af',
    'benchmark_line': '#f97316',
    'drawdown_fill': '#ef4444',
    'drawdown_line': '#991b1b',
    'background': '#ffffff',
    'white': '#ffffff',
    'text': '#111827',
    'text_light': '#4b5563',
    'text_muted': '#9ca3af',
    'text_info': '#374151',
    'grid': '#f3f4f6',
    'turnover_line': '#059669',
    'zero_line': '#6b7280',
}

_PLOT_STYLES = {
    'title_size': 14,
    'ylabel_size': 11,
    'xlabel_size': 11,
    'label_size': 10,
    'small_label_size': 9,
    'grid_alpha': 0.5,
    'grid_width': 0.8,
    'grid_alpha_secondary': 0.4,
    'spine_width': 0.8,
    'tick_length': 4,
    'linewidth': 1.5,
    'benchmark_linewidth': 1.5,
    'benchmark_alpha': 0.7,
    'thin_linewidth': 1.0,
    'line_alpha': 1.0,
    'box_alpha': 0.9,
    'fill_alpha': 0.3,
}

_TRANSLATIONS = {
    'en': {
        'equity_title': 'Equity Curve',
        'equity_ylabel': 'Equity Value',
        'drawdown_ylabel': 'Drawdown',
        'turnover_ylabel': 'Turnover',
        'date_xlabel': 'Date',
        'no_turnover': 'No Turnover Data',
        'benchmark_label': 'Benchmark',
        'equity_label': 'Equity',
        'strategy': 'Strategy',
        'period': 'Period',
        'total_return': 'Total Return',
        'annual_return': 'Annual Return',
        'sharpe': 'Sharpe Ratio',
        'psr': 'PSR',
        'sortino': 'Sortino Ratio',
        'calmar': 'Calmar Ratio',
        'linearity': 'Linearity',
        'max_dd': 'Max Drawdown',
        'var_95': 'VaR 95%',
        'cvar': 'CVaR',
        'turnover': 'Avg. Annual Turnover',
        'corr_matrix': 'Correlation Matrix',
        'weights': 'Strategy Weights',
        'to': 'to',
    },
    'zh': {
        'equity_title': '權益曲線',
        'equity_ylabel': '權益淨值',
        'drawdown_ylabel': '回撤幅度',
        'turnover_ylabel': '換手率',
        'date_xlabel': '日期',
        'no_turnover': '無換手率數據',
        'benchmark_label': '基準',
        'equity_label': '策略淨值',
        'strategy': '策略名稱',
        'period': '回測期間',
        'total_return': '總回報率',
        'annual_return': '年化回報率',
        'sharpe': '夏普比率',
        'psr': 'PSR (概率夏普)',
        'sortino': '索提諾比率',
        'calmar': '卡瑪比率',
        'linearity': '線性度',
        'max_dd': '最大回撤',
        'var_95': '風險價值 (95%)',
        'cvar': '條件風險價值',
        'turnover': '平均年化換手率',
        'corr_matrix': '相關係數矩陣',
        'weights': '策略權重',
        'to': '至',
    }
}

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def _apply_plot_style():
    """Apply default style and re-set custom fonts."""
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


def _plot_equity_line(ax, equity_series: pd.Series, y_min: float, label: str = 'Equity') -> None:
    """Layered equity curve visualization."""
    # Gradient Fill
    for alpha, color in _PLOT_COLORS['equity_fill']:
        ax.fill_between(equity_series.index, y_min, equity_series, alpha=alpha, color=color, interpolate=True)
    
    # Main Equity Line
    ax.plot(equity_series.index, equity_series, color=_PLOT_COLORS['equity_line'], 
           linewidth=_PLOT_STYLES['linewidth'], alpha=_PLOT_STYLES['line_alpha'], label=label)


def _plot_drawdown(ax, drawdown_series: pd.Series) -> None:
    """Drawdown visualization."""
    ax.fill_between(drawdown_series.index, 0, drawdown_series, 
                   color=_PLOT_COLORS['drawdown_fill'], alpha=_PLOT_STYLES['fill_alpha'], step='pre')
    ax.plot(drawdown_series.index, drawdown_series, color=_PLOT_COLORS['drawdown_line'], 
           linewidth=_PLOT_STYLES['thin_linewidth'])
    # Add zero line
    ax.axhline(0, color=_PLOT_COLORS['zero_line'], linewidth=0.8, linestyle='-', alpha=0.5)


class BacktestPlotter:
    """Plotter for Backtester and CombinedBacktester results."""
    
    def __init__(self, backtester: 'Backtester'):
        """Initialize with a Backtester instance."""
        self.bt = backtester
    
    def plot_equity(self, figsize: tuple = (14, 8), show_summary: bool = True, 
                   show_benchmark: bool = True, language: str = 'en') -> None:
        """Plot equity, drawdown, turnover, summary, and benchmark."""
        texts = _TRANSLATIONS.get(language, _TRANSLATIONS['en'])
        
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
        
        _apply_plot_style()
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        
        # Create GridSpec with 2 columns: Main plots (left) and Summary (right)
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], width_ratios=[5, 1])
        
        ax = fig.add_subplot(gs[0, 0])
        ax_dd = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_to = fig.add_subplot(gs[2, 0], sharex=ax)
        
        # Summary subplot (spans all rows in the right column)
        ax_summary = fig.add_subplot(gs[:, 1])
        ax_summary.axis('off')
        
        ax.set_facecolor(_PLOT_COLORS['background'])
        y_min = equity_curve.min()
        _plot_equity_line(ax, equity_curve, y_min, label=texts['equity_label'])
        
        if benchmark_norm is not None and len(benchmark_norm) > 0:
            benchmark_abs = benchmark_norm * self.bt.portfolio.initial_capital
            y_min = min(y_min, benchmark_abs.min())
            ax.plot(benchmark_norm.index, benchmark_abs, color=_PLOT_COLORS['benchmark_line'], 
                   linewidth=_PLOT_STYLES['benchmark_linewidth'], alpha=_PLOT_STYLES['benchmark_alpha'], 
                   linestyle='--', label=texts['benchmark_label'])
        
        # Add legend
        ax.legend(loc='upper left', frameon=True, framealpha=_PLOT_STYLES['box_alpha'], 
                  fontsize=_PLOT_STYLES['small_label_size'])
        
        ax.set_title(f"{texts['equity_title']} ({self.bt.strategy_factor.name})", 
                    fontsize=_PLOT_STYLES['title_size'], fontweight='400', color=_PLOT_COLORS['text'], pad=14)
        ax.set_ylabel(texts['equity_ylabel'], fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_light'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.grid(True, alpha=_PLOT_STYLES['grid_alpha'], color=_PLOT_COLORS['grid'], linestyle='-', linewidth=_PLOT_STYLES['grid_width'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=_PLOT_STYLES['label_size'], colors=_PLOT_COLORS['text_light'], 
                      width=_PLOT_STYLES['spine_width'], length=_PLOT_STYLES['tick_length'])
        
        if show_summary:
            summary_lines = [f"{texts['strategy']}: {self.bt.strategy_factor.name}"]
            if not equity_curve.empty:
                start = equity_curve.index[0].strftime(_DATE_FORMAT)
                end = equity_curve.index[-1].strftime(_DATE_FORMAT)
                summary_lines.append(f"{texts['period']}: {start} {texts['to']} {end}")
                
                metrics = self.bt.metrics
                if metrics:
                    summary_lines.extend([
                        f"{texts['total_return']}: {metrics.get('total_return', 0):.2%}",
                        f"{texts['annual_return']}: {metrics.get('annual_return', 0):.2%}",
                        f"{texts['sharpe']}: {metrics.get('sharpe_ratio', 0):.2f}",
                        f"{texts['psr']}: {metrics.get('psr', 0):.1%}",
                        f"{texts['sortino']}: {metrics.get('sortino_ratio', 0):.2f}",
                        f"{texts['calmar']}: {metrics.get('calmar_ratio', 0):.2f}",
                        f"{texts['linearity']}: {metrics.get('linearity', 0):.4f}",
                        f"{texts['max_dd']}: {metrics.get('max_drawdown', 0):.2%}",
                        f"{texts['var_95']}: {metrics.get('var_95', 0):.2%}",
                        f"{texts['cvar']}: {metrics.get('cvar', 0):.2%}",
                    ])
                
                if not turnover_df.empty:
                    avg_turnover = turnover_df['turnover'].mean() * 365
                    summary_lines.append(f"{texts['turnover']}: {avg_turnover:.2%}")
            
            summary_text = "\n".join(summary_lines)
            ax_summary.text(0.0, 0.98, summary_text, transform=ax_summary.transAxes, 
                            fontsize=_PLOT_STYLES['label_size'], 
                            verticalalignment='top', horizontalalignment='left', 
                            color=_PLOT_COLORS['text_info'])
        
        ax_dd.set_facecolor(_PLOT_COLORS['white'])
        _plot_drawdown(ax_dd, drawdown)
        ax_dd.set_ylabel(texts['drawdown_ylabel'], fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_light'])
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
            ax_to.set_ylabel(texts['turnover_ylabel'], fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_light'])
            ax_to.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
            ax_to.grid(True, alpha=_PLOT_STYLES['grid_alpha_secondary'], color=_PLOT_COLORS['grid'], linestyle='-', linewidth=_PLOT_STYLES['grid_width'])
        else:
            ax_to.text(0.5, 0.5, texts['no_turnover'], transform=ax_to.transAxes, 
                      ha='center', va='center', fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_muted'])
        
        ax_to.set_xlabel(texts['date_xlabel'], fontsize=_PLOT_STYLES['xlabel_size'], color=_PLOT_COLORS['text_light'])
        ax_to.spines['top'].set_visible(False)
        ax_to.spines['right'].set_visible(False)
        ax_to.tick_params(axis='both', which='major', labelsize=_PLOT_STYLES['small_label_size'], colors=_PLOT_COLORS['text_light'], 
                          width=_PLOT_STYLES['spine_width'], length=_PLOT_STYLES['tick_length'])
        
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax_dd.get_xticklabels(), visible=False)
        
        # Align y-labels
        fig.align_ylabels([ax, ax_dd, ax_to])
        
        plt.show()


class CombinedBacktestPlotter:
    """Plotter for CombinedBacktester results."""
    
    def __init__(self, combined_bt: 'CombinedBacktester'):
        """Initialize with a CombinedBacktester instance."""
        self.cbt = combined_bt
    
    def plot_equity(self, figsize: tuple = (14, 8), show_summary: bool = True, language: str = 'en') -> None:
        """Plot portfolio equity, drawdown, and summary."""
        texts = _TRANSLATIONS.get(language, _TRANSLATIONS['en'])
        
        equity = self.cbt.get_portfolio_equity()
        if equity.empty or len(equity) < 2:
            return
        
        equity_norm = equity / equity.iloc[0]
        rolling_max = equity_norm.cummax()
        drawdown = equity_norm / rolling_max - 1.0
        
        _apply_plot_style()
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        
        # Create GridSpec with 2 columns: Main plots (left) and Summary (right)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[5, 1])
        
        ax = fig.add_subplot(gs[0, 0])
        ax_dd = fig.add_subplot(gs[1, 0], sharex=ax)
        
        # Summary subplot (spans all rows in the right column)
        ax_summary = fig.add_subplot(gs[:, 1])
        ax_summary.axis('off')
        
        ax.set_facecolor(_PLOT_COLORS['background'])
        y_min = equity.min()
        _plot_equity_line(ax, equity, y_min, label=texts['equity_label'])
        
        strategy_names = ", ".join([bt.strategy_factor.name for bt in self.cbt.backtests])
        ax.set_title(f"{texts['equity_title']} ({strategy_names})", 
                    fontsize=_PLOT_STYLES['title_size'], fontweight='400', color=_PLOT_COLORS['text'], pad=14)
        ax.set_ylabel(texts['equity_ylabel'], fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_light'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.grid(True, alpha=_PLOT_STYLES['grid_alpha'], color=_PLOT_COLORS['grid'], linestyle='-', linewidth=_PLOT_STYLES['grid_width'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=_PLOT_STYLES['label_size'], colors=_PLOT_COLORS['text_light'], 
                      width=_PLOT_STYLES['spine_width'], length=_PLOT_STYLES['tick_length'])
        
        if show_summary:
            summary_lines = []
            
            summary_lines.append(f"{texts['strategy']}: {strategy_names}")
            
            if not equity.empty and len(equity) > 0:
                start_date = equity.index[0].strftime(_DATE_FORMAT)
                end_date = equity.index[-1].strftime(_DATE_FORMAT)
                summary_lines.append(f"{texts['period']}: {start_date} {texts['to']} {end_date}")
            
            metrics = self.cbt.metrics
            if metrics:
                summary_lines.extend([
                    f"{texts['total_return']}: {metrics.get('total_return', 0):.2%}",
                    f"{texts['annual_return']}: {metrics.get('annual_return', 0):.2%}",
                    f"{texts['sharpe']}: {metrics.get('sharpe_ratio', 0):.2f}",
                    f"{texts['sortino']}: {metrics.get('sortino_ratio', 0):.2f}",
                    f"{texts['calmar']}: {metrics.get('calmar_ratio', 0):.2f}",
                    f"{texts['linearity']}: {metrics.get('linearity', 0):.4f}",
                    f"{texts['max_dd']}: {metrics.get('max_drawdown', 0):.2%}",
                    f"{texts['var_95']}: {metrics.get('var_95', 0):.2%}",
                    f"{texts['cvar']}: {metrics.get('cvar', 0):.2%}",
                ])
            
            summary_lines.append("")
            summary_lines.append(f"{texts['weights']}:")
            for bt, w in zip(self.cbt.backtests, self.cbt.weights):
                summary_lines.append(f"  {bt.strategy_factor.name}: {w*100:.1f}%")
            
            corr = self.cbt.correlation_matrix()
            if not corr.empty and len(corr) > 1:
                summary_lines.append("")
                summary_lines.append(f"{texts['corr_matrix']}:")
                corr_str = corr.to_string(max_cols=None, max_rows=None, float_format=lambda x: f'{x:.4f}')
                summary_lines.extend(corr_str.split('\n'))
            
            summary_text = "\n".join(summary_lines)
            ax_summary.text(0.0, 0.98, summary_text, transform=ax_summary.transAxes, 
                            fontsize=_PLOT_STYLES['label_size'], 
                            verticalalignment='top', horizontalalignment='left', 
                            color=_PLOT_COLORS['text_info'])
        
        ax_dd.set_facecolor(_PLOT_COLORS['white'])
        _plot_drawdown(ax_dd, drawdown)
        ax_dd.set_ylabel(texts['drawdown_ylabel'], fontsize=_PLOT_STYLES['ylabel_size'], color=_PLOT_COLORS['text_light'])
        ax_dd.set_xlabel(texts['date_xlabel'], fontsize=_PLOT_STYLES['xlabel_size'], color=_PLOT_COLORS['text_light'])
        ax_dd.grid(True, alpha=_PLOT_STYLES['grid_alpha_secondary'], color=_PLOT_COLORS['grid'], linestyle='-', linewidth=_PLOT_STYLES['grid_width'])
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax_dd.spines['top'].set_visible(False)
        ax_dd.spines['right'].set_visible(False)
        ax_dd.tick_params(axis='both', which='major', labelsize=_PLOT_STYLES['small_label_size'], colors=_PLOT_COLORS['text_light'], 
                          width=_PLOT_STYLES['spine_width'], length=_PLOT_STYLES['tick_length'])
        
        plt.setp(ax.get_xticklabels(), visible=False)
        
        # Align y-labels
        fig.align_ylabels([ax, ax_dd])
        
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
            logger.warning(f"No data found for symbol: {symbol}")
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
            logger.warning("No data to plot")
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

