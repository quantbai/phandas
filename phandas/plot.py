"""Plotting utilities for backtesting results and factor analysis."""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .backtest import Backtester
    from .core import Factor
    from .analysis import FactorAnalyzer

_DATE_FORMAT = '%Y-%m-%d'

_PLOT_COLORS = {
    'equity_fill': '#3b82f6',
    'equity_line': '#1d4ed8',
    'benchmark_line': '#ea580c',
    'drawdown_fill': '#dc2626',
    'drawdown_line': '#b91c1c',
    'background': '#ffffff',
    'background_subtle': '#fafafa',
    'white': '#ffffff',
    'text': '#0f172a',
    'text_dark': '#020617',
    'text_light': '#1e293b',
    'text_muted': '#475569',
    'text_info': '#334155',
    'grid': '#e2e8f0',
    'grid_subtle': '#f1f5f9',
    'turnover_line': '#475569',
    'zero_line': '#94a3b8',
    'table_header': '#020617',
    'table_label': '#1e293b',
    'table_value': '#0f172a',
    'table_line': '#64748b',
    'table_line_light': '#94a3b8',
    'factor_palette': ['#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#f59e0b', '#06b6d4'],
}

_PLOT_STYLES = {
    'title_size': 16,
    'subtitle_size': 11,
    'ylabel_size': 11,
    'xlabel_size': 11,
    'label_size': 10,
    'small_label_size': 9.5,
    'table_fontsize': 10.5,
    'table_header_fontsize': 10.5,
    'legend_fontsize': 10,
    'ylabel_labelpad': 8,
    'xlabel_labelpad': 6,
    'grid_alpha': 0.4,
    'grid_width': 0.5,
    'grid_alpha_secondary': 0.35,
    'spine_width': 0.8,
    'spine_color': '#94a3b8',
    'tick_length': 4,
    'linewidth': 1.8,
    'benchmark_linewidth': 1.5,
    'benchmark_alpha': 0.85,
    'thin_linewidth': 1.2,
    'line_alpha': 1.0,
    'box_alpha': 0.95,
    'fill_alpha': 0.25,
    'drawdown_fill_alpha': 0.22,
    'table_row_height': 0.058,
    'table_line_width': 1.0,
    'table_header_line_width': 0.6,
    'factor_title_size': 12,
    'factor_label_size': 10,
    'factor_tick_size': 9,
    'factor_subgrid_title_size': 10.5,
    'factor_subgrid_label_size': 9,
    'factor_subgrid_tick_size': 8,
    'factor_grid_alpha': 0.15,
    'factor_grid_alpha_subgrid': 0.12,
    'factor_grid_width': 0.5,
    'factor_fill_alpha': 0.18,
    'factor_line_alpha': 0.9,
    'factor_title_pad': 12,
}

_TEXT_LABELS = {
    'equity_ylabel': 'Equity Value',
    'drawdown_ylabel': 'Drawdown',
    'turnover_ylabel': 'Turnover',
    'date_xlabel': 'Date',
    'no_turnover': 'No Turnover Data',
    'benchmark_label': 'Benchmark',
    'equity_label': 'Strategy',
    'strategy': 'Strategy',
    'period': 'Period',
    'total_return': 'Total Return',
    'annual_return': 'Annual Return',
    'sharpe': 'Sharpe Ratio',
    'linearity': 'Linearity',
    'max_dd': 'Max Drawdown',
    'turnover': 'Annual Turnover',
    'corr_matrix': 'Correlation Matrix',
    'weights': 'Strategy Weights',
    'to': 'to',
}


def _apply_plot_style() -> None:
    plt.style.use('default')
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Helvetica Neue', 'Arial', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'stixsans'
    plt.rcParams['axes.unicode_minus'] = False


def _plot_equity_line(ax, equity_series: pd.Series, y_min: float, label: str = 'Strategy') -> None:
    ax.fill_between(
        equity_series.index, y_min, equity_series, 
        alpha=_PLOT_STYLES['fill_alpha'], 
        color=_PLOT_COLORS['equity_fill'], 
        linewidth=0
    )
    ax.plot(
        equity_series.index, equity_series, 
        color=_PLOT_COLORS['equity_line'], 
        linewidth=_PLOT_STYLES['linewidth'], 
        alpha=_PLOT_STYLES['line_alpha'], 
        label=label
    )


def _plot_drawdown(ax, drawdown_series: pd.Series) -> None:
    ax.fill_between(
        drawdown_series.index, 0, drawdown_series, 
        color=_PLOT_COLORS['drawdown_fill'], 
        alpha=_PLOT_STYLES['drawdown_fill_alpha'], 
        step='pre',
        linewidth=0
    )
    ax.plot(
        drawdown_series.index, drawdown_series, 
        color=_PLOT_COLORS['drawdown_line'], 
        linewidth=_PLOT_STYLES['thin_linewidth'],
        alpha=0.9
    )
    ax.axhline(0, color=_PLOT_COLORS['zero_line'], linewidth=0.5, linestyle='-', alpha=0.6)


def _style_axis(ax, ylabel: str, is_bottom: bool = False, xlabel: str = None) -> None:
    ax.set_facecolor(_PLOT_COLORS['white'])
    ax.set_ylabel(
        ylabel, 
        fontsize=_PLOT_STYLES['ylabel_size'], 
        color=_PLOT_COLORS['text_light'], 
        labelpad=_PLOT_STYLES['ylabel_labelpad']
    )
    
    if is_bottom and xlabel:
        ax.set_xlabel(
            xlabel, 
            fontsize=_PLOT_STYLES['xlabel_size'], 
            color=_PLOT_COLORS['text_light'], 
            labelpad=_PLOT_STYLES['xlabel_labelpad']
        )
    
    ax.grid(
        True, 
        alpha=_PLOT_STYLES['grid_alpha'], 
        color=_PLOT_COLORS['grid'], 
        linestyle='-', 
        linewidth=_PLOT_STYLES['grid_width']
    )
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(_PLOT_STYLES['spine_color'])
        ax.spines[spine].set_linewidth(_PLOT_STYLES['spine_width'])
    
    ax.tick_params(
        axis='both', 
        which='major', 
        labelsize=_PLOT_STYLES['label_size'], 
        colors=_PLOT_COLORS['text_muted'],
        width=_PLOT_STYLES['spine_width'], 
        length=_PLOT_STYLES['tick_length']
    )


def _render_summary_table(ax, summary_data: List[tuple]) -> None:
    if not summary_data:
        return
    
    has_three_columns = any(len(row) == 3 for row in summary_data)
    
    if has_three_columns:
        cell_text = [[row[0], row[1], row[2] if len(row) > 2 else ''] for row in summary_data]
        num_cols = 3
        col_widths = [0.48, 0.26, 0.26]
    else:
        cell_text = [[row[0], row[1]] for row in summary_data]
        num_cols = 2
        col_widths = None
    
    num_rows = len(cell_text)
    
    ROW_HEIGHT = _PLOT_STYLES['table_row_height']
    table_height = num_rows * ROW_HEIGHT
    
    y_bottom = (1.0 - table_height) / 2
    
    bbox = [0.02, y_bottom, 0.96, table_height]
    
    table = ax.table(
        cellText=cell_text, 
        cellLoc='left', 
        loc='center',
        bbox=bbox, 
        edges='open'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(_PLOT_STYLES['table_fontsize'])
    
    if col_widths:
        for i, width in enumerate(col_widths):
            for row in range(num_rows):
                table[(row, i)].set_width(width)
    
    COLOR_HEADER = _PLOT_COLORS['table_header']
    COLOR_LABEL = _PLOT_COLORS['table_label']
    COLOR_VALUE = _PLOT_COLORS['table_value']
    COLOR_LINE = _PLOT_COLORS['table_line']
    COLOR_LINE_LIGHT = _PLOT_COLORS['table_line_light']
    
    fontsize = _PLOT_STYLES['table_fontsize']
    header_fontsize = _PLOT_STYLES['table_header_fontsize']
    
    for cell_key, cell in table.get_celld().items():
        row, col = cell_key
        
        cell.set_facecolor('none')
        cell.set_linewidth(0)
        cell.set_edgecolor('none')
        cell.PAD = 0.02
        
        if has_three_columns and row == 0:
            if col == 0:
                cell.set_text_props(
                    weight='medium', 
                    color=COLOR_HEADER, 
                    fontsize=header_fontsize, 
                    ha='left'
                )
            else:
                cell.set_text_props(
                    weight='medium', 
                    color=COLOR_HEADER, 
                    fontsize=header_fontsize, 
                    ha='right'
                )
        else:
            label_text = cell_text[row][0]
            
            is_spacer = all(not str(cell_text[row][i]).strip() for i in range(len(cell_text[row])))
            
            is_section_header = (
                label_text and len(cell_text[row]) >= 2 and 
                not cell_text[row][1] and label_text.strip() 
                and not label_text.startswith('  ')
            )
            
            if is_spacer:
                cell.set_text_props(fontsize=4)
            elif is_section_header and not has_three_columns:
                cell.set_text_props(
                    weight='medium', 
                    color=COLOR_HEADER, 
                    fontsize=fontsize, 
                    ha='left'
                )
            else:
                if col == 0:
                    cell.set_text_props(
                        weight='normal', 
                        color=COLOR_LABEL, 
                        fontsize=fontsize, 
                        ha='left'
                    )
                else:
                    cell.set_text_props(
                        weight='normal', 
                        color=COLOR_VALUE, 
                        fontsize=fontsize, 
                        ha='right'
                    )
    
    line_width = _PLOT_STYLES['table_line_width']
    header_line_width = _PLOT_STYLES['table_header_line_width']
    
    ax.plot(
        [0.02, 0.98], [y_bottom + table_height, y_bottom + table_height], 
        linewidth=line_width, 
        color=COLOR_LINE, 
        transform=ax.transAxes, 
        solid_capstyle='butt'
    )
    
    if has_three_columns:
        header_y = y_bottom + table_height - ROW_HEIGHT
        ax.plot(
            [0.02, 0.98], [header_y, header_y], 
            linewidth=header_line_width, 
            color=COLOR_LINE_LIGHT, 
            transform=ax.transAxes, 
            solid_capstyle='butt'
        )
    
    ax.plot(
        [0.02, 0.98], [y_bottom, y_bottom], 
        linewidth=line_width, 
        color=COLOR_LINE, 
        transform=ax.transAxes, 
        solid_capstyle='butt'
    )
    
    ax.axis('off')


class BacktestPlotter:
    """Equity curve and drawdown visualization for backtest results."""
    
    def __init__(self, backtester: 'Backtester'):
        self.bt = backtester
    
    def _calculate_benchmark_metrics(self, benchmark_norm: pd.Series, strategy_returns: pd.Series) -> Dict:
        if benchmark_norm.empty or len(benchmark_norm) < 2:
            return {}
        
        benchmark_returns = benchmark_norm.pct_change(fill_method=None).dropna()
        if benchmark_returns.empty or len(benchmark_returns) < 2:
            return {}
        
        bmk_total_return = benchmark_norm.iloc[-1] / benchmark_norm.iloc[0] - 1
        days = (benchmark_returns.index[-1] - benchmark_returns.index[0]).days
        bmk_annual_return = (1 + bmk_total_return) ** (365 / days) - 1 if days > 0 else 0
        
        bmk_annual_vol = benchmark_returns.std() * np.sqrt(365)
        risk_free_rate = 0.03
        bmk_sharpe = (bmk_annual_return - risk_free_rate) / bmk_annual_vol if bmk_annual_vol > 0 else 0
        
        rolling_max = benchmark_norm.cummax()
        drawdown = benchmark_norm / rolling_max - 1
        bmk_max_dd = drawdown.min()
        
        from scipy.stats import linregress
        t = np.arange(len(benchmark_norm))
        r_value = linregress(t, benchmark_norm.values)[2]
        bmk_linearity = r_value ** 2
        
        return {
            'bmk_total_return': bmk_total_return,
            'bmk_annual_return': bmk_annual_return,
            'bmk_sharpe': bmk_sharpe,
            'bmk_linearity': bmk_linearity,
            'bmk_max_drawdown': bmk_max_dd,
        }
    
    def plot_equity(self, figsize: tuple = (14, 7.5), show_summary: bool = True, 
                   show_benchmark: bool = True) -> None:
        texts = _TEXT_LABELS
        
        history = self.bt.portfolio.get_history_df()
        if history.empty:
            return
        
        equity_curve = history['total_value']
        equity_norm = equity_curve / equity_curve.iloc[0]
        rolling_max = equity_norm.cummax()
        drawdown = equity_norm / rolling_max - 1.0
        
        benchmark_series = None
        benchmark_norm = None
        benchmark_metrics = {}
        if show_benchmark:
            benchmark_series = self.bt._calculate_benchmark_equity()
            if not benchmark_series.empty and len(benchmark_series) > 0:
                benchmark_norm = benchmark_series / benchmark_series.iloc[0]
                strategy_returns = self.bt.returns
                if not strategy_returns.empty:
                    benchmark_metrics = self._calculate_benchmark_metrics(benchmark_norm, strategy_returns)
        
        turnover_df = self.bt.turnover
        
        _apply_plot_style()
        fig = plt.figure(figsize=figsize)
        
        fig.subplots_adjust(top=0.91, bottom=0.08, left=0.065, right=0.98, wspace=0.02, hspace=0.12)
        
        gs = fig.add_gridspec(3, 2, height_ratios=[3.5, 1, 1], width_ratios=[3, 1])
        
        ax = fig.add_subplot(gs[0, 0])
        ax_dd = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_to = fig.add_subplot(gs[2, 0], sharex=ax)
        
        ax_summary = fig.add_subplot(gs[:, 1])
        ax_summary.axis('off')
        
        y_min = equity_curve.min()
        _plot_equity_line(ax, equity_curve, y_min, label=texts['equity_label'])
        
        if benchmark_norm is not None and len(benchmark_norm) > 0:
            benchmark_abs = benchmark_norm * self.bt.portfolio.initial_capital
            y_min = min(y_min, benchmark_abs.min())
            ax.plot(
                benchmark_norm.index, benchmark_abs, 
                color=_PLOT_COLORS['benchmark_line'], 
                linewidth=_PLOT_STYLES['benchmark_linewidth'], 
                alpha=_PLOT_STYLES['benchmark_alpha'], 
                linestyle='--', 
                label=texts['benchmark_label']
            )
        
        ax.legend(
            loc='upper left', 
            frameon=False,
            fontsize=_PLOT_STYLES['legend_fontsize'],
            labelcolor=_PLOT_COLORS['text_muted']
        )
        
        fig.suptitle(
            self.bt.strategy_factor.name, 
            fontsize=_PLOT_STYLES['title_size'], 
            fontweight='500', 
            color=_PLOT_COLORS['text_dark'], 
            y=0.97
        )
        
        if not equity_curve.empty:
            start = equity_curve.index[0].strftime(_DATE_FORMAT)
            end = equity_curve.index[-1].strftime(_DATE_FORMAT)
            period_text = f"{start} {texts['to']} {end}"
            fig.text(
                0.5, 0.935, period_text, 
                fontsize=_PLOT_STYLES['subtitle_size'], 
                color=_PLOT_COLORS['text_muted'], 
                ha='center', va='top'
            )
        
        _style_axis(ax, texts['equity_ylabel'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        if show_summary:
            metrics = self.bt.metrics
            
            if benchmark_metrics:
                summary_data = [
                    ('Metric', 'Strategy', 'Benchmark'),
                ]
                
                if metrics:
                    avg_turnover = turnover_df['turnover'].mean() * 365 if not turnover_df.empty else 0
                    
                    summary_data.extend([
                        (texts['total_return'], f"{metrics.get('total_return', 0):.2%}", 
                         f"{benchmark_metrics.get('bmk_total_return', 0):.2%}"),
                        (texts['annual_return'], f"{metrics.get('annual_return', 0):.2%}", 
                         f"{benchmark_metrics.get('bmk_annual_return', 0):.2%}"),
                        (texts['sharpe'], f"{metrics.get('sharpe_ratio', 0):.2f}", 
                         f"{benchmark_metrics.get('bmk_sharpe', 0):.2f}"),
                        (texts['linearity'], f"{metrics.get('linearity', 0):.4f}", 
                         f"{benchmark_metrics.get('bmk_linearity', 0):.4f}"),
                        (texts['max_dd'], f"{metrics.get('max_drawdown', 0):.2%}", 
                         f"{benchmark_metrics.get('bmk_max_drawdown', 0):.2%}"),
                        (texts['turnover'], f"{avg_turnover:.2%}", '-'),
                    ])
            else:
                summary_data = []
                
                if metrics:
                    avg_turnover = turnover_df['turnover'].mean() * 365 if not turnover_df.empty else 0
                    
                    summary_data.extend([
                        (texts['total_return'], f"{metrics.get('total_return', 0):.2%}"),
                        (texts['annual_return'], f"{metrics.get('annual_return', 0):.2%}"),
                        (texts['sharpe'], f"{metrics.get('sharpe_ratio', 0):.2f}"),
                        (texts['linearity'], f"{metrics.get('linearity', 0):.4f}"),
                        (texts['max_dd'], f"{metrics.get('max_drawdown', 0):.2%}"),
                        (texts['turnover'], f"{avg_turnover:.2%}"),
                    ])
            
            _render_summary_table(ax_summary, summary_data)
        
        _plot_drawdown(ax_dd, drawdown)
        _style_axis(ax_dd, texts['drawdown_ylabel'])
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        
        if not turnover_df.empty:
            ax_to.plot(
                turnover_df.index, turnover_df['turnover'], 
                color=_PLOT_COLORS['turnover_line'], 
                linewidth=_PLOT_STYLES['thin_linewidth'], 
                alpha=0.9
            )
            _style_axis(ax_to, texts['turnover_ylabel'], is_bottom=True, xlabel=texts['date_xlabel'])
            ax_to.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        else:
            ax_to.text(
                0.5, 0.5, texts['no_turnover'], 
                transform=ax_to.transAxes, 
                ha='center', va='center', 
                fontsize=_PLOT_STYLES['ylabel_size'], 
                color=_PLOT_COLORS['text_muted']
            )
            _style_axis(ax_to, '', is_bottom=True, xlabel=texts['date_xlabel'])
        
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax_dd.get_xticklabels(), visible=False)
        
        fig.align_ylabels([ax, ax_dd, ax_to])
        
        plt.show()


class FactorPlotter:
    """Time series visualization for Factor data."""
    
    def __init__(self, factor: 'Factor'):
        self.factor = factor
    
    def plot(self, symbol: Optional[str] = None, figsize: tuple = (12, 5), 
             title: Optional[str] = None) -> None:
        if symbol is None:
            self._plot_all_symbols(figsize, title)
        else:
            self._plot_single_symbol(symbol, figsize, title)
    
    def _plot_single_symbol(self, symbol: str, figsize: tuple, title: Optional[str]) -> None:
        data = self.factor.data[self.factor.data['symbol'] == symbol].copy()
        if data.empty:
            warnings.warn(f"No data found for symbol: {symbol}")
            return
        
        data = data.sort_values('timestamp')
        
        _apply_plot_style()
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(_PLOT_COLORS['background_subtle'])
        
        nan_dates = data[data['factor'].isna()]['timestamp']
        for nan_date in nan_dates:
            ax.axvline(
                nan_date, 
                color='#dc2626', 
                linewidth=0.9, 
                linestyle=':', 
                alpha=1.0,
                zorder=10
            )
        
        line_color = _PLOT_COLORS['factor_palette'][0]
        ax.plot(
            data['timestamp'], data['factor'], 
            color=line_color, 
            linewidth=_PLOT_STYLES['thin_linewidth'], 
            alpha=_PLOT_STYLES['factor_line_alpha']
        )
        ax.fill_between(
            data['timestamp'], data['factor'], 
            alpha=_PLOT_STYLES['factor_fill_alpha'], 
            color=line_color
        )
        
        plot_title = title or f'{self.factor.name} ({symbol})'
        ax.set_title(
            plot_title, 
            fontsize=_PLOT_STYLES['factor_title_size'], 
            fontweight='400', 
            color=_PLOT_COLORS['text_light'], 
            pad=_PLOT_STYLES['factor_title_pad']
        )
        ax.set_xlabel('Date', fontsize=_PLOT_STYLES['factor_label_size'], color=_PLOT_COLORS['text_muted'])
        ax.set_ylabel('Factor Value', fontsize=_PLOT_STYLES['factor_label_size'], color=_PLOT_COLORS['text_muted'])
        ax.grid(
            True, 
            alpha=_PLOT_STYLES['factor_grid_alpha'], 
            color=_PLOT_COLORS['grid_subtle'], 
            linestyle='-', 
            linewidth=_PLOT_STYLES['factor_grid_width']
        )
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(_PLOT_STYLES['spine_color'])
            ax.spines[spine].set_linewidth(_PLOT_STYLES['spine_width'])
        
        ax.tick_params(
            axis='both', which='major', 
            labelsize=_PLOT_STYLES['factor_tick_size'], 
            colors=_PLOT_COLORS['text_muted'], 
            width=0.5, length=3
        )
        
        plt.tight_layout()
        plt.show()
    
    def _plot_all_symbols(self, figsize: tuple, title: Optional[str]) -> None:
        symbols = sorted(self.factor.data['symbol'].unique())
        n_symbols = len(symbols)
        
        if n_symbols == 0:
            warnings.warn("No data to plot")
            return
        
        n_cols = min(3, n_symbols)
        n_rows = (n_symbols + n_cols - 1) // n_cols
        
        _apply_plot_style()
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(top=0.88, bottom=0.10, left=0.07, right=0.97, hspace=0.50, wspace=0.30)
        
        gs = fig.add_gridspec(n_rows, n_cols)
        axes = []
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_cols + j
                if idx < n_symbols:
                    axes.append(fig.add_subplot(gs[i, j]))
        
        plot_title = title or self.factor.name
        fig.suptitle(
            plot_title,
            fontsize=_PLOT_STYLES['title_size'],
            fontweight='500',
            color=_PLOT_COLORS['text_dark'],
            y=0.97
        )
        
        palette = _PLOT_COLORS['factor_palette']
        
        for idx, symbol in enumerate(symbols):
            ax = axes[idx]
            data = self.factor.data[self.factor.data['symbol'] == symbol].copy()
            data = data.sort_values('timestamp')
            
            nan_dates = data[data['factor'].isna()]['timestamp']
            for nan_date in nan_dates:
                ax.axvline(
                    nan_date, 
                    color='#dc2626', 
                    linewidth=0.9, 
                    linestyle=':', 
                    alpha=1.0,
                    zorder=10
                )
            
            color = palette[idx % len(palette)]
            ax.plot(
                data['timestamp'], data['factor'], 
                color=color, 
                linewidth=_PLOT_STYLES['thin_linewidth'], 
                alpha=_PLOT_STYLES['factor_line_alpha']
            )
            ax.fill_between(
                data['timestamp'], data['factor'], 
                alpha=_PLOT_STYLES['factor_fill_alpha'], 
                color=color
            )
            
            ax.set_title(
                symbol, 
                fontsize=_PLOT_STYLES['factor_subgrid_title_size'], 
                fontweight='500', 
                color=_PLOT_COLORS['text_light'],
                pad=8
            )
            ax.set_xlabel('Date', fontsize=_PLOT_STYLES['factor_subgrid_label_size'], color=_PLOT_COLORS['text_muted'])
            ax.set_ylabel('Value', fontsize=_PLOT_STYLES['factor_subgrid_label_size'], color=_PLOT_COLORS['text_muted'])
            ax.grid(
                True, 
                alpha=_PLOT_STYLES['factor_grid_alpha_subgrid'], 
                color=_PLOT_COLORS['grid_subtle'], 
                linestyle='-', 
                linewidth=_PLOT_STYLES['factor_grid_width']
            )
            
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color(_PLOT_STYLES['spine_color'])
                ax.spines[spine].set_linewidth(_PLOT_STYLES['spine_width'])
            
            ax.tick_params(
                axis='both', which='major', 
                labelsize=_PLOT_STYLES['factor_subgrid_tick_size'], 
                colors=_PLOT_COLORS['text_muted'], 
                width=0.5, length=3
            )
            ax.set_facecolor(_PLOT_COLORS['background_subtle'])
            
            dates = data['timestamp'].values
            n_dates = len(dates)
            if n_dates > 2:
                tick_indices = [0, n_dates // 2, n_dates - 1]
                ax.set_xticks([dates[i] for i in tick_indices])
        
        plt.show()


class FactorAnalysisPlotter:
    """Professional factor analysis visualization for quant research.
    
    Provides 4 core charts optimized for small asset universes:
    1. Rolling IC time series (7D, 30D smoothed) with inset histogram
    2. Cumulative IC - factor alpha trajectory
    3. IC Decay Curve - optimal holding period analysis
    4. IC by Symbol Heatmap - per-asset contribution analysis
    """
    
    def __init__(self, analyzer: 'FactorAnalyzer'):
        self.analyzer = analyzer
    
    def plot(self, factor_idx: int = 0, horizon_idx: int = 0, 
             figsize: tuple = (14, 10), rolling_windows: list = None) -> None:
        """Generate 4-panel factor analysis chart.
        
        Parameters
        ----------
        factor_idx : int
            Index of factor to plot (default first factor)
        horizon_idx : int  
            Index of horizon to use (default first horizon)
        figsize : tuple
            Figure size
        rolling_windows : list
            Windows for rolling IC (default [7, 30])
        """
        if factor_idx >= len(self.analyzer.factors):
            warnings.warn(f"factor_idx={factor_idx} out of range")
            return
        
        if rolling_windows is None:
            rolling_windows = [7, 30]
        
        factor = self.analyzer.factors[factor_idx]
        horizon = self.analyzer.horizons[horizon_idx]
        
        _apply_plot_style()
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.96, 
                           hspace=0.35, wspace=0.25)
        
        gs = fig.add_gridspec(2, 2)
        
        ax_ic_ts = fig.add_subplot(gs[0, 0])
        ax_cum_ic = fig.add_subplot(gs[0, 1])
        ax_ic_decay = fig.add_subplot(gs[1, 0])
        ax_ic_heatmap = fig.add_subplot(gs[1, 1])
        
        fig.suptitle(
            f'{factor.name}  |  {horizon}D Horizon',
            fontsize=_PLOT_STYLES['title_size'],
            fontweight='500',
            color=_PLOT_COLORS['text_dark'],
            y=0.97
        )
        
        ic_results = self.analyzer.ic()
        ic_data = ic_results.get(factor.name, {}).get(horizon, {})
        ic_series = ic_data.get('ic_series', pd.Series(dtype=float))
        
        self._plot_ic_timeseries_with_histogram(ax_ic_ts, ic_series, rolling_windows, ic_data)
        self._plot_cumulative_ic(ax_cum_ic, ic_series, ic_data)
        self._plot_ic_decay(ax_ic_decay, factor)
        self._plot_ic_by_symbol(ax_ic_heatmap, factor, horizon)
        
        plt.show()
    
    def _plot_ic_timeseries_with_histogram(self, ax, ic_series: pd.Series, 
                                           rolling_windows: list, ic_data: dict) -> None:
        """Panel 1: Rolling IC time series with inset histogram."""
        if ic_series.empty:
            ax.text(0.5, 0.5, 'No IC Data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color=_PLOT_COLORS['text_muted'])
            ax.set_title('Rolling IC', fontsize=_PLOT_STYLES['factor_title_size'],
                        color=_PLOT_COLORS['text_light'])
            return
        
        colors = ['#94a3b8', '#3b82f6', '#1d4ed8']
        labels = []
        
        for i, window in enumerate(rolling_windows):
            rolling_ic = ic_series.rolling(window=window, min_periods=1).mean()
            color = colors[min(i + 1, len(colors) - 1)]
            alpha = 0.6 + 0.15 * i
            linewidth = 1.0 + 0.3 * i
            
            ax.plot(rolling_ic.index, rolling_ic, 
                   color=color,
                   linewidth=linewidth,
                   alpha=alpha,
                   label=f'{window}D')
            labels.append(f'{window}D')
        
        ax.axhline(0, color=_PLOT_COLORS['zero_line'], linewidth=0.8, linestyle='-', alpha=0.6)
        
        ic_mean = ic_data.get('ic_mean', np.nan)
        ir = ic_data.get('ir', np.nan)
        t_stat = ic_data.get('t_stat', np.nan)
        
        if not np.isnan(ic_mean):
            ax.axhline(ic_mean, color=_PLOT_COLORS['benchmark_line'], 
                      linewidth=1.0, linestyle='--', alpha=0.7)
        
        ax.legend(loc='upper left', frameon=False, 
                 fontsize=_PLOT_STYLES['small_label_size'],
                 labelcolor=_PLOT_COLORS['text_muted'])
        
        stats_text = []
        if not np.isnan(ic_mean):
            stats_text.append(f'IC={ic_mean:.4f}')
        if not np.isnan(ir):
            stats_text.append(f'IR={ir:.2f}')
        if not np.isnan(t_stat):
            stats_text.append(f't={t_stat:.1f}')
        
        if stats_text:
            ax.text(0.98, 0.02, '  '.join(stats_text),
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=_PLOT_STYLES['small_label_size'],
                   color=_PLOT_COLORS['text_info'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.9, edgecolor=_PLOT_COLORS['grid']))
        
        inset_ax = ax.inset_axes([0.72, 0.60, 0.25, 0.35])
        inset_ax.hist(ic_series.dropna(), bins=20, alpha=0.7, 
                     color=_PLOT_COLORS['equity_fill'],
                     edgecolor='none')
        inset_ax.axvline(0, color=_PLOT_COLORS['zero_line'], linewidth=0.8, alpha=0.6)
        if not np.isnan(ic_mean):
            inset_ax.axvline(ic_mean, color=_PLOT_COLORS['benchmark_line'], 
                           linewidth=1.0, linestyle='--', alpha=0.8)
        inset_ax.set_facecolor(_PLOT_COLORS['background_subtle'])
        inset_ax.tick_params(axis='both', labelsize=7, colors=_PLOT_COLORS['text_muted'])
        for spine in inset_ax.spines.values():
            spine.set_visible(False)
        
        ax.set_title('Rolling IC', 
                    fontsize=_PLOT_STYLES['factor_title_size'],
                    color=_PLOT_COLORS['text_light'], pad=10)
        ax.set_ylabel('IC', fontsize=_PLOT_STYLES['factor_label_size'],
                     color=_PLOT_COLORS['text_muted'])
        
        self._style_subplot(ax)
    
    def _plot_cumulative_ic(self, ax, ic_series: pd.Series, ic_data: dict) -> None:
        """Panel 2: Cumulative IC curve."""
        if ic_series.empty:
            ax.text(0.5, 0.5, 'No IC Data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color=_PLOT_COLORS['text_muted'])
            ax.set_title('Cumulative IC', fontsize=_PLOT_STYLES['factor_title_size'],
                        color=_PLOT_COLORS['text_light'])
            return
        
        cum_ic = ic_series.cumsum()
        
        ax.fill_between(
            cum_ic.index, 0, cum_ic,
            where=(cum_ic >= 0),
            alpha=0.2, color=_PLOT_COLORS['equity_fill']
        )
        ax.fill_between(
            cum_ic.index, 0, cum_ic,
            where=(cum_ic < 0),
            alpha=0.2, color=_PLOT_COLORS['drawdown_fill']
        )
        ax.plot(cum_ic.index, cum_ic,
               color=_PLOT_COLORS['equity_line'],
               linewidth=_PLOT_STYLES['linewidth'],
               alpha=0.9)
        
        ax.axhline(0, color=_PLOT_COLORS['zero_line'], linewidth=0.8, alpha=0.6)
        
        rolling_max = cum_ic.cummax()
        drawdown = cum_ic - rolling_max
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        if max_dd < 0:
            dd_start_idx = rolling_max[:max_dd_idx].idxmax()
            ax.axvspan(dd_start_idx, max_dd_idx, alpha=0.1, color=_PLOT_COLORS['drawdown_fill'])
        
        positive_ratio = (ic_series > 0).mean()
        final_cum_ic = cum_ic.iloc[-1] if len(cum_ic) > 0 else 0
        
        ax.text(0.02, 0.98, f'Pos%: {positive_ratio:.1%}\nMax DD: {max_dd:.2f}',
               transform=ax.transAxes, ha='left', va='top',
               fontsize=_PLOT_STYLES['small_label_size'],
               color=_PLOT_COLORS['text_info'],
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.9, edgecolor=_PLOT_COLORS['grid']))
        
        ax.set_title(f'Cumulative IC ({final_cum_ic:.2f})', 
                    fontsize=_PLOT_STYLES['factor_title_size'],
                    color=_PLOT_COLORS['text_light'], pad=10)
        ax.set_ylabel('Cumulative IC', fontsize=_PLOT_STYLES['factor_label_size'],
                     color=_PLOT_COLORS['text_muted'])
        
        self._style_subplot(ax)
    
    def _plot_ic_decay(self, ax, factor: 'Factor') -> None:
        """Panel 3: IC Decay Curve across horizons."""
        decay_horizons = [1, 2, 3, 5, 7, 10, 14, 21, 30]
        
        price_pivot = self.analyzer.price.data.pivot(
            index='timestamp', columns='symbol', values='factor'
        )
        factor_pivot = factor.data.pivot(
            index='timestamp', columns='symbol', values='factor'
        )
        
        ic_by_horizon = []
        
        for h in decay_horizons:
            fwd_ret = price_pivot.shift(-h) / price_pivot - 1
            aligned_factor, aligned_ret = factor_pivot.align(fwd_ret, join='inner')
            
            ic_series = self._compute_ic(aligned_factor, aligned_ret)
            
            if len(ic_series) > 0:
                ic_mean = ic_series.mean()
                ic_by_horizon.append({'horizon': h, 'ic': ic_mean})
            else:
                ic_by_horizon.append({'horizon': h, 'ic': np.nan})
        
        df = pd.DataFrame(ic_by_horizon)
        
        if df['ic'].isna().all():
            ax.text(0.5, 0.5, 'No IC Data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color=_PLOT_COLORS['text_muted'])
            ax.set_title('IC Decay', fontsize=_PLOT_STYLES['factor_title_size'],
                        color=_PLOT_COLORS['text_light'])
            return
        
        colors = [_PLOT_COLORS['equity_fill'] if v >= 0 else _PLOT_COLORS['drawdown_fill'] 
                 for v in df['ic'].fillna(0)]
        edge_colors = [_PLOT_COLORS['equity_line'] if v >= 0 else _PLOT_COLORS['drawdown_line'] 
                      for v in df['ic'].fillna(0)]
        
        ax.bar(range(len(df)), df['ic'], color=colors, alpha=0.8,
              edgecolor=edge_colors, linewidth=1.0)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([str(h) for h in df['horizon']])
        
        ax.axhline(0, color=_PLOT_COLORS['zero_line'], linewidth=0.8, alpha=0.6)
        
        valid_ics = df.dropna()
        if len(valid_ics) > 0:
            max_ic = valid_ics['ic'].max()
            max_idx = valid_ics['ic'].idxmax()
            optimal_horizon = valid_ics.loc[max_idx, 'horizon']
            
            half_life = None
            if max_ic > 0:
                half_target = max_ic / 2
                for _, row in valid_ics.iterrows():
                    if row['horizon'] > optimal_horizon and row['ic'] <= half_target:
                        half_life = row['horizon']
                        break
            
            title_suffix = f'Peak: {optimal_horizon}D'
            if half_life:
                title_suffix += f', T½: {half_life}D'
        else:
            title_suffix = ''
        
        ax.set_title(f'IC Decay ({title_suffix})', 
                    fontsize=_PLOT_STYLES['factor_title_size'],
                    color=_PLOT_COLORS['text_light'], pad=10)
        ax.set_xlabel('Horizon (Days)', fontsize=_PLOT_STYLES['factor_label_size'],
                     color=_PLOT_COLORS['text_muted'])
        ax.set_ylabel('IC', fontsize=_PLOT_STYLES['factor_label_size'],
                     color=_PLOT_COLORS['text_muted'])
        
        self._style_subplot(ax)
    
    def _plot_ic_by_symbol(self, ax, factor: 'Factor', horizon: int) -> None:
        """Panel 4: IC contribution by symbol over time."""
        price_pivot = self.analyzer.price.data.pivot(
            index='timestamp', columns='symbol', values='factor'
        )
        factor_pivot = factor.data.pivot(
            index='timestamp', columns='symbol', values='factor'
        )
        
        fwd_ret = price_pivot.shift(-horizon) / price_pivot - 1
        aligned_factor, aligned_ret = factor_pivot.align(fwd_ret, join='inner')
        
        symbols = sorted(aligned_factor.columns)
        
        monthly_ic = {}
        
        for symbol in symbols:
            f_col = aligned_factor[symbol]
            r_col = aligned_ret[symbol]
            
            valid_mask = f_col.notna() & r_col.notna()
            f_valid = f_col[valid_mask]
            r_valid = r_col[valid_mask]
            
            if len(f_valid) < 10:
                continue
            
            symbol_df = pd.DataFrame({'factor': f_valid, 'return': r_valid})
            symbol_df['month'] = symbol_df.index.to_period('M')
            
            monthly_contrib = {}
            for month, group in symbol_df.groupby('month'):
                if len(group) >= 3:
                    f_rank = group['factor'].rank()
                    r_rank = group['return'].rank()
                    corr = f_rank.corr(r_rank)
                    monthly_contrib[month] = corr
            
            if monthly_contrib:
                monthly_ic[symbol] = monthly_contrib
        
        if not monthly_ic:
            ax.text(0.5, 0.5, 'Insufficient Data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color=_PLOT_COLORS['text_muted'])
            ax.set_title('IC by Symbol', fontsize=_PLOT_STYLES['factor_title_size'],
                        color=_PLOT_COLORS['text_light'])
            return
        
        ic_df = pd.DataFrame(monthly_ic).T
        ic_df = ic_df.reindex(sorted(ic_df.index))
        
        if ic_df.shape[1] > 12:
            ic_df = ic_df.iloc[:, -12:]
        
        from matplotlib.colors import TwoSlopeNorm
        
        vmax = max(abs(ic_df.min().min()), abs(ic_df.max().max()), 0.3)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        im = ax.imshow(ic_df.values, aspect='auto', cmap='RdBu_r', norm=norm)
        
        ax.set_yticks(range(len(ic_df.index)))
        ax.set_yticklabels(ic_df.index)
        
        n_cols = len(ic_df.columns)
        if n_cols <= 6:
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels([str(c) for c in ic_df.columns], rotation=45, ha='right')
        else:
            step = max(1, n_cols // 6)
            ax.set_xticks(range(0, n_cols, step))
            ax.set_xticklabels([str(ic_df.columns[i]) for i in range(0, n_cols, step)], 
                              rotation=45, ha='right')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=8, colors=_PLOT_COLORS['text_muted'])
        
        ax.set_title('IC by Symbol (Monthly)', 
                    fontsize=_PLOT_STYLES['factor_title_size'],
                    color=_PLOT_COLORS['text_light'], pad=10)
        
        ax.tick_params(axis='both', which='major',
                      labelsize=_PLOT_STYLES['factor_tick_size'],
                      colors=_PLOT_COLORS['text_muted'])
    
    def _compute_ic(self, factor_pivot: pd.DataFrame, ret_pivot: pd.DataFrame) -> pd.Series:
        """Compute IC series using Spearman correlation."""
        f_data = factor_pivot.rank(axis=1, na_option='keep')
        r_data = ret_pivot.rank(axis=1, na_option='keep')
        
        valid_mask = factor_pivot.notna() & ret_pivot.notna()
        valid_count = valid_mask.sum(axis=1)
        
        f_std = f_data.std(axis=1, skipna=True)
        r_std = r_data.std(axis=1, skipna=True)
        std_valid = (f_std > 1e-10) & (r_std > 1e-10) & (valid_count >= 3)
        
        f_demean = f_data.sub(f_data.mean(axis=1, skipna=True), axis=0)
        r_demean = r_data.sub(r_data.mean(axis=1, skipna=True), axis=0)
        
        numer = (f_demean * r_demean).sum(axis=1, skipna=True)
        denom = (f_demean.pow(2).sum(axis=1, skipna=True) * 
                 r_demean.pow(2).sum(axis=1, skipna=True)).pow(0.5)
        
        ic = numer / denom
        ic = ic[std_valid]
        
        return ic.dropna()
    
    def _style_subplot(self, ax) -> None:
        """Apply consistent styling to subplot."""
        ax.set_facecolor(_PLOT_COLORS['white'])
        ax.grid(True, alpha=_PLOT_STYLES['grid_alpha'], 
               color=_PLOT_COLORS['grid'],
               linestyle='-', linewidth=_PLOT_STYLES['grid_width'])
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(_PLOT_STYLES['spine_color'])
            ax.spines[spine].set_linewidth(_PLOT_STYLES['spine_width'])
        
        ax.tick_params(axis='both', which='major',
                      labelsize=_PLOT_STYLES['factor_tick_size'],
                      colors=_PLOT_COLORS['text_muted'],
                      width=0.5, length=3)

