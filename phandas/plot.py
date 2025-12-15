"""Plotting utilities for backtesting results and factor analysis."""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .backtest import Backtester
    from .core import Factor

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
    'psr': 'PSR',
    'sortino': 'Sortino Ratio',
    'calmar': 'Calmar Ratio',
    'linearity': 'Linearity',
    'max_dd': 'Max Drawdown',
    'var_95': 'VaR 95%',
    'cvar': 'CVaR',
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
        
        downside_returns = benchmark_returns[benchmark_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 0
        bmk_sortino = (bmk_annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        rolling_max = benchmark_norm.cummax()
        drawdown = benchmark_norm / rolling_max - 1
        bmk_max_dd = drawdown.min()
        bmk_calmar = bmk_annual_return / abs(bmk_max_dd) if bmk_max_dd < 0 else 0
        
        from scipy.stats import linregress
        t = np.arange(len(benchmark_norm))
        r_value = linregress(t, benchmark_norm.values)[2]
        bmk_linearity = r_value ** 2
        
        bmk_var_95 = benchmark_returns.quantile(0.05)
        bmk_cvar = benchmark_returns[benchmark_returns <= bmk_var_95].mean() if (benchmark_returns <= bmk_var_95).any() else 0
        
        return {
            'bmk_total_return': bmk_total_return,
            'bmk_annual_return': bmk_annual_return,
            'bmk_sharpe': bmk_sharpe,
            'bmk_sortino': bmk_sortino,
            'bmk_calmar': bmk_calmar,
            'bmk_linearity': bmk_linearity,
            'bmk_max_drawdown': bmk_max_dd,
            'bmk_var_95': bmk_var_95,
            'bmk_cvar': bmk_cvar,
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
                        (texts['psr'], f"{metrics.get('psr', 0):.1%}", '-'),
                        (texts['sortino'], f"{metrics.get('sortino_ratio', 0):.2f}", 
                         f"{benchmark_metrics.get('bmk_sortino', 0):.2f}"),
                        (texts['calmar'], f"{metrics.get('calmar_ratio', 0):.2f}", 
                         f"{benchmark_metrics.get('bmk_calmar', 0):.2f}"),
                        (texts['linearity'], f"{metrics.get('linearity', 0):.4f}", 
                         f"{benchmark_metrics.get('bmk_linearity', 0):.4f}"),
                        (texts['max_dd'], f"{metrics.get('max_drawdown', 0):.2%}", 
                         f"{benchmark_metrics.get('bmk_max_drawdown', 0):.2%}"),
                        (texts['var_95'], f"{metrics.get('var_95', 0):.2%}", 
                         f"{benchmark_metrics.get('bmk_var_95', 0):.2%}"),
                        (texts['cvar'], f"{metrics.get('cvar', 0):.2%}", 
                         f"{benchmark_metrics.get('bmk_cvar', 0):.2%}"),
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
                        (texts['psr'], f"{metrics.get('psr', 0):.1%}"),
                        (texts['sortino'], f"{metrics.get('sortino_ratio', 0):.2f}"),
                        (texts['calmar'], f"{metrics.get('calmar_ratio', 0):.2f}"),
                        (texts['linearity'], f"{metrics.get('linearity', 0):.4f}"),
                        (texts['max_dd'], f"{metrics.get('max_drawdown', 0):.2%}"),
                        (texts['var_95'], f"{metrics.get('var_95', 0):.2%}"),
                        (texts['cvar'], f"{metrics.get('cvar', 0):.2%}"),
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
        if n_symbols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten() if n_symbols > 1 else np.array([axes])
        
        palette = _PLOT_COLORS['factor_palette']
        
        for idx, symbol in enumerate(symbols):
            ax = axes[idx]
            data = self.factor.data[self.factor.data['symbol'] == symbol].copy()
            data = data.sort_values('timestamp')
            
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
                color=_PLOT_COLORS['text_light']
            )
            ax.set_xlabel('Date', fontsize=_PLOT_STYLES['factor_subgrid_label_size'], color=_PLOT_COLORS['text_muted'])
            ax.set_ylabel('Factor Value', fontsize=_PLOT_STYLES['factor_subgrid_label_size'], color=_PLOT_COLORS['text_muted'])
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
        
        for idx in range(n_symbols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.show()
