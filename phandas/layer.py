"""
Layer test (stratified test) for factor evaluation.

Stratifies assets into N quantile groups by factor values each day,
then tracks equal-weighted portfolio returns for each layer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

from .core import Factor

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def _assign_layers(factor_values: pd.Series, n_layers: int) -> pd.Series:
    """Assign layer numbers (1 to n_layers) based on factor value quantiles."""
    return pd.qcut(factor_values.rank(method='first'), n_layers, 
                   labels=False, duplicates='drop') + 1


def daily_layer_returns(
    factor: Factor,
    price: Factor,
    n_layers: int = 5,
    periods: int = 1
) -> pd.DataFrame:
    """
    Calculate daily returns for each factor layer.
    
    Parameters
    ----------
    factor : Factor
        Factor values for stratification
    price : Factor
        Price factor for return calculation
    n_layers : int, default 5
        Number of layers to divide assets into
    periods : int, default 1
        Holding period in days
        
    Returns
    -------
    pd.DataFrame
        Daily returns for each layer, indexed by timestamp, columns are layer numbers
        
    Notes
    -----
    Each day:
    1. Rank all assets by factor values
    2. Divide into n_layers equal groups
    3. Calculate equal-weighted return for each layer over next 'periods' days
    """
    returns = price.returns(periods)
    
    factor_data = factor.data
    returns_data = returns.data
    
    merged = pd.merge(
        factor_data,
        returns_data.rename(columns={'factor': 'return'}),
        on=['timestamp', 'symbol'],
        how='inner'
    )
    
    if merged.empty:
        raise ValueError("No overlapping data between factor and returns")
    
    timestamps = merged['timestamp'].unique()
    layer_returns_list = []
    
    for ts in timestamps:
        ts_data = merged[merged['timestamp'] == ts].dropna()
        
        if len(ts_data) < n_layers:
            continue
        
        ts_data['layer'] = _assign_layers(ts_data['factor'], n_layers)
        
        layer_ret = ts_data.groupby('layer')['return'].mean()
        
        layer_returns_list.append({
            'timestamp': ts,
            **{f'layer_{i}': layer_ret.get(i, np.nan) for i in range(1, n_layers + 1)}
        })
    
    df = pd.DataFrame(layer_returns_list)
    df = df.set_index('timestamp')
    
    return df


def layer_summary(layer_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for each layer.
    
    Parameters
    ----------
    layer_returns : pd.DataFrame
        Daily layer returns from daily_layer_returns()
        
    Returns
    -------
    pd.DataFrame
        Statistics for each layer including cumulative return, annualized return,
        volatility, Sharpe ratio, and max drawdown
    """
    stats_list = []
    
    for col in layer_returns.columns:
        ret = layer_returns[col].dropna()
        
        if len(ret) == 0:
            continue
        
        cum_ret = (1 + ret).prod() - 1
        ann_ret = (1 + cum_ret) ** (365 / len(ret)) - 1
        ann_vol = ret.std() * np.sqrt(365)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        cum_series = (1 + ret).cumprod()
        rolling_max = cum_series.expanding().max()
        drawdown = cum_series / rolling_max - 1
        max_dd = drawdown.min()
        
        layer_num = int(col.split('_')[1])
        
        stats_list.append({
            'layer': layer_num,
            'cum_return': cum_ret,
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'n_periods': len(ret)
        })
    
    return pd.DataFrame(stats_list).set_index('layer').sort_index()


def analyze_layers(
    factor: Factor,
    price: Factor,
    n_layers: int = 5,
    periods: int = 1,
    y_scale: Literal['auto', 'linear', 'log'] = 'auto'
) -> dict:
    """
    Comprehensive layer test with visualization.
    
    Parameters
    ----------
    factor : Factor
        Factor values for stratification
    price : Factor
        Price factor
    n_layers : int, default 5
        Number of layers
    periods : int, default 1
        Holding period
        
    Returns
    -------
    dict
        Results with layer_returns, summary, long_short
        
    Notes
    -----
    Layer test divides assets into quantile groups by factor values each day.
    Layer 1 = lowest factor values (bottom 20% if n_layers=5)
    Layer N = highest factor values (top 20% if n_layers=5)
    
    Long-Short = Layer N - Layer 1 (buying high factor, shorting low factor)
    
    Visualization y-axis:
    - auto: switch to log-scale when spread is large (default)
    - linear: percentage cumulative return
    - log: plot cumulative wealth in log-scale
    """
    layer_rets = daily_layer_returns(factor, price, n_layers, periods)
    summary = layer_summary(layer_rets)
    
    long_short = 0.5 * (layer_rets[f'layer_{n_layers}'] - layer_rets['layer_1'])
    long_short_cum = (1 + long_short).cumprod() - 1
    
    ls_stats = {
        'cum_return': (1 + long_short).prod() - 1,
        'ann_return': ((1 + long_short).prod() ** (365 / len(long_short)) - 1) if len(long_short) > 0 else 0,
        'ann_vol': long_short.std() * np.sqrt(365),
        'sharpe': 0
    }
    ls_stats['sharpe'] = ls_stats['ann_return'] / ls_stats['ann_vol'] if ls_stats['ann_vol'] > 0 else 0
    
    _plot_layer_analysis(layer_rets, summary, long_short_cum, ls_stats, factor.name, n_layers, y_scale)
    
    return {
        'layer_returns': layer_rets,
        'summary': summary,
        'long_short': long_short,
        'long_short_stats': ls_stats
    }


def _plot_layer_analysis(layer_rets: pd.DataFrame, summary: pd.DataFrame,
                         long_short_cum: pd.Series, ls_stats: dict,
                         factor_name: str, n_layers: int,
                         y_scale: Literal['auto', 'linear', 'log']):
    """Generate layer test visualization."""
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])
    
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_layers))
    
    cum_returns = {}
    cum_wealth = {}
    for i in range(1, n_layers + 1):
        col = f'layer_{i}'
        if col in layer_rets.columns:
            ret = layer_rets[col].dropna()
            cr = (1 + ret).cumprod() - 1
            cw = cr + 1.0
            cum_returns[i] = cr
            cum_wealth[i] = cw
    
    use_log = False
    if y_scale == 'log':
        use_log = True
    elif y_scale == 'auto':
        if len(cum_wealth) >= 2:
            w_max = max(series.max() for series in cum_wealth.values())
            w_min = min(series.min() for series in cum_wealth.values())
            if w_min > 0 and (w_max / max(w_min, 1e-8)) >= 20:
                use_log = True
    
    for i in range(1, n_layers + 1):
        if i not in cum_returns:
            continue
        series = cum_wealth[i] if use_log else cum_returns[i]
        ax1.plot(series.index, series.values,
                 linewidth=1.4 if i not in [1, n_layers] else 1.8,
                 alpha=0.9,
                 color=colors[i-1], label=f'Layer {i}')
    
    ax1.axhline(y=0, color='#6b7280', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_title(f'Layer Test: {factor_name}', fontsize=13, fontweight='500', pad=15)
    ax1.set_xlabel('Date', fontsize=10)
    if use_log:
        ax1.set_ylabel('Cumulative Wealth (×, log)', fontsize=10)
        ax1.set_yscale('log')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}×'))
    else:
        ax1.set_ylabel('Cumulative Return', fontsize=10)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax1.legend(loc='best', fontsize=9, ncol=min(n_layers, 6), framealpha=0.95)
    ax1.grid(True, alpha=0.2)
    
    top_layer_ret = summary.loc[n_layers, 'cum_return']
    bottom_layer_ret = summary.loc[1, 'cum_return']
    monotonicity = "Yes" if (summary['cum_return'].diff().dropna() > 0).all() else "No"
    
    stats_text = (
        f"Top Layer: {top_layer_ret:.2%}\n"
        f"Bottom Layer: {bottom_layer_ret:.2%}\n"
        f"Long-Short: {ls_stats['cum_return']:.2%}\n"
        f"LS Sharpe: {ls_stats['sharpe']:.2f}\n"
        f"Monotonic: {monotonicity}"
    )
    ax1.text(0.015, 0.97, stats_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                     edgecolor='#d1d5db', alpha=0.95, linewidth=1))
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(long_short_cum.index, long_short_cum.values, linewidth=1.8, color='#8b5cf6')
    ax2.fill_between(long_short_cum.index, 0, long_short_cum.values,
                     where=(long_short_cum.values > 0), alpha=0.2, color='#8b5cf6')
    ax2.axhline(y=0, color='#6b7280', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_title('Long-Short Return', fontsize=11, fontweight='400')
    ax2.set_xlabel('Date', fontsize=9)
    ax2.set_ylabel('Cumulative', fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(axis='x', rotation=15)
    
    ax3 = fig.add_subplot(gs[1, 1])
    layers = summary.index.values
    cum_rets = summary['cum_return'].values
    colors_bar = [colors[i-1] for i in layers]
    ax3.bar(layers, cum_rets, color=colors_bar, alpha=0.8, edgecolor='white', linewidth=1)
    ax3.axhline(y=0, color='#6b7280', linestyle='--', linewidth=0.8)
    ax3.set_title('Cumulative Return by Layer', fontsize=11, fontweight='400')
    ax3.set_xlabel('Layer', fontsize=9)
    ax3.set_ylabel('Cumulative Return', fontsize=9)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax3.grid(True, alpha=0.2, axis='y')
    
    ax4 = fig.add_subplot(gs[1, 2])
    sharpes = summary['sharpe'].values
    ax4.bar(layers, sharpes, color=colors_bar, alpha=0.8, edgecolor='white', linewidth=1)
    ax4.axhline(y=0, color='#6b7280', linestyle='--', linewidth=0.8)
    ax4.set_title('Sharpe Ratio by Layer', fontsize=11, fontweight='400')
    ax4.set_xlabel('Layer', fontsize=9)
    ax4.set_ylabel('Sharpe Ratio', fontsize=9)
    ax4.grid(True, alpha=0.2, axis='y')
    
    plt.show()
