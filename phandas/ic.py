"""
Information Coefficient (IC) analysis for factor evaluation.

Core metrics: Daily/period IC (Pearson, Spearman, Kendall), ICIR, rolling IC, autocorr, Newey-West adjustment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional
from scipy import stats

from .core import Factor

def daily_ic(
    factor: Factor,
    price: Factor,
    periods: int = 1,
    method: Literal['spearman', 'pearson', 'kendall'] = 'spearman'
) -> pd.Series:
    """
    Calculate IC between factor and forward returns (factor[t] predicts return[t+periods]).
    
    Parameters
    ----------
    factor : Factor
        Factor values (predictors)
    price : Factor
        Price factor for returns
    periods : int, default 1
        Forward return periods
    method : {'spearman', 'pearson', 'kendall'}, default 'spearman'
        Correlation method
        
    Returns
    -------
    pd.Series
        IC time series indexed by timestamp
    """
    factor_shifted = factor.data.copy()
    factor_shifted['timestamp'] = factor_shifted.groupby('symbol')['timestamp'].shift(-periods)
    factor_shifted = factor_shifted.dropna(subset=['timestamp'])
    
    returns = (price / price.ts_delay(periods)) - 1
    returns_data = returns.data
    
    merged = pd.merge(
        factor_shifted,
        returns_data,
        on=['timestamp', 'symbol'],
        suffixes=('_factor', '_return'),
        how='inner'
    )
    
    merged = merged.rename(columns={'factor_factor': 'factor', 'factor_return': 'return'})
    
    if merged.empty:
        raise ValueError("No overlapping data between factor and returns")
    
    ic_list = []
    
    for ts, ts_data in merged.groupby('timestamp'):
        
        valid_data = ts_data[['factor', 'return']].dropna()
        
        if len(valid_data) < 3:
            ic_list.append({'timestamp': ts, 'ic': np.nan})
            continue
        
        corr: float
        if method == 'spearman':
            result = stats.spearmanr(valid_data['factor'], valid_data['return'])
            corr = result.correlation
        elif method == 'pearson':
            corr, _ = stats.pearsonr(valid_data['factor'], valid_data['return'])
        elif method == 'kendall':
            corr, _ = stats.kendalltau(valid_data['factor'], valid_data['return'])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        ic_list.append({'timestamp': ts, 'ic': corr})
    
    ic_df = pd.DataFrame(ic_list)
    ic_series = ic_df.set_index('timestamp')['ic']
    
    return ic_series


def ic_summary(
    ic: pd.Series,
    newey_west: bool = True,
    lag: Optional[int] = None
) -> dict:
    """
    Calculate IC statistics (mean, std, ICIR, hit rate, etc).
    
    Parameters
    ----------
    ic : pd.Series
        IC time series
    newey_west : bool, default True
        Apply Newey-West autocorrelation adjustment
    lag : int, optional
        Lag for Newey-West (default: auto)
        
    Returns
    -------
    dict
        Statistics: mean_ic, std_ic, icir, icir_nw, hit_rate, n_periods, etc.
    """
    ic_clean = ic.dropna()
    
    if len(ic_clean) == 0:
        return {'mean_ic': np.nan, 'std_ic': np.nan, 'icir': np.nan, 
                'icir_nw': np.nan, 'hit_rate': np.nan, 'n_periods': 0}
    
    mean_ic = ic_clean.mean()
    std_ic = ic_clean.std()
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    
    if newey_west and len(ic_clean) > 10:
        if lag is None:
            lag = min(int(4 * (len(ic_clean) / 100) ** (2/9)), len(ic_clean) // 4)
        
        nw_std = _newey_west_std(ic_clean.values, lag)
        icir_nw = mean_ic / nw_std if nw_std > 0 else np.nan
    else:
        icir_nw = icir
    
    hit_rate = (ic_clean > 0).mean()
    
    ic_pos_mean = ic_clean[ic_clean > 0].mean() if (ic_clean > 0).any() else 0
    ic_neg_mean = ic_clean[ic_clean < 0].mean() if (ic_clean < 0).any() else 0
    
    return {
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'icir': icir,
        'icir_nw': icir_nw,
        'hit_rate': hit_rate,
        'ic_pos_mean': ic_pos_mean,
        'ic_neg_mean': ic_neg_mean,
        'n_periods': len(ic_clean),
        'ic_min': ic_clean.min(),
        'ic_max': ic_clean.max()
    }


def rolling_ic(
    factor: Factor,
    price: Factor,
    window: int = 63,
    periods: int = 1,
    method: Literal['spearman', 'pearson', 'kendall'] = 'spearman',
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate rolling IC over time windows.
    
    Parameters
    ----------
    factor : Factor
        Factor values
    price : Factor
        Price factor
    window : int, default 63
        Rolling window size (trading days)
    periods : int, default 1
        Forward return periods
    method : {'spearman', 'pearson', 'kendall'}, default 'spearman'
        Correlation method
    min_periods : int, optional
        Minimum observations per window
        
    Returns
    -------
    pd.Series
        Rolling IC time series
    """
    ic = daily_ic(factor, price, periods, method)
    
    if min_periods is None:
        min_periods = max(window // 2, 20)
    
    rolling_mean = ic.rolling(window, min_periods=min_periods).mean()
    
    return rolling_mean


def ic_autocorr(ic: pd.Series, max_lag: int = 10) -> pd.Series:
    """
    Calculate IC autocorrelation at different lags.
    
    Parameters
    ----------
    ic : pd.Series
        IC time series
    max_lag : int, default 10
        Maximum lag
        
    Returns
    -------
    pd.Series
        Autocorrelation coefficients by lag
    """
    ic_clean = ic.dropna()
    
    autocorr_list = []
    for lag in range(1, max_lag + 1):
        if len(ic_clean) > lag:
            ac = ic_clean.autocorr(lag)
            autocorr_list.append({'lag': lag, 'autocorr': ac})
        else:
            autocorr_list.append({'lag': lag, 'autocorr': np.nan})
    
    autocorr_df = pd.DataFrame(autocorr_list)
    return autocorr_df.set_index('lag')['autocorr']


def ic_decay(
    factor: Factor,
    price: Factor,
    lags: list[int] = [1, 2, 3, 5, 10, 20],
    method: Literal['spearman', 'pearson', 'kendall'] = 'spearman'
) -> pd.DataFrame:
    """
    Calculate IC at different forward return horizons (decay analysis).
    
    Parameters
    ----------
    factor : Factor
        Factor values
    price : Factor
        Price factor
    lags : list of int, default [1, 2, 3, 5, 10, 20]
        Forward return periods
    method : {'spearman', 'pearson', 'kendall'}, default 'spearman'
        Correlation method
        
    Returns
    -------
    pd.DataFrame
        IC statistics for each lag (mean_ic, icir, icir_nw, hit_rate)
    """
    results = []
    
    for lag in lags:
        ic = daily_ic(factor, price, periods=lag, method=method)
        summary = ic_summary(ic, newey_west=True)
        
        results.append({
            'lag': lag,
            'mean_ic': summary['mean_ic'],
            'icir': summary['icir'],
            'icir_nw': summary['icir_nw'],
            'hit_rate': summary['hit_rate']
        })
    
    return pd.DataFrame(results).set_index('lag')


def _newey_west_std(data: np.ndarray, lag: int) -> float:
    """Calculate Newey-West adjusted standard error accounting for autocorrelation."""
    n = len(data)
    mean = np.mean(data)
    
    var = np.sum((data - mean) ** 2) / n
    
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        autocov = np.sum((data[k:] - mean) * (data[:-k] - mean)) / n
        var += 2 * weight * autocov
    
    return np.sqrt(var)


def describe_ic(ic: pd.Series) -> None:
    """Print IC summary."""
    stats_dict = ic_summary(ic, newey_west=True)
    print(f"Mean IC: {stats_dict['mean_ic']:.4f} | ICIR: {stats_dict['icir_nw']:.3f} | Hit Rate: {stats_dict['hit_rate']:.1%} | n={stats_dict['n_periods']}")


def analyze_ic(
    factor: Factor,
    price: Factor,
    periods: int = 1,
    method: Literal['spearman', 'pearson', 'kendall'] = 'spearman',
    rolling_window: int = 63,
    decay_lags: list[int] = None
) -> dict:
    """
    Comprehensive IC analysis with visualization.
    
    Parameters
    ----------
    factor : Factor
        Factor values
    price : Factor
        Price factor
    periods : int, default 1
        Forward return periods
    method : {'spearman', 'pearson', 'kendall'}, default 'spearman'
        Correlation method
    rolling_window : int, default 63
        Rolling window size
    decay_lags : list of int, optional
        Forward periods for decay analysis
        
    Returns
    -------
    dict
        Results with ic, summary, rolling_ic, decay
    """
    if decay_lags is None:
        decay_lags = [1, 2, 3, 5, 10, 20]
    
    ic = daily_ic(factor, price, periods, method)
    summary = ic_summary(ic, newey_west=True)
    roll_ic = rolling_ic(factor, price, rolling_window, periods, method)
    decay_df = ic_decay(factor, price, decay_lags, method)
    
    _plot_ic_analysis(ic, summary, roll_ic, decay_df, rolling_window, factor.name)
    
    return {
        'ic': ic,
        'summary': summary,
        'rolling_ic': roll_ic,
        'decay': decay_df
    }


def _plot_ic_analysis(ic: pd.Series, summary: dict, roll_ic: pd.Series, 
                      decay_df: pd.DataFrame, rolling_window: int, factor_name: str):
    """Generate IC analysis visualization."""
    ic_clean = ic.dropna()
    if len(ic_clean) == 0:
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])
    
    ma20 = ic_clean.rolling(20, min_periods=10).mean()
    ma63 = ic_clean.rolling(63, min_periods=30).mean()
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ic_clean.index, ic_clean.values, linewidth=0.8, alpha=0.4, 
             color='#94a3b8', label='Daily IC')
    ax1.plot(ma20.index, ma20.values, linewidth=1.5, alpha=0.9, 
             color='#3b82f6', label='MA20')
    ax1.plot(ma63.index, ma63.values, linewidth=1.8, alpha=0.95, 
             color='#dc2626', label='MA63')
    ax1.axhline(y=0, color='#6b7280', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.fill_between(ic_clean.index, 0, ic_clean.values, 
                     where=(ic_clean.values > 0), alpha=0.08, color='#3b82f6')
    ax1.fill_between(ic_clean.index, 0, ic_clean.values, 
                     where=(ic_clean.values < 0), alpha=0.08, color='#ef4444')
    ax1.set_title(f'IC Analysis: {factor_name}', fontsize=13, fontweight='500', pad=15)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('IC', fontsize=10)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax1.grid(True, alpha=0.2)
    
    stats_text = (
        f"Mean IC: {summary['mean_ic']:.4f}\n"
        f"ICIR (NW): {summary['icir_nw']:.3f}\n"
        f"Hit Rate: {summary['hit_rate']:.1%}\n"
        f"Periods: {summary['n_periods']}"
    )
    ax1.text(0.015, 0.97, stats_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                     edgecolor='#d1d5db', alpha=0.95, linewidth=1))
    
    ax2 = fig.add_subplot(gs[1, 0])
    values = ic_clean.values
    ax2.hist(values, bins=30, density=True, color='#c7d2fe', alpha=0.9,
             edgecolor='white', linewidth=0.4)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(values[~np.isnan(values)])
        xs = np.linspace(np.nanmin(values), np.nanmax(values), 200)
        ax2.plot(xs, kde(xs), color='#3b82f6', linewidth=1.6, label='KDE')
    except Exception:
        pass
    ax2.axvline(x=0, color='#6b7280', linestyle='--', linewidth=0.8)
    ax2.axvline(x=ic_clean.mean(), color='#dc2626', linestyle='-', linewidth=1.2)
    ax2.set_title('Distribution', fontsize=11, fontweight='400')
    ax2.set_xlabel('IC', fontsize=9)
    ax2.set_ylabel('Density', fontsize=9)
    ax2.grid(True, alpha=0.2, axis='y')
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(roll_ic.index, roll_ic.values, linewidth=1.5, color='#8b5cf6')
    ax3.axhline(y=0, color='#6b7280', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_title(f'Rolling {rolling_window}D', fontsize=11, fontweight='400')
    ax3.set_xlabel('Date', fontsize=9)
    ax3.set_ylabel('IC', fontsize=9)
    ax3.grid(True, alpha=0.2)
    ax3.tick_params(axis='x', rotation=15)
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(decay_df.index, decay_df['mean_ic'], marker='o', linewidth=2, 
            markersize=6, color='#2563eb')
    ax4.axhline(y=0, color='#6b7280', linestyle='--', linewidth=0.8, alpha=0.5)
    
    best_lag = decay_df['icir_nw'].idxmax()
    ax4.text(0.05, 0.95, f'Best: {best_lag}D', transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef08a', 
                     edgecolor='#facc15', alpha=0.9, linewidth=1))
    
    ax4.set_title('IC Decay', fontsize=11, fontweight='400')
    ax4.set_xlabel('Forward Period (Days)', fontsize=9)
    ax4.set_ylabel('Mean IC', fontsize=9)
    ax4.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()
