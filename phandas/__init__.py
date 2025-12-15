"""Phantom Data Analysis"""

__author__ = "Phantom Management"
__version__ = "0.18.0"

from .core import Factor
from .panel import Panel

from .data import fetch_data

from .backtest import backtest, Backtester

from .analysis import analyze, FactorAnalyzer

from .trader import rebalance, Rebalancer, OKXTrader

from .operators import (
    vector_neut, regression_neut,
    
    group, group_neutralize, group_mean, group_median,
    group_rank, group_scale, group_zscore, group_normalize,
    
    rank, mean, median, normalize, quantile, scale, zscore, spread, signal,
    
    ts_rank, ts_mean, ts_median, ts_product, ts_sum, ts_std_dev, ts_corr, ts_delay, ts_delta, 
    ts_arg_max, ts_arg_min, ts_min, ts_max, ts_count_nans, ts_av_diff,
    ts_covariance, ts_quantile, ts_scale, ts_zscore, ts_backfill,
    ts_decay_exp_window, ts_decay_linear, ts_step, ts_regression,
    ts_kurtosis, ts_skewness,
    ts_cv, ts_jumpiness, ts_trend_strength, ts_vr, ts_autocorr, ts_reversal_count,
    
    log, ln, s_log_1p, sign, sqrt, inverse, maximum, minimum, power, signed_power,
    
    add, multiply, subtract, divide, reverse, where,
    
    show, to_csv, to_df,
)

__all__ = [
    'Factor', 'Panel',
    
    'fetch_data',
    
    'backtest', 'Backtester',

    'analyze', 'FactorAnalyzer',

    'rebalance', 'Rebalancer', 'OKXTrader',

    'vector_neut', 'regression_neut',
    
    'group', 'group_neutralize', 'group_mean', 'group_median',
    'group_rank', 'group_scale', 'group_zscore', 'group_normalize',
    
    'rank', 'mean', 'median', 'normalize', 'quantile', 'scale', 'zscore', 'spread', 'signal',
    
    'ts_rank', 'ts_mean', 'ts_median', 'ts_product', 'ts_sum', 'ts_std_dev', 'ts_corr', 'ts_delay', 'ts_delta', 
    'ts_arg_max', 'ts_arg_min', 'ts_min', 'ts_max', 'ts_count_nans', 'ts_av_diff',
    'ts_covariance', 'ts_quantile', 'ts_scale', 'ts_zscore', 'ts_backfill',
    'ts_decay_exp_window', 'ts_decay_linear', 'ts_step', 'ts_regression',
    'ts_kurtosis', 'ts_skewness',
    'ts_cv', 'ts_jumpiness', 'ts_trend_strength', 'ts_vr', 'ts_autocorr', 'ts_reversal_count',
    
    'log', 'ln', 's_log_1p', 'sign', 'sqrt', 'inverse', 'maximum', 'minimum', 'power', 'signed_power',
    
    'add', 'multiply', 'subtract', 'divide', 'reverse', 'where',
    
    'show', 'to_csv', 'to_df',
]