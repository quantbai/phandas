"""
phandas: A quantitative analysis framework for financial markets.

phandas provides high-performance, easy-to-use data structures and
financial analysis tools designed for factor-based trading strategy
research, particularly in cryptocurrency markets.
"""

__author__ = "Phantom Management"
__version__ = "0.5.5"

from .core import (
    Factor
)
from .panel import Panel

from .operators import (
    ts_rank, ts_mean, ts_median, ts_product, ts_sum, ts_std_dev, ts_corr, ts_delay, ts_delta, 
    ts_arg_max, ts_arg_min, ts_av_diff, ts_backfill, ts_decay_exp_window, ts_decay_linear,
    ts_count_nans, ts_covariance, ts_quantile, ts_scale, ts_zscore, ts_min, ts_max, ts_regression,
    
    rank, normalize, quantile, scale, zscore,
    
    log, ln, s_log_1p, sign, sqrt, maximum, minimum, multiply, power, reverse, 
    signed_power, subtract, divide, inverse, add, where,
    
    group_neutralize, vector_neut, regression_neut
)
from .data import fetch_data, check_data_quality
from .backtest import Backtester, backtest
from .layer_backtest import backtest_layer
from .layer import analyze_layers
from .trade_simulator import simulate_trade_replay

from .ic import (
    analyze_ic 
)

__all__ = [
    'Factor', 'load_factor', 'Panel',
    
    'fetch_data', 'check_data_quality',
    
    'Backtester', 'backtest', 'backtest_layer', 'simulate_trade_replay',
    
    'ts_rank', 'ts_mean', 'ts_median', 'ts_product', 'ts_sum', 'ts_std_dev', 'ts_corr', 'ts_delay', 'ts_delta', 
    'ts_arg_max', 'ts_arg_min', 'ts_av_diff', 'ts_backfill', 'ts_decay_exp_window', 'ts_decay_linear',
    'ts_count_nans', 'ts_covariance', 'ts_quantile', 'ts_scale', 'ts_zscore', 'ts_min', 'ts_max', 'ts_regression',
    
    'rank', 'normalize', 'quantile', 'scale', 'zscore',
    
    'log', 'ln', 's_log_1p', 'sign', 'sqrt', 'maximum', 'minimum', 'multiply', 'power', 'reverse',
    'signed_power', 'subtract', 'divide', 'inverse', 'add', 'where',
    
    'group_neutralize', 'vector_neut', 'regression_neut',
    
    'analyze_ic', 'analyze_layers',
]