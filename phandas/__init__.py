"""
Phandas: Professional quantitative analysis framework for cryptocurrency markets.

Clean, efficient factor analysis with pandas-like API.
"""

__author__ = "Phantom Management"
__version__ = "0.4.2"

# Core components
from .core import (
    Factor, load_factor
)

from .operators import (
    # Time-series operations
    ts_rank, ts_mean, ts_median, ts_product, ts_sum, ts_std_dev, ts_corr, ts_delay, ts_delta, 
    ts_arg_max, ts_arg_min, ts_av_diff, ts_backfill, ts_decay_exp_window, ts_decay_linear,
    ts_count_nans, ts_covariance, ts_quantile, ts_scale, ts_zscore, ts_min, ts_max, ts_regression,
    
    # Cross-sectional operations
    rank, normalize, quantile, scale, zscore,
    
    # Mathematical operations
    log, ln, s_log_1p, sign, sqrt, maximum, minimum, multiply, power, reverse, 
    signed_power, subtract, divide, inverse, add, where,
    
    # Neutralization operations
    group_neutralize, vector_neutralize, regression_neutralize, vector_neut, regression_neut
)
from .data import fetch_data, check_data_quality
from .backtest import Backtester, backtest
from .utils import save_factor, load_saved_factor, factor_info

__all__ = [
    # Core classes and factories
    'Factor', 'load_factor',
    
    # Data management
    'fetch_data', 'check_data_quality',
    
    # Backtesting
    'Backtester', 'backtest',
    
    # Time-series operations
    'ts_rank', 'ts_mean', 'ts_median', 'ts_product', 'ts_sum', 'ts_std_dev', 'ts_corr', 'ts_delay', 'ts_delta', 
    'ts_arg_max', 'ts_arg_min', 'ts_av_diff', 'ts_backfill', 'ts_decay_exp_window', 'ts_decay_linear',
    'ts_count_nans', 'ts_covariance', 'ts_quantile', 'ts_scale', 'ts_zscore', 'ts_min', 'ts_max', 'ts_regression',
    
    # Cross-sectional operations
    'rank', 'normalize', 'quantile', 'scale', 'zscore',
    
    # Mathematical operations
    'log', 'ln', 's_log_1p', 'sign', 'sqrt', 'maximum', 'minimum', 'multiply', 'power', 'reverse',
    'signed_power', 'subtract', 'divide', 'inverse', 'add', 'where',
    
    # Neutralization operations
    'group_neutralize', 'vector_neutralize', 'regression_neutralize', 'vector_neut', 'regression_neut',
    
    # Utilities
    'save_factor', 'load_saved_factor', 'factor_info'
]