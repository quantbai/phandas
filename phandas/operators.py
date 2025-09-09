import pandas as pd
from typing import Union, List
from .core import Factor 

def group_neutralize(factor: 'Factor', group_data: 'Factor') -> 'Factor':
    """
    Neutralizes the factor against specified groups (e.g., industry) (functional style).
    """
    return factor.group_neutralize(group_data)

def vector_neutralize(factor: 'Factor', other: 'Factor') -> 'Factor':
    """
    Neutralizes the factor against another factor using vector projection (functional style).
    """
    return factor.vector_neutralize(other)

def regression_neutralize(factor: 'Factor', neut_factors: Union['Factor', List['Factor']]) -> 'Factor':
    """
    Neutralizes the factor against one or more factors using OLS regression (functional style).
    """
    return factor.regression_neutralize(neut_factors)

def rank(factor: 'Factor') -> 'Factor':
    """Cross-sectional rank within each timestamp (functional style)."""
    return factor.rank()

def ts_rank(factor: 'Factor', window: int) -> 'Factor':
    """Rolling time-series rank within window (functional style)."""
    return factor.ts_rank(window)

def ts_mean(factor: 'Factor', window: int) -> 'Factor':
    """Returns average value of x for the past d days (functional style)."""
    return factor.ts_mean(window)

def ts_median(factor: 'Factor', window: int) -> 'Factor':
    """Returns median value of x for the past d days (functional style)."""
    return factor.ts_median(window)

def ts_product(factor: 'Factor', window: int) -> 'Factor':
    """Returns product of x for the past d days (functional style)."""
    return factor.ts_product(window)

def ts_sum(factor: 'Factor', window: int) -> 'Factor':
    """Sum values of x for the past d days (functional style)."""
    return factor.ts_sum(window)

def ts_std_dev(factor: 'Factor', window: int) -> 'Factor':
    """Returns standard deviation of x for the past d days (functional style)."""
    return factor.ts_std_dev(window)

def where(condition: Factor, x: Union[Factor, float], y: Union[Factor, float]) -> Factor:
    """
    Operator wrapper for Factor.where method.
    
    Selects elements from x where condition is True, and from y where it is False.
    
    Parameters
    ----------
    condition : Factor
        A boolean Factor. Where True, yield x, otherwise yield y.
    x : Factor or scalar
        Value to yield where condition is True.
    y : Factor or scalar
        Value to yield where condition is False.
        
    Returns
    -------
    Factor
        A new Factor with elements from x and y, depending on the condition.
    """
    if not isinstance(x, Factor):
        # If x is a scalar, we need a Factor of the same shape as condition
        # to call the .where method.
        # We can create a dummy factor and then apply the logic.
        # A simpler way is to use x.where(condition, y) which is what is desired.
        # Let's check the implementation of where I added.
        # `self.where(cond, other)` keeps `self` where `cond` is true, and takes `other` where false.
        # So `where(cond, x, y)` should be `x.where(cond, y)`.
        raise TypeError("The `x` argument (value if True) must be a Factor object for `where` operator.")
        
    return x.where(condition, other=y)

def ts_corr(factor1: 'Factor', factor2: 'Factor', window: int) -> 'Factor':
    """Rolling correlation between two factors (functional style)."""
    return factor1.ts_corr(factor2, window)

def ts_delay(factor: 'Factor', window: int) -> 'Factor':
    """Returns x value d days ago (functional style)."""
    return factor.ts_delay(window)

def ts_delta(factor: 'Factor', window: int) -> 'Factor':
    """Returns x - ts_delay(x, d) (functional style)."""
    return factor.ts_delta(window)

def ts_arg_max(factor: 'Factor', window: int) -> 'Factor':
    """Returns relative index of max value in time series for past d days (functional style)."""
    return factor.ts_arg_max(window)

def ts_arg_min(factor: 'Factor', window: int) -> 'Factor':
    """Returns relative index of min value in time series for past d days (functional style)."""
    return factor.ts_arg_min(window)

def ts_count_nans(factor: 'Factor', window: int) -> 'Factor':
    """Returns the number of NaN values in x for the past d days (functional style)."""
    return factor.ts_count_nans(window)

def ts_covariance(factor1: 'Factor', factor2: 'Factor', window: int) -> 'Factor':
    """Returns covariance of factor1 and factor2 for the past d days (functional style)."""
    return factor1.ts_covariance(factor2, window)

def ts_quantile(factor: 'Factor', window: int, driver: str = "gaussian") -> 'Factor':
    """Calculates ts_rank and applies an inverse cumulative density function (functional style)."""
    return factor.ts_quantile(window, driver)

def ts_av_diff(factor: 'Factor', window: int) -> 'Factor':
    """Returns x - ts_mean(x, d), ignoring NaNs during mean computation (functional style)."""
    return factor.ts_av_diff(window)

def ts_scale(factor: 'Factor', window: int, constant: float = 0) -> 'Factor':
    """Returns (x - ts_min(x, d)) / (ts_max(x, d) - ts_min(x, d)) + constant (functional style)."""
    return factor.ts_scale(window, constant)

def ts_zscore(factor: 'Factor', window: int) -> 'Factor':
    """Returns Z-score: (x - ts_mean(x,d)) / ts_std_dev(x,d) (functional style)."""
    return factor.ts_zscore(window)

def normalize(factor: 'Factor', useStd: bool = False, limit: float = 0.0) -> 'Factor':
    """Cross-sectional normalization: (x - mean) / std (optional), then limit (optional) (functional style)."""
    return factor.normalize(useStd, limit)

def quantile(factor: 'Factor', driver: str = "gaussian", sigma: float = 1.0) -> 'Factor':
    """Cross-sectional quantile transformation with specified driver and scale (functional style)."""
    return factor.quantile(driver, sigma)

def scale(factor: 'Factor', scale: float = 1.0, longscale: float = -1.0, shortscale: float = -1.0) -> 'Factor':
    """Scales input to booksize, with optional separate scaling for long and short positions (functional style)."""
    return factor.scale(scale, longscale, shortscale)

def zscore(factor: 'Factor') -> 'Factor':
    """Computes the cross-sectional Z-score of the factor (functional style)."""
    return factor.zscore()

def ts_backfill(factor: 'Factor', window: int, k: int = 1) -> 'Factor':
    """Backfills NaN values with the k-th most recent non-NaN value (functional style)."""
    return factor.ts_backfill(window, k)

def ts_decay_exp_window(factor: 'Factor', window: int, factor_arg: float = 1.0, nan: bool = True) -> 'Factor':
    """Returns exponential decay of x with smoothing factor for the past d days (functional style)."""
    return factor.ts_decay_exp_window(window, factor_arg, nan)

def ts_decay_linear(factor: 'Factor', window: int, dense: bool = False) -> 'Factor':
    """Returns the linear decay on x for the past d days (functional style)."""
    return factor.ts_decay_linear(window, dense)

def log(factor: 'Factor') -> 'Factor':
    """Natural logarithm of factor (functional style)."""
    return factor.log()

def s_log_1p(factor: 'Factor') -> 'Factor':
    """Confine factor values to a shorter range using sign(x) * log(1 + abs(x)) (functional style)."""
    return factor.s_log_1p()

def sign(factor: 'Factor') -> 'Factor':
    """Returns the sign of factor values (functional style). NaN input returns NaN."""
    return factor.sign()

def sqrt(factor: 'Factor') -> 'Factor':
    """Square root of factor values (functional style)."""
    return factor.sqrt()

def maximum(*inputs: Union['Factor', float]) -> 'Factor':
    """Maximum value of multiple factors or scalars (functional style)."""
    if not inputs:
        raise ValueError("At least one input is required for maximum")
    
    result_factor = None
    for item in inputs:
        if isinstance(item, Factor): # This Factor will need to be imported
            if result_factor is None:
                result_factor = item.data.copy()
            else:
                merged = pd.merge(result_factor, item.data, on=['timestamp', 'symbol'],
                                  suffixes=('_x', '_y'), how='outer')
                result_factor = merged.copy()
                result_factor['factor'] = merged[['factor_x', 'factor_y']].max(axis=1)
        else: # assume it's a scalar
            if result_factor is None:
                # If the first item is a scalar, we need to convert it to a Factor
                # This is a bit tricky without directly importing Factor here.
                # For now, we'll assume the first input is always a Factor if there are Factors.
                # Or, we can create a dummy Factor for the scalar case if all inputs are scalars.
                # For simplicity, let's assume at least one Factor input or handle pure scalar case.
                # Given the context of "Functional API for Factor objects", a Factor input is expected.
                raise TypeError("First input cannot be a scalar if subsequent inputs are factors for maximum")
            result_factor['factor'] = result_factor['factor'].apply(lambda x: max(x, item))
            
    if result_factor is None:
        raise ValueError("No valid factor found for maximum operation")
        
    # Name the resulting factor
    names = [getattr(i, 'name', str(i)) if isinstance(i, Factor) else str(i) for i in inputs]
    return Factor(result_factor, f"max({','.join(names)})")

def minimum(*inputs: Union['Factor', float]) -> 'Factor':
    """Minimum value of multiple factors or scalars (functional style)."""
    if not inputs:
        raise ValueError("At least one input is required for minimum")
    
    result_factor = None
    for item in inputs:
        if isinstance(item, Factor): # This Factor will need to be imported
            if result_factor is None:
                result_factor = item.data.copy()
            else:
                merged = pd.merge(result_factor, item.data, on=['timestamp', 'symbol'],
                                  suffixes=('_x', '_y'), how='outer')
                result_factor = merged.copy()
                result_factor['factor'] = merged[['factor_x', 'factor_y']].min(axis=1)
        else: # assume it's a scalar
            if result_factor is None:
                raise TypeError("First input cannot be a scalar if subsequent inputs are factors for minimum")
            result_factor['factor'] = result_factor['factor'].apply(lambda x: min(x, item))
            
    if result_factor is None:
        raise ValueError("No valid factor found for minimum operation")
        
    names = [getattr(i, 'name', str(i)) if isinstance(i, Factor) else str(i) for i in inputs]
    return Factor(result_factor, f"min({','.join(names)})")

def divide(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Division of factors (functional style)."""
    return factor1 / factor2

def inverse(factor: 'Factor') -> 'Factor':
    """Inverse of factor (functional style)."""
    return factor.inverse()

def add(factor1: 'Factor', factor2: Union['Factor', float], filter_nan: bool = False) -> 'Factor':
    """Addition with optional NaN filtering (functional style)."""
    return factor1.add(factor2, filter_nan)

def multiply(factor1: 'Factor', factor2: Union['Factor', float], filter_nan: bool = False) -> 'Factor':
    """Multiplication of factors with optional NaN filtering (functional style)."""
    return factor1.multiply(factor2, filter_nan)

def subtract(factor1: 'Factor', factor2: Union['Factor', float], filter_nan: bool = False) -> 'Factor':
    """Subtraction of factors with optional NaN filtering (functional style)."""
    return factor1.subtract(factor2, filter_nan)

def power(base: 'Factor', exponent: Union['Factor', float]) -> 'Factor':
    """Factor to the power of exponent (functional style)."""
    return base.power(exponent)

def reverse(factor: 'Factor') -> 'Factor':
    """Reverse sign of factor (functional style): -factor"""
    return factor.reverse()

def signed_power(base: 'Factor', exponent: Union['Factor', float]) -> 'Factor':
    """x raised to the power of y such that final result preserves sign of x (functional style)."""
    return base.signed_power(exponent)
