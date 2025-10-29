"""Functional operator API for Factor operations."""

from typing import Union, List, Optional
from .core import Factor 


def group_neutralize(factor: 'Factor', group_data: 'Factor') -> 'Factor':
    """Neutralize factor against specified groups."""
    return factor.group_neutralize(group_data)

def vector_neut(x: 'Factor', y: 'Factor') -> 'Factor':
    """Find vector x* orthogonal to y using vector projection."""
    return x.vector_neut(y)

def regression_neut(y: 'Factor', x: 'Factor') -> 'Factor':
    """Neutralize y against x using OLS regression."""
    return y.regression_neut(x)

def rank(factor: 'Factor') -> 'Factor':
    """Cross-sectional rank."""
    return factor.rank()

def mean(factor: 'Factor') -> 'Factor':
    """Cross-sectional mean."""
    return factor.mean()

def median(factor: 'Factor') -> 'Factor':
    """Cross-sectional median."""
    return factor.median()

def normalize(factor: 'Factor', useStd: bool = False, limit: float = 0.0) -> 'Factor':
    """Cross-sectional normalization with optional standardization."""
    return factor.normalize(useStd, limit)

def quantile(factor: 'Factor', driver: str = "gaussian", sigma: float = 1.0) -> 'Factor':
    """Cross-sectional quantile transformation."""
    return factor.quantile(driver, sigma)

def scale(factor: 'Factor', scale: float = 1.0, longscale: float = -1.0, shortscale: float = -1.0) -> 'Factor':
    """Scale to target position size."""
    return factor.scale(scale, longscale, shortscale)

def zscore(factor: 'Factor') -> 'Factor':
    """Cross-sectional Z-score."""
    return factor.zscore()

def spread(factor: 'Factor', pct: float = 0.5) -> 'Factor':
    """Spread transformation (top/bottom pct% long-short)."""
    return factor.spread(pct)

def ts_rank(factor: 'Factor', window: int) -> 'Factor':
    """Rolling time-series rank."""
    return factor.ts_rank(window)

def ts_mean(factor: 'Factor', window: int) -> 'Factor':
    """Rolling mean."""
    return factor.ts_mean(window)

def ts_median(factor: 'Factor', window: int) -> 'Factor':
    """Rolling median."""
    return factor.ts_median(window)

def ts_product(factor: 'Factor', window: int) -> 'Factor':
    """Rolling product."""
    return factor.ts_product(window)

def ts_sum(factor: 'Factor', window: int) -> 'Factor':
    """Rolling sum."""
    return factor.ts_sum(window)

def ts_std_dev(factor: 'Factor', window: int) -> 'Factor':
    """Rolling standard deviation."""
    return factor.ts_std_dev(window)

def where(condition: Factor, x: Union[Factor, float], y: Union[Factor, float]) -> Factor:
    """Select from x where condition is True, else from y."""
    if not isinstance(x, Factor):
        raise TypeError("The `x` argument must be a Factor object for `where` operator.")
    return x.where(condition, other=y)

def ts_corr(factor1: 'Factor', factor2: 'Factor', window: int) -> 'Factor':
    """Rolling correlation."""
    return factor1.ts_corr(factor2, window)

def ts_delay(factor: 'Factor', window: int) -> 'Factor':
    """Lag factor."""
    return factor.ts_delay(window)

def ts_delta(factor: 'Factor', window: int) -> 'Factor':
    """Difference from lagged value."""
    return factor.ts_delta(window)

def ts_arg_max(factor: 'Factor', window: int) -> 'Factor':
    """Relative index of maximum."""
    return factor.ts_arg_max(window)

def ts_arg_min(factor: 'Factor', window: int) -> 'Factor':
    """Relative index of minimum."""
    return factor.ts_arg_min(window)

def ts_min(factor: 'Factor', window: int) -> 'Factor':
    """Rolling minimum."""
    return factor.ts_min(window)

def ts_max(factor: 'Factor', window: int) -> 'Factor':
    """Rolling maximum."""
    return factor.ts_max(window)

def ts_count_nans(factor: 'Factor', window: int) -> 'Factor':
    """Count NaN values in window."""
    return factor.ts_count_nans(window)

def ts_covariance(factor1: 'Factor', factor2: 'Factor', window: int) -> 'Factor':
    """Rolling covariance."""
    return factor1.ts_covariance(factor2, window)

def ts_quantile(factor: 'Factor', window: int, driver: str = "gaussian") -> 'Factor':
    """Rolling quantile transformation."""
    return factor.ts_quantile(window, driver)

def ts_av_diff(factor: 'Factor', window: int) -> 'Factor':
    """Difference from rolling mean."""
    return factor.ts_av_diff(window)

def ts_scale(factor: 'Factor', window: int, constant: float = 0) -> 'Factor':
    """Min-max normalization within window."""
    return factor.ts_scale(window, constant)

def ts_zscore(factor: 'Factor', window: int) -> 'Factor':
    """Rolling Z-score."""
    return factor.ts_zscore(window)

def ts_backfill(factor: 'Factor', window: int, k: int = 1) -> 'Factor':
    """Fill NaN with k-th most recent non-NaN."""
    return factor.ts_backfill(window, k)

def ts_decay_exp_window(factor: 'Factor', window: int, factor_arg: float = 1.0, nan: bool = True) -> 'Factor':
    """Exponentially weighted rolling average."""
    return factor.ts_decay_exp_window(window, factor_arg, nan)

def ts_decay_linear(factor: 'Factor', window: int, dense: bool = False) -> 'Factor':
    """Linearly weighted rolling average."""
    return factor.ts_decay_linear(window, dense)

def ts_regression(y: 'Factor', x: 'Factor', window: int, lag: int = 0, rettype: int = 0) -> 'Factor':
    """Rolling linear regression (rettype: 0=residuals, 1=intercept, 2=slope, 3=fitted, 4=SSE, 5=SST, 6=R², 7=MSE, 8=SE(beta), 9=SE(alpha))."""
    return y.ts_regression(x, window, lag, rettype)

def log(factor: 'Factor', base: Optional[float] = None) -> 'Factor':
    """Logarithm."""
    return factor.log(base)

def ln(factor: 'Factor') -> 'Factor':
    """Natural logarithm."""
    return factor.ln()

def s_log_1p(factor: 'Factor') -> 'Factor':
    """Signed logarithm: sign(x) * log(1 + abs(x))."""
    return factor.s_log_1p()

def sign(factor: 'Factor') -> 'Factor':
    """Sign of values."""
    return factor.sign()

def sqrt(factor: 'Factor') -> 'Factor':
    """Square root."""
    return factor.sqrt()

def maximum(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise maximum."""
    return factor1.maximum(factor2)

def minimum(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise minimum."""
    return factor1.minimum(factor2)

def divide(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Division."""
    return factor1.divide(factor2)

def inverse(factor: 'Factor') -> 'Factor':
    """Reciprocal: 1 / factor."""
    return factor.inverse()

def add(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Addition."""
    return factor1.add(factor2)

def multiply(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Multiplication."""
    return factor1.multiply(factor2)

def subtract(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Subtraction."""
    return factor1.subtract(factor2)

def power(base: 'Factor', exponent: Union['Factor', float]) -> 'Factor':
    """Power operation."""
    return base.power(exponent)

def reverse(factor: 'Factor') -> 'Factor':
    """Negate factor."""
    return factor.reverse()

def signed_power(base: 'Factor', exponent: Union['Factor', float]) -> 'Factor':
    """Power preserving sign of base."""
    return base.signed_power(exponent)