"""Functional operator API for Factor operations."""

from typing import Union, List, Optional
from .core import Factor 


def vector_neut(x: 'Factor', y: 'Factor') -> 'Factor':
    """Remove y projection from x (vector orthogonalization)."""
    return x.vector_neut(y)

def regression_neut(y: 'Factor', x: 'Factor') -> 'Factor':
    """Orthogonalize y relative to x via OLS residuals."""
    return y.regression_neut(x)

def rank(factor: 'Factor') -> 'Factor':
    """Cross-sectional percentile rank (0-1)."""
    return factor.rank()

def mean(factor: 'Factor') -> 'Factor':
    """Cross-sectional mean per timestamp."""
    return factor.mean()

def median(factor: 'Factor') -> 'Factor':
    """Cross-sectional median per timestamp."""
    return factor.median()

def normalize(factor: 'Factor', useStd: bool = False, limit: float = 0.0) -> 'Factor':
    """Cross-sectional demean; optional std normalization and clipping."""
    return factor.normalize(useStd, limit)

def quantile(factor: 'Factor', driver: str = "gaussian", sigma: float = 1.0) -> 'Factor':
    """Cross-sectional quantile transform (normal/uniform/Cauchy)."""
    return factor.quantile(driver, sigma)

def scale(factor: 'Factor', scale: float = 1.0, longscale: float = -1.0, shortscale: float = -1.0) -> 'Factor':
    """Scale to sum(|factor|)=scale; support separate long/short sizing."""
    return factor.scale(scale, longscale, shortscale)

def zscore(factor: 'Factor') -> 'Factor':
    """Cross-sectional standardization (mean=0, std=1)."""
    return factor.zscore()

def spread(factor: 'Factor', pct: float = 0.5) -> 'Factor':
    """Binary long-short signal: top pct% → +0.5, bottom pct% → -0.5."""
    return factor.spread(pct)

def signal(factor: 'Factor') -> 'Factor':
    """Dollar-neutral signal: demean and scale so long sum=0.5, short=-0.5."""
    return factor.signal()

def ts_rank(factor: 'Factor', window: int) -> 'Factor':
    """Rolling percentile rank within window."""
    return factor.ts_rank(window)

def ts_mean(factor: 'Factor', window: int) -> 'Factor':
    """Rolling mean over window (requires full window)."""
    return factor.ts_mean(window)

def ts_median(factor: 'Factor', window: int) -> 'Factor':
    """Rolling median over window (requires full window)."""
    return factor.ts_median(window)

def ts_product(factor: 'Factor', window: int) -> 'Factor':
    """Rolling product over window (requires full window)."""
    return factor.ts_product(window)

def ts_sum(factor: 'Factor', window: int) -> 'Factor':
    """Rolling sum over window (requires full window)."""
    return factor.ts_sum(window)

def ts_std_dev(factor: 'Factor', window: int) -> 'Factor':
    """Rolling standard deviation over window (requires full window)."""
    return factor.ts_std_dev(window)

def where(condition: Factor, x: Union[Factor, float], y: Union[Factor, float]) -> Factor:
    """Select x when condition=True, else y."""
    if not isinstance(x, Factor):
        raise TypeError("The `x` argument must be a Factor object for `where` operator.")
    return x.where(condition, other=y)

def ts_corr(factor1: 'Factor', factor2: 'Factor', window: int) -> 'Factor':
    """Rolling Pearson correlation over window."""
    return factor1.ts_corr(factor2, window)

def ts_delay(factor: 'Factor', window: int) -> 'Factor':
    """Lag factor by window periods."""
    return factor.ts_delay(window)

def ts_delta(factor: 'Factor', window: int) -> 'Factor':
    """Difference between current and lagged value."""
    return factor.ts_delta(window)

def ts_arg_max(factor: 'Factor', window: int) -> 'Factor':
    """Relative index of maximum in window (0=oldest)."""
    return factor.ts_arg_max(window)

def ts_arg_min(factor: 'Factor', window: int) -> 'Factor':
    """Relative index of minimum in window (0=oldest)."""
    return factor.ts_arg_min(window)

def ts_min(factor: 'Factor', window: int) -> 'Factor':
    """Rolling minimum over window (requires full window)."""
    return factor.ts_min(window)

def ts_max(factor: 'Factor', window: int) -> 'Factor':
    """Rolling maximum over window (requires full window)."""
    return factor.ts_max(window)

def ts_count_nans(factor: 'Factor', window: int) -> 'Factor':
    """Count NaN values in rolling window."""
    return factor.ts_count_nans(window)

def ts_covariance(factor1: 'Factor', factor2: 'Factor', window: int) -> 'Factor':
    """Rolling covariance over window."""
    return factor1.ts_covariance(factor2, window)

def ts_quantile(factor: 'Factor', window: int, driver: str = "gaussian") -> 'Factor':
    """Rolling quantile transform (normal/uniform/Cauchy)."""
    return factor.ts_quantile(window, driver)

def ts_kurtosis(factor: 'Factor', window: int) -> 'Factor':
    """Rolling excess kurtosis: E[(x-mean)^4]/std^4 - 3."""
    return factor.ts_kurtosis(window)

def ts_skewness(factor: 'Factor', window: int) -> 'Factor':
    """Rolling sample skewness with Bessel correction."""
    return factor.ts_skewness(window)

def ts_av_diff(factor: 'Factor', window: int) -> 'Factor':
    """Deviation from rolling mean."""
    return factor.ts_av_diff(window)

def ts_scale(factor: 'Factor', window: int, constant: float = 0) -> 'Factor':
    """Rolling min-max scale: (x-min)/(max-min) + constant."""
    return factor.ts_scale(window, constant)

def ts_zscore(factor: 'Factor', window: int) -> 'Factor':
    """Rolling Z-score over window."""
    return factor.ts_zscore(window)

def ts_backfill(factor: 'Factor', window: int, k: int = 1) -> 'Factor':
    """Backfill NaN with k-th most recent non-NaN in window."""
    return factor.ts_backfill(window, k)

def ts_decay_exp_window(factor: 'Factor', window: int, factor_arg: float = 1.0, nan: bool = True) -> 'Factor':
    """Exponentially weighted rolling average (recent heavier)."""
    return factor.ts_decay_exp_window(window, factor_arg, nan)

def ts_decay_linear(factor: 'Factor', window: int, dense: bool = False) -> 'Factor':
    """Linearly weighted rolling average (recent heavier)."""
    return factor.ts_decay_linear(window, dense)

def ts_regression(y: 'Factor', x: 'Factor', window: int, lag: int = 0, rettype: int = 0) -> 'Factor':
    """Rolling OLS regression (rettype: 0=residual, 1=α, 2=β, 3=pred, 4=SSE, 5=SST, 6=R², 7=MSE, 8=SEβ, 9=SEα)."""
    return y.ts_regression(x, window, lag, rettype)

def log(factor: 'Factor', base: Optional[float] = None) -> 'Factor':
    """Logarithm (base=None → natural)."""
    return factor.log(base)

def ln(factor: 'Factor') -> 'Factor':
    """Natural logarithm."""
    return factor.ln()

def s_log_1p(factor: 'Factor') -> 'Factor':
    """Sign-preserving log: sign(x)·ln(1+|x|)."""
    return factor.s_log_1p()

def sign(factor: 'Factor') -> 'Factor':
    """Sign of values (-1/0/+1)."""
    return factor.sign()

def sqrt(factor: 'Factor') -> 'Factor':
    """Square root (x<0 → NaN)."""
    return factor.sqrt()

def maximum(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise maximum."""
    return factor1.maximum(factor2)

def minimum(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise minimum."""
    return factor1.minimum(factor2)

def divide(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise division (div by 0 → NaN)."""
    return factor1.divide(factor2)

def inverse(factor: 'Factor') -> 'Factor':
    """Reciprocal: 1/x (x=0 → NaN)."""
    return factor.inverse()

def add(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise addition."""
    return factor1.add(factor2)

def multiply(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise multiplication."""
    return factor1.multiply(factor2)

def subtract(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise subtraction."""
    return factor1.subtract(factor2)

def power(base: 'Factor', exponent: Union['Factor', float]) -> 'Factor':
    """Element-wise power."""
    return base.power(exponent)

def reverse(factor: 'Factor') -> 'Factor':
    """Negate factor."""
    return factor.reverse()

def signed_power(base: 'Factor', exponent: Union['Factor', float]) -> 'Factor':
    """Sign-preserving power."""
    return base.signed_power(exponent)