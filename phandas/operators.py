"""Functional operator API for Factor operations."""

from typing import Union, List, Optional
from .core import Factor 


def vector_neut(x: 'Factor', y: 'Factor') -> 'Factor':
    """Remove y projection from x via vector orthogonalization.
    
    Parameters
    ----------
    x : Factor
        Factor to orthogonalize
    y : Factor
        Factor defining projection direction
    
    Returns
    -------
    Factor
        Orthogonalized factor with y projection removed
    """
    return x.vector_neut(y)

def regression_neut(y: 'Factor', x: 'Factor') -> 'Factor':
    """Orthogonalize y relative to x via OLS residuals.
    
    Parameters
    ----------
    y : Factor
        Factor to orthogonalize
    x : Factor
        Factor for regression
    
    Returns
    -------
    Factor
        Residuals from OLS regression of y on x
    """
    return y.regression_neut(x)

def group_neutralize(x: 'Factor', group: 'Factor') -> 'Factor':
    """Neutralize factor against groups (demean within group).
    
    Parameters
    ----------
    x : Factor
        Factor to neutralize
    group : Factor
        Factor defining the group for each asset
    
    Returns
    -------
    Factor
        Neutralized factor (x - group_mean)
    """
    return x.group_neutralize(group)

def group(factor: 'Factor', mapping: Union[str, dict]) -> 'Factor':
    """Map symbol to group ID using predefined definition or custom dict.
    
    Parameters
    ----------
    factor : Factor
        Base factor providing timestamp/symbol index
    mapping : str or dict
        Mapping name (in constants.GROUP_DEFINITIONS) or dict
    
    Returns
    -------
    Factor
        Group ID factor
    """
    return factor.group(mapping)

def group_mean(x: 'Factor', group: 'Factor') -> 'Factor':
    """Calculate group mean for each asset.
    
    Parameters
    ----------
    x : Factor
        Factor to calculate mean for
    group : Factor
        Factor defining the group for each asset
    
    Returns
    -------
    Factor
        Mean value of the group the asset belongs to
    """
    return x.group_mean(group)

def group_median(x: 'Factor', group: 'Factor') -> 'Factor':
    """Calculate group median for each asset.
    
    Parameters
    ----------
    x : Factor
        Factor to calculate median for
    group : Factor
        Factor defining the group for each asset
    
    Returns
    -------
    Factor
        Median value of the group the asset belongs to
    """
    return x.group_median(group)

def group_rank(x: 'Factor', group: 'Factor') -> 'Factor':
    """Calculate percentile rank within each group.
    
    Parameters
    ----------
    x : Factor
        Factor to rank
    group : Factor
        Factor defining the group for each asset
    
    Returns
    -------
    Factor
        Percentile rank (0-1) within the group
    """
    return x.group_rank(group)

def group_scale(x: 'Factor', group: 'Factor') -> 'Factor':
    """Scale values within each group to 0-1 range.
    
    Parameters
    ----------
    x : Factor
        Factor to scale
    group : Factor
        Factor defining the group for each asset
    
    Returns
    -------
    Factor
        Scaled values: (x - min) / (max - min)
    """
    return x.group_scale(group)

def group_zscore(x: 'Factor', group: 'Factor') -> 'Factor':
    """Calculate Z-score within each group.
    
    Parameters
    ----------
    x : Factor
        Factor to calculate zscore for
    group : Factor
        Factor defining the group for each asset
    
    Returns
    -------
    Factor
        Z-score: (x - mean) / std
    """
    return x.group_zscore(group)

def group_normalize(x: 'Factor', group: 'Factor', scale: float = 1.0) -> 'Factor':
    """Normalize such that each group's absolute sum equals scale.
    
    Parameters
    ----------
    x : Factor
        Factor to normalize
    group : Factor
        Factor defining the group for each asset
    scale : float, default 1.0
        Target sum of absolute values for each group
    
    Returns
    -------
    Factor
        Normalized values: x / sum(|x|_group) * scale
    """
    return x.group_normalize(group, scale)

def rank(factor: 'Factor') -> 'Factor':
    """Cross-sectional percentile rank (0-1).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Ranked factor per timestamp
    """
    return factor.rank()

def mean(factor: 'Factor') -> 'Factor':
    """Cross-sectional mean per timestamp.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Mean-replicated factor per timestamp
    """
    return factor.mean()

def median(factor: 'Factor') -> 'Factor':
    """Cross-sectional median per timestamp.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Median-replicated factor per timestamp
    """
    return factor.median()

def normalize(factor: 'Factor', use_std: bool = False, limit: float = 0.0) -> 'Factor':
    """Cross-sectional demean with optional std normalization.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    use_std : bool, default False
        Whether to divide by standard deviation
    limit : float, default 0.0
        Clipping limit (e.g., ±limit)
    
    Returns
    -------
    Factor
        Demeaned and optionally normalized factor
    """
    return factor.normalize(use_std, limit)

def quantile(factor: 'Factor', driver: str = "gaussian", sigma: float = 1.0) -> 'Factor':
    """Cross-sectional quantile transform (normal/uniform/Cauchy).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    driver : {'gaussian', 'uniform', 'cauchy'}, default 'gaussian'
        Quantile transform distribution
    sigma : float, default 1.0
        Scale parameter
    
    Returns
    -------
    Factor
        Quantile-transformed factor
    """
    return factor.quantile(driver, sigma)

def scale(factor: 'Factor', scale: float = 1.0, longscale: Optional[float] = None, 
          shortscale: Optional[float] = None) -> 'Factor':
    """Scale to sum(|factor|)=scale with optional separate long/short sizing.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    scale : float, default 1.0
        Target sum of absolute values
    longscale : float, optional
        Long-only scale. If None, uses `scale`.
    shortscale : float, optional
        Short-only scale. If None, uses `scale`.
    
    Returns
    -------
    Factor
        Scaled factor
    """
    return factor.scale(scale, longscale, shortscale)

def zscore(factor: 'Factor') -> 'Factor':
    """Cross-sectional standardization (mean=0, std=1).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Standardized factor per timestamp
    """
    return factor.zscore()

def spread(factor: 'Factor', pct: float = 0.5) -> 'Factor':
    """Binary long-short signal based on top and bottom percentiles.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    pct : float, default 0.5
        Percentile threshold (top pct% → +0.5, bottom pct% → -0.5)
    
    Returns
    -------
    Factor
        Binary long-short signal per timestamp
    """
    return factor.spread(pct)

def signal(factor: 'Factor') -> 'Factor':
    """Dollar-neutral signal (long sum=0.5, short=-0.5).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Demeaned and scaled to dollar-neutral weights
    """
    return factor.signal()

def ts_rank(factor: 'Factor', window: int) -> 'Factor':
    """Rolling percentile rank within window.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling percentile rank
    """
    return factor.ts_rank(window)

def ts_mean(factor: 'Factor', window: int) -> 'Factor':
    """Rolling mean over window.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling mean, NaN if window incomplete
    """
    return factor.ts_mean(window)

def ts_median(factor: 'Factor', window: int) -> 'Factor':
    """Rolling median over window.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling median, NaN if window incomplete
    """
    return factor.ts_median(window)

def ts_product(factor: 'Factor', window: int) -> 'Factor':
    """Rolling product over window.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling product, NaN if window incomplete
    """
    return factor.ts_product(window)

def ts_sum(factor: 'Factor', window: int) -> 'Factor':
    """Rolling sum over window.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling sum, NaN if window incomplete
    """
    return factor.ts_sum(window)

def ts_std_dev(factor: 'Factor', window: int) -> 'Factor':
    """Rolling standard deviation over window.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling std dev, NaN if window incomplete
    """
    return factor.ts_std_dev(window)

def where(condition: 'Factor', x: Union['Factor', float], y: Union['Factor', float]) -> 'Factor':
    """Select x when condition is True, else y.
    
    Parameters
    ----------
    condition : Factor
        Boolean condition factor
    x : Factor or float
        Values where condition is True
    y : Factor or float
        Values where condition is False
    
    Returns
    -------
    Factor
        Conditional selection result
    """
    if not isinstance(x, Factor):
        raise TypeError("The `x` argument must be a Factor object for `where` operator.")
    return x.where(condition, other=y)

def ts_corr(factor1: 'Factor', factor2: 'Factor', window: int) -> 'Factor':
    """Rolling Pearson correlation over window.
    
    Parameters
    ----------
    factor1 : Factor
        First factor
    factor2 : Factor
        Second factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling correlation, NaN if window incomplete
    """
    return factor1.ts_corr(factor2, window)

def ts_step(factor: 'Factor', start: int = 1) -> 'Factor':
    """Time step counter (1, 2, 3, ...) per symbol.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    start : int, default 1
        Starting value
    
    Returns
    -------
    Factor
        Incrementing time steps per symbol
    """
    return factor.ts_step(start)

def ts_delay(factor: 'Factor', window: int) -> 'Factor':
    """Lag factor by window periods.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Lag periods
    
    Returns
    -------
    Factor
        Lagged factor
    """
    return factor.ts_delay(window)

def ts_delta(factor: 'Factor', window: int) -> 'Factor':
    """Difference between current and lagged value.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Lag periods
    
    Returns
    -------
    Factor
        First difference (x - lag(x, window))
    """
    return factor.ts_delta(window)

def ts_arg_max(factor: 'Factor', window: int) -> 'Factor':
    """Relative index of maximum in window (0=oldest).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Index of max value in window
    """
    return factor.ts_arg_max(window)

def ts_arg_min(factor: 'Factor', window: int) -> 'Factor':
    """Relative index of minimum in window (0=oldest).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Index of min value in window
    """
    return factor.ts_arg_min(window)

def ts_min(factor: 'Factor', window: int) -> 'Factor':
    """Rolling minimum over window.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling minimum, NaN if window incomplete
    """
    return factor.ts_min(window)

def ts_max(factor: 'Factor', window: int) -> 'Factor':
    """Rolling maximum over window.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling maximum, NaN if window incomplete
    """
    return factor.ts_max(window)

def ts_count_nans(factor: 'Factor', window: int) -> 'Factor':
    """Count NaN values in rolling window.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Count of NaN values in window
    """
    return factor.ts_count_nans(window)

def ts_covariance(factor1: 'Factor', factor2: 'Factor', window: int) -> 'Factor':
    """Rolling covariance over window.
    
    Parameters
    ----------
    factor1 : Factor
        First factor
    factor2 : Factor
        Second factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling covariance, NaN if window incomplete
    """
    return factor1.ts_covariance(factor2, window)

def ts_quantile(factor: 'Factor', window: int, driver: str = "gaussian") -> 'Factor':
    """Rolling quantile transform (normal/uniform/Cauchy).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    driver : {'gaussian', 'uniform', 'cauchy'}, default 'gaussian'
        Quantile transform distribution
    
    Returns
    -------
    Factor
        Rolling quantile-transformed factor
    """
    return factor.ts_quantile(window, driver)

def ts_kurtosis(factor: 'Factor', window: int) -> 'Factor':
    """Rolling excess kurtosis.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling excess kurtosis: E[(x-mean)^4]/std^4 - 3
    """
    return factor.ts_kurtosis(window)

def ts_skewness(factor: 'Factor', window: int) -> 'Factor':
    """Rolling sample skewness with Bessel correction.
    
    Expression: (power(x - ts_mean(x, n), 3).ts_sum(n) * n) / 
                (power(power(x - ts_mean(x, n), 2).ts_sum(n), 1.5) * (n-1) * (n-2))
    Composed of: ts_mean, power, ts_sum
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling skewness, NaN if window incomplete
    """
    return factor.ts_skewness(window)

def ts_av_diff(factor: 'Factor', window: int) -> 'Factor':
    """Deviation from rolling mean.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Deviation from rolling mean
    """
    return factor.ts_av_diff(window)

def ts_scale(factor: 'Factor', window: int, constant: float = 0) -> 'Factor':
    """Rolling min-max scale.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    constant : float, default 0
        Offset constant
    
    Returns
    -------
    Factor
        Scaled factor: (x-min)/(max-min) + constant
    """
    return factor.ts_scale(window, constant)

def ts_zscore(factor: 'Factor', window: int) -> 'Factor':
    """Rolling Z-score over window.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling Z-score, NaN if window incomplete
    """
    return factor.ts_zscore(window)

def ts_backfill(factor: 'Factor', window: int, k: int = 1) -> 'Factor':
    """Backfill NaN with k-th most recent non-NaN in window.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    k : int, default 1
        Which recent non-NaN to use (1=most recent, 2=second-most, ...)
    
    Returns
    -------
    Factor
        Factor with NaN backfilled
    """
    return factor.ts_backfill(window, k)

def ts_decay_exp_window(factor: 'Factor', window: int, factor_arg: float = 1.0, nan: bool = True) -> 'Factor':
    """Exponentially weighted rolling average (recent heavier).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    factor_arg : float, default 1.0
        Exponential decay factor
    nan : bool, default True
        Whether to skip NaN values
    
    Returns
    -------
    Factor
        Exponentially weighted rolling average
    """
    return factor.ts_decay_exp_window(window, factor_arg, nan)

def ts_decay_linear(factor: 'Factor', window: int, dense: bool = False) -> 'Factor':
    """Linearly weighted rolling average (recent heavier).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    dense : bool, default False
        Whether to use dense weighting
    
    Returns
    -------
    Factor
        Linearly weighted rolling average
    """
    return factor.ts_decay_linear(window, dense)

def ts_regression(y: 'Factor', x: 'Factor', window: int, lag: int = 0, rettype: int = 0) -> 'Factor':
    """Rolling OLS regression with multiple return types.
    
    Parameters
    ----------
    y : Factor
        Dependent variable
    x : Factor
        Independent variable
    window : int
        Window size (periods)
    lag : int, default 0
        Lag for x (periods)
    rettype : int, default 0
        Return type: 0=residual, 1=α, 2=β, 3=pred, 4=SSE, 5=SST, 
        6=R², 7=MSE, 8=SE(β), 9=SE(α)
    
    Returns
    -------
    Factor
        Regression result of specified type
    """
    return y.ts_regression(x, window, lag, rettype)

def ts_cv(factor: 'Factor', window: int) -> 'Factor':
    """Rolling coefficient of variation.
    
    Expression: ts_std_dev(x, n) / (abs(ts_mean(x, n)) + eps)
    Composed of: ts_mean, ts_std_dev, abs
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling CV: std / abs(mean)
    """
    return factor.ts_cv(window)

def ts_jumpiness(factor: 'Factor', window: int) -> 'Factor':
    """Rolling jumpiness (jump intensity).
    
    Expression: ts_sum(abs(ts_delta(x, 1)), n) / (ts_max(x, n) - ts_min(x, n) + eps)
    Composed of: ts_delta, abs, ts_sum, ts_max, ts_min
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling jumpiness: sum(|diff|) / (max - min)
    """
    return factor.ts_jumpiness(window)

def ts_trend_strength(factor: 'Factor', window: int) -> 'Factor':
    """Rolling trend strength (R² of linear regression on time).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling trend strength (R² value)
    """
    return factor.ts_trend_strength(window)

def ts_vr(factor: 'Factor', window: int, k: int = 2) -> 'Factor':
    """Rolling variance ratio.
    
    Expression: power(ts_std_dev(ts_delta(x, k), n), 2) / 
                (k * power(ts_std_dev(ts_delta(x, 1), n), 2) + eps)
    Composed of: ts_delta, ts_std_dev, power
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    k : int, default 2
        Variance ratio period
    
    Returns
    -------
    Factor
        Rolling variance ratio: Var(k-period) / (k * Var(1-period))
    """
    return factor.ts_vr(window, k)

def ts_autocorr(factor: 'Factor', window: int, lag: int = 1) -> 'Factor':
    """Rolling autocorrelation at specified lag.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    lag : int, default 1
        Lag for autocorrelation
    
    Returns
    -------
    Factor
        Rolling autocorrelation
    """
    return factor.ts_autocorr(window, lag)

def ts_reversal_count(factor: 'Factor', window: int) -> 'Factor':
    """Rolling reversal count (direction changes).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Window size (periods)
    
    Returns
    -------
    Factor
        Rolling reversal count: direction changes / window
    """
    return factor.ts_reversal_count(window)

def log(factor: 'Factor', base: Optional[float] = None) -> 'Factor':
    """Logarithm with optional base.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    base : float, optional
        Logarithm base (None → natural logarithm)
    
    Returns
    -------
    Factor
        Logarithm of factor
    """
    return factor.log(base)

def ln(factor: 'Factor') -> 'Factor':
    """Natural logarithm.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Natural logarithm of factor
    """
    return factor.ln()

def s_log_1p(factor: 'Factor') -> 'Factor':
    """Sign-preserving logarithm.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Sign-preserving log: sign(x)·ln(1+|x|)
    """
    return factor.s_log_1p()

def sign(factor: 'Factor') -> 'Factor':
    """Sign of values.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Sign: -1, 0, or +1
    """
    return factor.sign()

def sqrt(factor: 'Factor') -> 'Factor':
    """Square root.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Square root (x<0 → NaN)
    """
    return factor.sqrt()

def inverse(factor: 'Factor') -> 'Factor':
    """Reciprocal (1/x).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Reciprocal (x=0 → NaN)
    """
    return factor.inverse()

def maximum(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise maximum.
    
    Parameters
    ----------
    factor1 : Factor
        First factor
    factor2 : Factor or float
        Second factor or scalar
    
    Returns
    -------
    Factor
        Element-wise maximum
    """
    return factor1.maximum(factor2)

def minimum(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise minimum.
    
    Parameters
    ----------
    factor1 : Factor
        First factor
    factor2 : Factor or float
        Second factor or scalar
    
    Returns
    -------
    Factor
        Element-wise minimum
    """
    return factor1.minimum(factor2)

def power(base: 'Factor', exponent: Union['Factor', float]) -> 'Factor':
    """Element-wise power.
    
    Parameters
    ----------
    base : Factor
        Base factor
    exponent : Factor or float
        Exponent
    
    Returns
    -------
    Factor
        Element-wise power
    """
    return base.power(exponent)

def signed_power(base: 'Factor', exponent: Union['Factor', float]) -> 'Factor':
    """Sign-preserving power.
    
    Parameters
    ----------
    base : Factor
        Base factor
    exponent : Factor or float
        Exponent
    
    Returns
    -------
    Factor
        Sign-preserving power: sign(x)·|x|^exp
    """
    return base.signed_power(exponent)

def divide(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise division.
    
    Parameters
    ----------
    factor1 : Factor
        Numerator
    factor2 : Factor or float
        Denominator
    
    Returns
    -------
    Factor
        Element-wise quotient (div by 0 → NaN)
    """
    return factor1.divide(factor2)

def add(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise addition.
    
    Parameters
    ----------
    factor1 : Factor
        First factor
    factor2 : Factor or float
        Second factor or scalar
    
    Returns
    -------
    Factor
        Element-wise sum
    """
    return factor1.add(factor2)

def multiply(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise multiplication.
    
    Parameters
    ----------
    factor1 : Factor
        First factor
    factor2 : Factor or float
        Second factor or scalar
    
    Returns
    -------
    Factor
        Element-wise product
    """
    return factor1.multiply(factor2)

def subtract(factor1: 'Factor', factor2: Union['Factor', float]) -> 'Factor':
    """Element-wise subtraction.
    
    Parameters
    ----------
    factor1 : Factor
        First factor
    factor2 : Factor or float
        Second factor or scalar
    
    Returns
    -------
    Factor
        Element-wise difference
    """
    return factor1.subtract(factor2)

def reverse(factor: 'Factor') -> 'Factor':
    """Negate factor values.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Negated factor (-x)
    """
    return factor.reverse()


def show(obj) -> None:
    """Functional show for Factor/Panel.
    
    Parameters
    ----------
    obj : Factor or Panel
        Object to display
    
    Examples
    --------
    >>> show(factor)
    >>> factor.show()
    """
    if hasattr(obj, 'show'):
        obj.show()
    else:
        raise TypeError(f"show() not supported for {type(obj).__name__}")


def to_csv(obj, path: str) -> None:
    """Functional to_csv for Factor/Panel.
    
    Parameters
    ----------
    obj : Factor or Panel
        Object to save
    path : str
        File path to save CSV
    
    Examples
    --------
    >>> to_csv(factor, 'factor.csv')
    >>> factor.to_csv('factor.csv')
    """
    if hasattr(obj, 'to_csv'):
        obj.to_csv(path)
    else:
        raise TypeError(f"to_csv() not supported for {type(obj).__name__}")


def to_df(obj) -> 'pd.DataFrame':
    """Convert Factor or Panel to pandas DataFrame.
    
    Parameters
    ----------
    obj : Factor or Panel
        Object to convert
    
    Returns
    -------
    pd.DataFrame
        Copy of underlying data
    
    Examples
    --------
    >>> df = to_df(factor)
    >>> df = factor.to_df()
    """
    if hasattr(obj, 'to_df'):
        return obj.to_df()
    else:
        raise TypeError(f"to_df() not supported for {type(obj).__name__}")