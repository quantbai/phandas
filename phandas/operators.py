"""Functional operator API for Factor operations."""

from typing import Union, List, Optional
from .core import Factor 


def vector_neut(x: 'Factor', y: 'Factor') -> 'Factor':
    """Remove y's influence from x, keeping only the independent part.
    
    Formula: x - (x dot y / y dot y) * y
    
    Parameters
    ----------
    x : Factor
        Factor to process
    y : Factor
        Factor whose influence to remove
    
    Returns
    -------
    Factor
        x with y's influence removed
    """
    return x.vector_neut(y)

def regression_neut(y: 'Factor', x: 'Factor') -> 'Factor':
    """Remove x's influence from y using linear regression.
    
    Fits a line y = a + b*x, returns the residuals (actual - predicted).
    
    Parameters
    ----------
    y : Factor
        Factor to process
    x : Factor
        Factor whose influence to remove
    
    Returns
    -------
    Factor
        Residuals after removing x's linear effect
    """
    return y.regression_neut(x)

def group_neutralize(x: 'Factor', group: 'Factor') -> 'Factor':
    """Subtract group average from each value.
    
    Formula: x - mean(x within same group)
    
    Parameters
    ----------
    x : Factor
        Factor to neutralize
    group : Factor
        Group ID for each asset
    
    Returns
    -------
    Factor
        Values relative to group average
    """
    return x.group_neutralize(group)

def group(factor: 'Factor', mapping: dict) -> 'Factor':
    """Map symbol to group ID using a custom dict.
    
    Parameters
    ----------
    factor : Factor
        Base factor providing timestamp/symbol index
    mapping : dict
        Symbol to group ID mapping, e.g. {'ETH': 1, 'SOL': 1, 'ARB': 2}
    
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
    """Rank values from 0 to 1 across all assets at each time.
    
    Highest value gets 1.0, lowest gets close to 0.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Percentile rank (0-1) at each timestamp
    """
    return factor.rank()

def mean(factor: 'Factor') -> 'Factor':
    """Average across all assets at each time.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Average value at each timestamp
    """
    return factor.mean()

def median(factor: 'Factor') -> 'Factor':
    """Middle value across all assets at each time.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Median value at each timestamp
    """
    return factor.median()

def normalize(factor: 'Factor', use_std: bool = False, limit: float = 0.0) -> 'Factor':
    """Subtract average, optionally divide by spread.
    
    Formula: (x - mean) or (x - mean) / std
    
    Parameters
    ----------
    factor : Factor
        Input factor
    use_std : bool, default False
        If True, also divide by standard deviation
    limit : float, default 0.0
        If > 0, clip values to [-limit, +limit]
    
    Returns
    -------
    Factor
        Normalized factor centered at 0
    """
    return factor.normalize(use_std, limit)

def quantile(factor: 'Factor', driver: str = "gaussian", sigma: float = 1.0) -> 'Factor':
    """Transform ranks to follow a target distribution shape.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    driver : {'gaussian', 'uniform', 'cauchy'}, default 'gaussian'
        Target distribution (gaussian = bell curve)
    sigma : float, default 1.0
        Spread of the distribution
    
    Returns
    -------
    Factor
        Values reshaped to target distribution
    """
    return factor.quantile(driver, sigma)

def scale(factor: 'Factor', scale: float = 1.0, longscale: Optional[float] = None, 
          shortscale: Optional[float] = None) -> 'Factor':
    """Resize values so their absolute sum equals a target.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    scale : float, default 1.0
        Target for sum of absolute values
    longscale : float, optional
        Separate target for positive values
    shortscale : float, optional
        Separate target for negative values
    
    Returns
    -------
    Factor
        Rescaled factor
    """
    return factor.scale(scale, longscale, shortscale)

def zscore(factor: 'Factor') -> 'Factor':
    """Standardize to mean=0 and std=1 at each time.
    
    Formula: (x - mean) / std
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Standardized factor
    """
    return factor.zscore()

def spread(factor: 'Factor', pct: float = 0.5) -> 'Factor':
    """Simple long/short: go long top X%, short bottom X%.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    pct : float, default 0.5
        Percentage to include (top pct% gets +0.5, bottom pct% gets -0.5)
    
    Returns
    -------
    Factor
        Binary long/short weights
    """
    return factor.spread(pct)

def signal(factor: 'Factor') -> 'Factor':
    """Convert to trading weights (long +0.5, short -0.5, net zero).
    
    Subtracts mean then scales so positive values sum to 0.5
    and negative values sum to -0.5.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    
    Returns
    -------
    Factor
        Trading weights that sum to zero
    """
    return factor.signal()

def ts_rank(factor: 'Factor', window: int) -> 'Factor':
    """Where does today's value rank in the last N days? (0-1)
    
    1.0 = highest in window, 0 = lowest in window.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days to look at
    
    Returns
    -------
    Factor
        Percentile rank within the window
    """
    return factor.ts_rank(window)

def ts_mean(factor: 'Factor', window: int) -> 'Factor':
    """Average of the last N days.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days to average
    
    Returns
    -------
    Factor
        Rolling average
    """
    return factor.ts_mean(window)

def ts_median(factor: 'Factor', window: int) -> 'Factor':
    """Middle value of the last N days.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Rolling median
    """
    return factor.ts_median(window)

def ts_product(factor: 'Factor', window: int) -> 'Factor':
    """Multiply all values in the last N days.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Rolling product
    """
    return factor.ts_product(window)

def ts_sum(factor: 'Factor', window: int) -> 'Factor':
    """Add up all values in the last N days.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Rolling sum
    """
    return factor.ts_sum(window)

def ts_std_dev(factor: 'Factor', window: int) -> 'Factor':
    """How spread out are values in the last N days?
    
    Higher = more volatile, lower = more stable.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Rolling standard deviation
    """
    return factor.ts_std_dev(window)

def where(condition: 'Factor', x: Union['Factor', float], y: Union['Factor', float]) -> 'Factor':
    """If-else selection: pick x when condition is true, otherwise y.
    
    Parameters
    ----------
    condition : Factor
        True/False values for each cell
    x : Factor or float
        Value to use when True
    y : Factor or float
        Value to use when False
    
    Returns
    -------
    Factor
        Selected values
    """
    if not isinstance(x, Factor):
        raise TypeError("The `x` argument must be a Factor object for `where` operator.")
    return x.where(condition, other=y)

def ts_corr(factor1: 'Factor', factor2: 'Factor', window: int) -> 'Factor':
    """How closely do two factors move together over the last N days?
    
    +1 = move together, -1 = move opposite, 0 = no relationship.
    
    Parameters
    ----------
    factor1 : Factor
        First factor
    factor2 : Factor
        Second factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Rolling correlation (-1 to +1)
    """
    return factor1.ts_corr(factor2, window)

def ts_step(factor: 'Factor', start: int = 1) -> 'Factor':
    """Day counter for each asset (1, 2, 3, ...).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    start : int, default 1
        First day's number
    
    Returns
    -------
    Factor
        Day number for each row
    """
    return factor.ts_step(start)

def ts_delay(factor: 'Factor', window: int) -> 'Factor':
    """Get the value from N days ago.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        How many days back to look
    
    Returns
    -------
    Factor
        Value from N days ago
    """
    return factor.ts_delay(window)

def ts_delta(factor: 'Factor', window: int) -> 'Factor':
    """Change since N days ago.
    
    Formula: today - N_days_ago
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        How many days back to compare
    
    Returns
    -------
    Factor
        Difference from N days ago
    """
    return factor.ts_delta(window)

def ts_arg_max(factor: 'Factor', window: int) -> 'Factor':
    """When did the highest value occur in the last N days?
    
    Returns position: 0=oldest day, N-1=today.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Position of maximum value
    """
    return factor.ts_arg_max(window)

def ts_arg_min(factor: 'Factor', window: int) -> 'Factor':
    """When did the lowest value occur in the last N days?
    
    Returns position: 0=oldest day, N-1=today.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Position of minimum value
    """
    return factor.ts_arg_min(window)

def ts_min(factor: 'Factor', window: int) -> 'Factor':
    """Lowest value in the last N days.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Rolling minimum
    """
    return factor.ts_min(window)

def ts_max(factor: 'Factor', window: int) -> 'Factor':
    """Highest value in the last N days.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Rolling maximum
    """
    return factor.ts_max(window)

def ts_count_nans(factor: 'Factor', window: int) -> 'Factor':
    """How many missing values in the last N days?
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Count of missing values
    """
    return factor.ts_count_nans(window)

def ts_covariance(factor1: 'Factor', factor2: 'Factor', window: int) -> 'Factor':
    """How do two factors vary together over the last N days?
    
    Positive = move in same direction, negative = opposite.
    
    Parameters
    ----------
    factor1 : Factor
        First factor
    factor2 : Factor
        Second factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Rolling covariance
    """
    return factor1.ts_covariance(factor2, window)

def ts_quantile(factor: 'Factor', window: int, driver: str = "gaussian") -> 'Factor':
    """Transform rolling ranks to a target distribution shape.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    driver : {'gaussian', 'uniform', 'cauchy'}, default 'gaussian'
        Target distribution (gaussian = bell curve)
    
    Returns
    -------
    Factor
        Rolling quantile-transformed values
    """
    return factor.ts_quantile(window, driver)

def ts_kurtosis(factor: 'Factor', window: int) -> 'Factor':
    """Are there extreme values in the last N days? (tail heaviness)
    
    High = frequent extreme moves, low = mostly normal moves.
    Normal distribution has kurtosis = 0.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Rolling kurtosis (0 = normal, >0 = fat tails)
    """
    return factor.ts_kurtosis(window)

def ts_skewness(factor: 'Factor', window: int) -> 'Factor':
    """Is the distribution lopsided in the last N days?
    
    Positive = more extreme high values, negative = more extreme low values.
    Zero = symmetric distribution.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Rolling skewness
    """
    return factor.ts_skewness(window)

def ts_av_diff(factor: 'Factor', window: int) -> 'Factor':
    """How far is today's value from the N-day average?
    
    Formula: today - average(last N days)
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days for average
    
    Returns
    -------
    Factor
        Distance from rolling average
    """
    return factor.ts_av_diff(window)

def ts_scale(factor: 'Factor', window: int, constant: float = 0) -> 'Factor':
    """Scale values to 0-1 range based on last N days min/max.
    
    Formula: (today - min) / (max - min) + constant
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    constant : float, default 0
        Add this to the scaled value
    
    Returns
    -------
    Factor
        Scaled to 0-1 range
    """
    return factor.ts_scale(window, constant)

def ts_zscore(factor: 'Factor', window: int) -> 'Factor':
    """How unusual is today's value compared to recent history?
    
    Formula: (today - average) / std_dev
    Higher absolute value = more unusual.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days for comparison
    
    Returns
    -------
    Factor
        Rolling z-score
    """
    return factor.ts_zscore(window)

def ts_backfill(factor: 'Factor', window: int, k: int = 1) -> 'Factor':
    """Fill missing values with recent valid values.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        How far back to look for valid values
    k : int, default 1
        Use k-th most recent valid value (1=latest, 2=second-latest)
    
    Returns
    -------
    Factor
        Factor with missing values filled
    """
    return factor.ts_backfill(window, k)

def ts_decay_exp_window(factor: 'Factor', window: int, factor_arg: float = 1.0, nan: bool = True) -> 'Factor':
    """Weighted average where recent values count more (exponential weights).
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    factor_arg : float, default 1.0
        Decay rate (smaller = faster decay)
    nan : bool, default True
        Skip missing values if True
    
    Returns
    -------
    Factor
        Exponentially weighted average
    """
    return factor.ts_decay_exp_window(window, factor_arg, nan)

def ts_decay_linear(factor: 'Factor', window: int, dense: bool = False) -> 'Factor':
    """Weighted average where recent values count more (linear weights).
    
    Most recent day has weight N, oldest has weight 1.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    dense : bool, default False
        If True, skip missing values when weighting
    
    Returns
    -------
    Factor
        Linearly weighted average
    """
    return factor.ts_decay_linear(window, dense)

def ts_regression(y: 'Factor', x: 'Factor', window: int, lag: int = 0, rettype: int = 0) -> 'Factor':
    """Fit a line y = a + b*x over rolling window.
    
    Can return different parts of the regression result.
    
    Parameters
    ----------
    y : Factor
        Value to predict
    x : Factor
        Value used for prediction
    window : int
        Number of past days
    lag : int, default 0
        Shift x by this many days
    rettype : int, default 0
        What to return: 0=residual, 1=intercept, 2=slope, 3=predicted,
        4=sum squared error, 5=total variance, 6=R-squared, 7=mean squared error
    
    Returns
    -------
    Factor
        Regression result
    """
    return y.ts_regression(x, window, lag, rettype)

def ts_cv(factor: 'Factor', window: int) -> 'Factor':
    """Relative volatility: std / abs(mean) over rolling window.
    
    Higher = more volatile relative to average level.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Coefficient of variation
    """
    return factor.ts_cv(window)

def ts_jumpiness(factor: 'Factor', window: int) -> 'Factor':
    """How choppy is the price movement? (frequent small jumps vs smooth trend)
    
    Formula: sum of daily changes / total range
    Higher = more choppy, lower = smoother trend.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Jumpiness ratio
    """
    return factor.ts_jumpiness(window)

def ts_trend_strength(factor: 'Factor', window: int) -> 'Factor':
    """How strong is the trend direction? (0=no trend, 1=perfect trend)
    
    Uses R-squared from fitting a line through time.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Trend strength (0 to 1)
    """
    return factor.ts_trend_strength(window)

def ts_vr(factor: 'Factor', window: int, k: int = 2) -> 'Factor':
    """Compare volatility at different time scales.
    
    Ratio of k-day variance to 1-day variance.
    >1 suggests trending, <1 suggests mean-reverting.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    k : int, default 2
        Compare k-day changes to 1-day changes
    
    Returns
    -------
    Factor
        Variance ratio
    """
    return factor.ts_vr(window, k)

def ts_autocorr(factor: 'Factor', window: int, lag: int = 1) -> 'Factor':
    """How similar is today's value to N days ago?
    
    +1 = very similar, -1 = opposite, 0 = no pattern.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    lag : int, default 1
        Compare to value from lag days ago
    
    Returns
    -------
    Factor
        Autocorrelation
    """
    return factor.ts_autocorr(window, lag)

def ts_reversal_count(factor: 'Factor', window: int) -> 'Factor':
    """How often does direction change? (up to down or down to up)
    
    Higher = more choppy, lower = more consistent direction.
    
    Parameters
    ----------
    factor : Factor
        Input factor
    window : int
        Number of past days
    
    Returns
    -------
    Factor
        Reversal frequency (0 to 1)
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
    """Display Factor or Panel visualization.
    
    Parameters
    ----------
    obj : Factor or Panel
        Object to display
    """
    if hasattr(obj, 'show'):
        obj.show()
    else:
        raise TypeError(f"show() not supported for {type(obj).__name__}")


def to_csv(obj, path: str) -> None:
    """Export Factor or Panel to CSV file.
    
    Parameters
    ----------
    obj : Factor or Panel
        Object to save
    path : str
        Output file path
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
    """
    if hasattr(obj, 'to_df'):
        return obj.to_df()
    else:
        raise TypeError(f"to_df() not supported for {type(obj).__name__}")