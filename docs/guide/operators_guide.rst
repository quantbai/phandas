Operators Guide
===============

Phandas provides **50+ operators** for factor construction. Categorized into four types: cross-sectional, time series, neutralization, and math operations.

.. contents::
   :local:
   :depth: 2

Core Concepts
-------------

Factor Object and Panel Data Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core of Phandas is the **Factor object**, representing a complete time series panel data for a factor.

**Data Structure**: Each Factor contains three columns:

- ``timestamp``: Timestamp (date or datetime)
- ``symbol``: Asset code (e.g., 'BTC', 'ETH')
- ``factor``: Factor value (float)

This structure is called **long-format panel data**, the standard format in quantitative finance::

    timestamp    symbol    factor
    2024-01-01   BTC       45000.0
    2024-01-01   ETH       2500.0
    2024-01-02   BTC       46000.0
    2024-01-02   ETH       2550.0

Operators: Feature Engineering for Alpha Factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Operators** are functions that transform Factor objects, essentially **feature engineering for quantitative finance**.

**Purpose**: Transform raw market data (price, volume) into predictive **alpha factors**.

**Workflow**::

    Raw Data (OHLCV) 
      → Operator Transform (Feature Engineering)
      → Alpha Factor
      → Backtest Validation
      → Live Trading

**Operator Categories**:

1. **Cross-sectional operators**: Calculate independently at each timestamp (e.g., ranking, standardization)
2. **Time series operators**: Calculate across time dimension (e.g., moving average, momentum)
3. **Neutralization operators**: Remove unwanted factor exposure (e.g., volume bias)
4. **Math operators**: Basic mathematical operations (e.g., log, power)

**Design Philosophy**:

- **Composability**: Operators can be chained to build complex factors
- **Vectorization**: All calculations automatically parallelize across assets
- **NaN Safety**: Properly handles missing values, avoids data leakage

Cross-sectional Operators
-------------------------

Calculate independently at each time cross-section (date), used for standardization and ranking.

Ranking
~~~~~~~

**rank()** — Percentile ranking (0-1)
    Ranks factor values within each day, outputs 0-1 ranking. NaN returns NaN.

    ::

        factor_ranked = rank(factor)

**normalize()** — Demean
    Removes mean per day. Optional std division and clipping.

    ::

        factor_norm = normalize(factor)
        factor_norm_std = normalize(factor, use_std=True)  # Standard score

**zscore()** — Standardization (μ=0, σ=1)
    Equivalent to ``normalize(use_std=True)``.

    ::

        factor_z = zscore(factor)

Aggregate Statistics
~~~~~~~~~~~~~~~~~~~~

**mean()** — Cross-sectional mean
    Calculates daily mean (often used for diagnostics).

    ::

        mean_factor = mean(factor)

**median()** — Cross-sectional median
    Calculates daily median.

    ::

        median_factor = median(factor)

Transformation and Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~

**scale()** — Scale by absolute value
    Makes sum of absolute values equal to specified value (default 1.0).

    ::

        factor_scaled = scale(factor, scale=1.0)
        # Support separate long/short scaling
        factor_scaled = scale(factor, long_scale=0.5, short_scale=-0.5)

**quantile()** — Quantile transform
    Rank → Normal/Uniform/Cauchy PPF, supports scaling.

    ::

        factor_normal = quantile(factor, driver="gaussian", sigma=1.0)
        factor_uniform = quantile(factor, driver="uniform")

**spread()** — Binary signal
    Top pct% set to +0.5, bottom pct% set to -0.5, rest 0.

    ::

        signal = spread(factor, pct=0.3)  # Long/short top/bottom 30%

**signal()** — Dollar-neutral signal
    Demean, scale by absolute value so long sum = 0.5, short sum = -0.5.

    ::

        dn_signal = signal(factor)

Time Series Operators
---------------------

Calculate on each asset's time series, used for extracting momentum, mean reversion, volatility, etc.

Delay and Difference
~~~~~~~~~~~~~~~~~~~~

**ts_delay(factor, window)** — Lag
    Shifts data backward by window periods.

    ::

        prev_close = ts_delay(close, 1)

**ts_delta(factor, window)** — Change
    Difference between current and window periods ago: x - x_{t-window}.

    ::

        returns = ts_delta(close, 1)  # Daily returns

Basic Statistics
~~~~~~~~~~~~~~~~

**ts_mean(factor, window)** — Rolling mean
    Calculates mean over window periods (requires complete window).

    ::

        ma_20 = ts_mean(close, 20)

**ts_median(factor, window)** — Rolling median
    Calculates median over window periods.

    ::

        median_20 = ts_median(close, 20)

**ts_sum(factor, window)** — Rolling sum
    Calculates cumulative sum over window periods.

    ::

        volume_sum_10 = ts_sum(volume, 10)

**ts_product(factor, window)** — Rolling product
    Calculates cumulative product over window periods.

    ::

        cumprod_5 = ts_product(close, 5)

**ts_std_dev(factor, window)** — Rolling standard deviation
    Calculates standard deviation (volatility) over window periods.

    ::

        volatility_20 = ts_std_dev(close, 20)

Ranking and Extrema
~~~~~~~~~~~~~~~~~~~

**ts_rank(factor, window)** — Rolling rank
    Calculates percentile rank within window periods.

    ::

        rank_10 = ts_rank(close, 10)

**ts_max(factor, window)** — Rolling maximum
    Calculates maximum over window periods.

    ::

        highest_20 = ts_max(high, 20)

**ts_min(factor, window)** — Rolling minimum
    Calculates minimum over window periods.

    ::

        lowest_20 = ts_min(low, 20)

**ts_arg_max(factor, window)** — Periods since maximum
    Returns 0-1 relative index (0=earliest, window-1=latest).

    ::

        periods_since_max = ts_arg_max(close, 20)

**ts_arg_min(factor, window)** — Periods since minimum
    Returns 0-1 relative index.

    ::

        periods_since_min = ts_arg_min(close, 20)

Higher-order Statistics
~~~~~~~~~~~~~~~~~~~~~~~

**ts_skewness(factor, window)** — Rolling skewness
    Calculates sample skewness over window periods (with Bessel correction).

    ::

        skew_20 = ts_skewness(close, 20)

**ts_kurtosis(factor, window)** — Rolling kurtosis
    Calculates excess kurtosis over window periods.

    ::

        kurt_20 = ts_kurtosis(returns, 20)

Standardization
~~~~~~~~~~~~~~~

**ts_zscore(factor, window)** — Rolling z-score
    Calculates (x - mean) / std within window.

    ::

        zscore_20 = ts_zscore(close, 20)

**ts_scale(factor, window, constant)** — Rolling min-max scaling
    Calculates (x - min) / (max - min) + constant.

    ::

        scaled_20 = ts_scale(close, 20)

**ts_quantile(factor, window, driver)** — Rolling quantile transform
    Rank within window → Normal/Uniform/Cauchy PPF.

    ::

        ts_q_normal = ts_quantile(close, 20, driver="gaussian")

Decay Weighting
~~~~~~~~~~~~~~~

**ts_decay_linear(factor, window, dense)** — Linear decay weighting
    Recent data weighted higher, linearly decreasing.

    ::

        factor_decay_lin = ts_decay_linear(factor, 20)

**ts_decay_exp_window(factor, window, factor=0.9, nan)** — Exponential decay weighting
    Recent data weighted exponentially higher.

    ::

        factor_decay_exp = ts_decay_exp_window(factor, 20, factor=0.95)

Correlation and Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~

**ts_corr(factor1, factor2, window)** — Rolling Pearson correlation
    Calculates correlation coefficient between two factors over window periods.

    ::

        corr_momentum_volume = ts_corr(momentum, volume, 20)

**ts_covariance(factor1, factor2, window)** — Rolling covariance
    Calculates covariance between two factors over window periods.

    ::

        cov_close_volume = ts_covariance(close, volume, 20)

**ts_regression(y, x, window, lag, rettype)** — Rolling OLS regression
    Calculates y = α + β·x coefficients within window.

    - rettype=0: Residuals (default)
    - rettype=1: α (intercept)
    - rettype=2: β (slope)
    - rettype=3: Predicted values
    - rettype=6: R²

    ::

        residual = ts_regression(close, open, 20, rettype=0)
        beta = ts_regression(close, momentum, 20, rettype=2)

Other
~~~~~

**ts_count_nans(factor, window)** — Count NaNs
    Counts NaN values within window.

    ::

        nan_count = ts_count_nans(factor, 10)

**ts_backfill(factor, window, k)** — NaN backfill
    Fills NaN with k-th most recent non-NaN value within window.

    ::

        factor_filled = ts_backfill(factor, 20, k=1)

**ts_step(start)** — Time counter
    Generates incrementing sequence per asset: 1, 2, 3, ...

    ::

        time_counter = ts_step(1)

**ts_av_diff(factor, window)** — Average deviation
    Calculates x - ts_mean(x, window).

    ::

        deviation = ts_av_diff(close, 20)

Neutralization Operators
------------------------

Remove linear correlation between factor and specific variables.

Vector Neutralization
~~~~~~~~~~~~~~~~~~~~~

**vector_neut(x, y)** — Vector projection orthogonalization
    Removes linear projection of x onto y, retains orthogonal component. Uses dot product.

    ::

        # Remove correlation between momentum and volume
        momentum_neutral = vector_neut(momentum, rank(-volume))

Regression Neutralization
~~~~~~~~~~~~~~~~~~~~~~~~~

**regression_neut(y, x)** — OLS residual neutralization
    Removes linear dependence of y on x (can be multiple) via OLS regression.

    ::

        # Neutralize against both open price and volume
        factor_neutral = regression_neut(
            factor, 
            [open, volume]
        )

Math Operators
--------------

Basic mathematical operations and function transforms.

Elementary Functions
~~~~~~~~~~~~~~~~~~~~

**log(factor, base)** — Log transform
    Natural log (base=None) or specified base. x ≤ 0 → NaN.

    ::

        log_close = log(close)
        log2_volume = log(volume, base=2)

**ln(factor)** — Natural logarithm
    Equivalent to ``log(factor)``.

    ::

        ln_close = ln(close)

**sqrt(factor)** — Square root
    x < 0 → NaN.

    ::

        sqrt_volume = sqrt(volume)

**s_log_1p(factor)** — Sign-preserving log
    sign(x)·ln(1+|x|), preserves sign, handles zero.

    ::

        sl_returns = s_log_1p(returns)

Power and Roots
~~~~~~~~~~~~~~~

**power(base, exponent)** — Power function
    Calculates base^exponent, invalid values → NaN.

    ::

        factor_sq = power(factor, 2)

**signed_power(base, exponent)** — Sign-preserving power
    sign(x) times |x|^exponent, preserves sign.

    ::

        factor_pow = signed_power(factor, 0.5)

Sign Functions
~~~~~~~~~~~~~~

**sign(factor)** — Sign function
    Returns -1/0/+1.

    ::

        sign_factor = sign(factor)

**inverse(factor)** — Reciprocal
    Calculates 1/x, x=0 → NaN.

    ::

        inv_factor = inverse(factor)

Comparison and Conditional
~~~~~~~~~~~~~~~~~~~~~~~~~~

**maximum(factor1, factor2)** — Element-wise maximum
    Takes maximum of two factors element by element.

    ::

        max_factor = maximum(factor1, factor2)

**minimum(factor1, factor2)** — Element-wise minimum
    Takes minimum of two factors element by element.

    ::

        min_factor = minimum(factor1, factor2)

**where(condition, x, y)** — Conditional selection
    Selects x when condition=True, otherwise y.

    ::

        filtered = where(factor > 0, factor, 0)

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

Supports direct Python operators or functions:

- **add(a, b)** or ``a + b`` — Addition
- **subtract(a, b)** or ``a - b`` — Subtraction
- **multiply(a, b)** or ``a * b`` — Multiplication
- **divide(a, b)** or ``a / b`` — Division (div by 0 → NaN)
- **power(a, b)** or ``a ** b`` — Power

::

    factor = momentum + 0.5 * reversion
    ratio = close / open
    scaled = factor / ts_mean(factor, 20)

Common Combination Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Momentum Factor
~~~~~~~~~~~~~~~

::

    # Simple momentum (20-day returns)
    momentum = (close / ts_delay(close, 20)) - 1
    factor = rank(momentum)

    # Multi-period momentum combination
    mom_short = rank((close / ts_delay(close, 5)) - 1)   # Short-term momentum
    mom_long = rank((close / ts_delay(close, 20)) - 1)   # Long-term momentum
    
    # Equal-weight combination (reduces parameter sensitivity)
    momentum = 0.5 * mom_short + 0.5 * mom_long
    
    # Neutralize against high volume (avoid liquidity impact)
    factor = vector_neut(momentum, rank(volume))

Mean Reversion Factor
~~~~~~~~~~~~~~~~~~~~~

::

    # Stochastic Oscillator
    stoch_osc = (close - ts_min(low, 30)) / (ts_max(high, 30) - ts_min(low, 30))
    
    # Reversion signal: long at low, short at high
    factor = rank(1 - stoch_osc)  # rank already normalized, no need for zscore

Volatility Factor
~~~~~~~~~~~~~~~~~

::

    # Low Volatility Factor (Low Volatility Anomaly)
    returns = close / ts_delay(close, 1) - 1  # Calculate returns
    volatility = ts_std_dev(returns, 20)      # 20-day volatility
    factor = rank(-volatility)                # Low volatility ranking

Operators Reference
-------------------

For complete operator list and detailed documentation, refer to the sections above. All operators support chaining and can be flexibly combined to build complex alpha factors.
