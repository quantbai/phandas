算子詳細指南
============

Phandas 提供 **50+ 算子** 用於因子構建。分為四大類：橫截面、時序、中性化、數學運算。

.. contents::
   :local:
   :depth: 2

橫截面算子
-----------

每個時間截面（日期）獨立計算，用於標準化和排序。

排序與排名
~~~~~~~~~~

**rank()** — 百分位排序 (0-1)
    每日內將因子排序，輸出 0-1 的排名。NaN 時返回 NaN。

    ::

        factor_ranked = rank(factor)

**normalize()** — 去中心化
    每日去掉平均值。可選除以標準差和裁剪。

    ::

        factor_norm = normalize(factor)
        factor_norm_std = normalize(factor, use_std=True)  # 標準分數

**zscore()** — 標準化 (μ=0, σ=1)
    等同於 ``normalize(use_std=True)``。

    ::

        factor_z = zscore(factor)

聚集統計
~~~~~~~~

**mean()** — 橫截面平均
    計算每日平均值（常用於診斷）。

    ::

        mean_factor = mean(factor)

**median()** — 橫截面中位數
    計算每日中位數。

    ::

        median_factor = median(factor)

轉換與縮放
~~~~~~~~~~

**scale()** — 按絕對值縮放
    使因子絕對值之和等於指定值（默認 1.0）。

    ::

        factor_scaled = scale(factor, scale=1.0)
        # 支持長短邊分開縮放
        factor_scaled = scale(factor, long_scale=0.5, short_scale=-0.5)

**quantile()** — 分位數變換
    排序 → 正態/均勻/柯西分佈 PPF，支持縮放。

    ::

        factor_normal = quantile(factor, driver="gaussian", sigma=1.0)
        factor_uniform = quantile(factor, driver="uniform")

**spread()** — 二元信號
    取前 pct% 置 +0.5，後 pct% 置 -0.5，其餘 0。

    ::

        signal = spread(factor, pct=0.3)  # 多空前 30%

**signal()** — 美元中性信號
    去中心化，按絕對值縮放使長邊和 = 0.5，短邊和 = -0.5。

    ::

        dn_signal = signal(factor)

時序算子
--------

在每個資產的時間序列上計算，用於提取動量、均值回歸、波動率等。

延遲與差分
~~~~~~~~~~

**ts_delay(factor, window)** — 滯後
    將數據向後平移 window 期。

    ::

        prev_close = ts_delay(close, 1)

**ts_delta(factor, window)** — 變化
    當期與 window 期前的差值：x - x_{t-window}。

    ::

        returns = ts_delta(close, 1)  # 日收益

基礎統計
~~~~~~~~

**ts_mean(factor, window)** — 滾動平均
    計算 window 期內的平均值（要求完整窗口）。

    ::

        ma_20 = ts_mean(close, 20)

**ts_median(factor, window)** — 滾動中位數
    計算 window 期內的中位數。

    ::

        median_20 = ts_median(close, 20)

**ts_sum(factor, window)** — 滾動求和
    計算 window 期內的累計和。

    ::

        volume_sum_10 = ts_sum(volume, 10)

**ts_product(factor, window)** — 滾動乘積
    計算 window 期內的累計乘積。

    ::

        cumprod_5 = ts_product(close, 5)

**ts_std_dev(factor, window)** — 滾動標準差
    計算 window 期內的標準差（波動率）。

    ::

        volatility_20 = ts_std_dev(close, 20)

排名與極值
~~~~~~~~~~

**ts_rank(factor, window)** — 滾動排序
    在 window 期內計算百分位排序。

    ::

        rank_10 = ts_rank(close, 10)

**ts_max(factor, window)** — 滾動最大值
    計算 window 期內的最大值。

    ::

        highest_20 = ts_max(high, 20)

**ts_min(factor, window)** — 滾動最小值
    計算 window 期內的最小值。

    ::

        lowest_20 = ts_min(low, 20)

**ts_arg_max(factor, window)** — 最大值距今期數
    返回 0-1 的相對索引（0=最早，window-1=最新）。

    ::

        periods_since_max = ts_arg_max(close, 20)

**ts_arg_min(factor, window)** — 最小值距今期數
    返回 0-1 的相對索引。

    ::

        periods_since_min = ts_arg_min(close, 20)

高階統計
~~~~~~~~

**ts_skewness(factor, window)** — 滾動偏度
    計算 window 期內的樣本偏度（含 Bessel 修正）。

    ::

        skew_20 = ts_skewness(close, 20)

**ts_kurtosis(factor, window)** — 滾動峰度
    計算 window 期內的超峰度（excess kurtosis）。

    ::

        kurt_20 = ts_kurtosis(returns, 20)

標準化
~~~~~~

**ts_zscore(factor, window)** — 滾動標準分數
    計算 (x - mean) / std within window。

    ::

        zscore_20 = ts_zscore(close, 20)

**ts_scale(factor, window, constant)** — 滾動 Min-Max 縮放
    計算 (x - min) / (max - min) + constant。

    ::

        scaled_20 = ts_scale(close, 20)

**ts_quantile(factor, window, driver)** — 滾動分位數變換
    在 window 內排序 → 正態/均勻/柯西 PPF。

    ::

        ts_q_normal = ts_quantile(close, 20, driver="gaussian")

衰減加權
~~~~~~~~

**ts_decay_linear(factor, window, dense)** — 線性衰減加權
    最近數據權重更高，線性遞減。

    ::

        factor_decay_lin = ts_decay_linear(factor, 20)

**ts_decay_exp_window(factor, window, factor=0.9, nan)** — 指數衰減加權
    最近數據權重指數級更高。

    ::

        factor_decay_exp = ts_decay_exp_window(factor, 20, factor=0.95)

相關與回歸
~~~~~~~~~~

**ts_corr(factor1, factor2, window)** — 滾動皮爾遜相關
    計算兩個因子 window 期內的相關係數。

    ::

        corr_momentum_volume = ts_corr(momentum, volume, 20)

**ts_covariance(factor1, factor2, window)** — 滾動協方差
    計算兩個因子 window 期內的協方差。

    ::

        cov_close_volume = ts_covariance(close, volume, 20)

**ts_regression(y, x, window, lag, rettype)** — 滾動 OLS 迴歸
    計算 window 內 y = α + β·x 的係數。

    - rettype=0: 殘差（默認）
    - rettype=1: α（截距）
    - rettype=2: β（斜率）
    - rettype=3: 預測值
    - rettype=6: R²

    ::

        residual = ts_regression(close, open, 20, rettype=0)
        beta = ts_regression(close, momentum, 20, rettype=2)

其他
~~~~

**ts_count_nans(factor, window)** — 計數 NaN
    計算 window 內的 NaN 個數。

    ::

        nan_count = ts_count_nans(factor, 10)

**ts_backfill(factor, window, k)** — NaN 回填
    用 window 內第 k 個最近非 NaN 值填充 NaN。

    ::

        factor_filled = ts_backfill(factor, 20, k=1)

**ts_step(start)** — 時間計數器
    按資產生成遞增序列：1, 2, 3, ...

    ::

        time_counter = ts_step(1)

**ts_av_diff(factor, window)** — 平均偏差
    計算 x - ts_mean(x, window)。

    ::

        deviation = ts_av_diff(close, 20)

中性化算子
----------

移除因子與特定變量的線性相關性。

向量中性化
~~~~~~~~~~

**vector_neut(x, y)** — 向量投影正交化
    移除 x 對 y 的線性投影，保留正交成分。使用點積計算。

    ::

        # 移除動量與成交量的相關性
        momentum_neutral = vector_neut(momentum, rank(-volume))

迴歸中性化
~~~~~~~~~~

**regression_neut(y, x)** — OLS 殘差中性化
    通過 OLS 回歸移除 y 對 x（可多個）的線性依賴。

    ::

        # 同時中性化對開盤價和成交量
        factor_neutral = regression_neut(
            factor, 
            [open, volume]
        )

數學算子
--------

基本數學運算和函數變換。

初等函數
~~~~~~~~

**log(factor, base)** — 對數變換
    自然對數（base=None）或指定底數。x ≤ 0 → NaN。

    ::

        log_close = log(close)
        log2_volume = log(volume, base=2)

**ln(factor)** — 自然對數
    等同於 ``log(factor)``。

    ::

        ln_close = ln(close)

**sqrt(factor)** — 平方根
    x < 0 → NaN。

    ::

        sqrt_volume = sqrt(volume)

**s_log_1p(factor)** — 符號保留對數
    sign(x)·ln(1+|x|)，保留符號，處理零值。

    ::

        sl_returns = s_log_1p(returns)

冪與根式
~~~~~~~~

**power(base, exponent)** — 冪函數
    計算 base^exponent，非法值 → NaN。

    ::

        factor_sq = power(factor, 2)

**signed_power(base, exponent)** — 符號保留冪
    sign(x) 乘以 |x|^exponent，保留符號。

    ::

        factor_pow = signed_power(factor, 0.5)

符號函數
~~~~~~~~

**sign(factor)** — 符號函數
    返回 -1/0/+1。

    ::

        sign_factor = sign(factor)

**inverse(factor)** — 倒數
    計算 1/x，x=0 → NaN。

    ::

        inv_factor = inverse(factor)

比較與條件
~~~~~~~~~~

**maximum(factor1, factor2)** — 元素級最大值
    逐元素取兩個因子的最大值。

    ::

        max_factor = maximum(factor1, factor2)

**minimum(factor1, factor2)** — 元素級最小值
    逐元素取兩個因子的最小值。

    ::

        min_factor = minimum(factor1, factor2)

**where(condition, x, y)** — 條件選擇
    condition=True 時選擇 x，否則選擇 y。

    ::

        filtered = where(factor > 0, factor, 0)

算術運算
~~~~~~~~

支持直接使用 Python 運算符或函數：

- **add(a, b)** 或 ``a + b`` — 加法
- **subtract(a, b)** 或 ``a - b`` — 減法
- **multiply(a, b)** 或 ``a * b`` — 乘法
- **divide(a, b)** 或 ``a / b`` — 除法（除 0 → NaN）
- **power(a, b)** 或 ``a ** b`` — 冪

::

    factor = momentum + 0.5 * reversion
    ratio = close / open
    scaled = factor / ts_mean(factor, 20)

常見組合模式
~~~~~~~~~~~~

動量因子
~~~~~~~~

::

    # 簡單動量
    momentum = (close / ts_delay(close, 20)) - 1
    factor = rank(momentum)

    # 多周期動量
    mom_short = rank((close / ts_delay(close, 5)) - 1)
    mom_long = rank((close / ts_delay(close, 20)) - 1)
    momentum = 0.7 * mom_short + 0.3 * mom_long
    factor = vector_neut(momentum, rank(-volume))

均值回歸因子
~~~~~~~~~~~~

::

    # 相對位置
    relative_position = (close - ts_min(low, 30)) / (ts_max(high, 30) - ts_min(low, 30))
    reversion = rank(1 - relative_position)
    factor = zscore(reversion)

波動率因子
~~~~~~~~~~

::

    # 波動率反轉
    volatility = ts_std_dev(returns, 20)
    vol_factor = rank(1 / volatility)
    factor = scale(vol_factor)

組合因子
~~~~~~~~

::

    # 多因子組合
    momentum = rank((close / ts_delay(close, 20)) - 1)
    reversion = rank(1 / ts_rank(close, 30))
    volume_factor = rank(volume)

    factor = (momentum * 0.5 + reversion * 0.3 + volume_factor * 0.2)
    factor = normalize(factor)
    factor = vector_neut(factor, rank(-volume))

最佳實踐
--------

1. **特徵工程** — 先用時序算子萃取動量/波動率等統計量
2. **標準化** — 用橫截面算子進行排序和去中心化
3. **中性化** — 移除不需要的因子暴露（如成交量偏差）
4. **驗證** — 通過回測檢驗因子質量

完整例子見文檔中的範例部分。

