快速開始
========

3 分鐘上手 Phandas。

完整工作流
---------

1. 獲取市場數據
~~~~~~~~~~~~~~

::

    from phandas import *

    # 獲取加密貨幣 OHLCV 數據
    panel = fetch_data(
        symbols=['ETH', 'SOL', 'ARB', 'OP', 'POL', 'SUI'],
        timeframe='1d',
        start_date='2023-01-01',
        sources=['binance', 'benchmark', 'calendar'],
    )

2. 提取基礎數據
~~~~~~~~~~~~~~~

::

    # OHLCV 數據自動解析為 Factor 對象
    close = panel['close']
    open = panel['open']
    high = panel['high']
    low = panel['low']
    volume = panel['volume']

3. 使用算子構建 Alpha 因子
~~~~~~~~~~~~~~~~~~~~~~~~~~

**時序算子** — 計算滾動統計量::

    # 20 日動量
    momentum = (close / ts_delay(close, 20)) - 1

**橫截面算子** — 每日標準化::

    # 排序：轉換為 0-1 排序分
    factor = rank(momentum)

**組合算子** — 多層加工::

    # 正規化並對成交量中性化
    factor = normalize(rank(momentum))
    factor = vector_neut(factor, rank(-volume))

4. 回測策略
~~~~~~~~~~~

::

    result = backtest(
        entry_price_factor=open, 
        strategy_factor=factor,
        transaction_cost=(0.0003, 0.0003)
    )

    result.plot_equity()

輸出示例::

    Total Return:    125.3%
    Annual Return:   35.2%
    Sharpe Ratio:    1.45
    Max Drawdown:    -18.2%

完整例子
--------

完整示例 - 動量 + 均值回歸::

    from phandas import *

    # 獲取數據
    panel = fetch_data(
        symbols=['ETH', 'SOL', 'ARB', 'OP', 'POL', 'SUI'],
        timeframe='1d',
        start_date='2023-01-01',
    )

    close = panel['close']
    volume = panel['volume']
    open = panel['open']

    # 構建因子：20 日動量 + 均值回歸組合
    momentum = (close / ts_delay(close, 20)) - 1
    reversion = 1 / ts_rank(close, 30)
    
    factor = rank(momentum) + 0.5 * rank(reversion)
    factor = vector_neut(factor, rank(-volume))

    # 回測
    result = backtest(
        entry_price_factor=open,
        strategy_factor=factor,
        transaction_cost=(0.0003, 0.0003)
    )

    result.print_summary()

下一步
-----

- 了解更多算子：參考 :doc:`guide/operators_guide`
- 實踐範例：參考 :doc:`examples/01_basic_factor` 和 :doc:`examples/02_combine_factors`

