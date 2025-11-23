快速開始
========

3 分鐘上手 Phandas。

基本步驟
--------

1. 導入和加載數據
~~~~~~~~~~~~~~~~~

::

    from phandas import *

    # 從 CSV 加載歷史 OHLCV 數據
    panel = Panel.from_csv('crypto_1d.csv')

    # 或從 API 獲取數據
    # panel = fetch_data(
    #     symbols=['ETH', 'SOL', 'ARB', 'OP', 'POL', 'SUI'],
    #     timeframe='1d',
    #     start_date='2023-01-01'
    # )

2. 提取基礎數據
~~~~~~~~~~~~~~~

::

    close = panel['close']
    open = panel['open']
    high = panel['high']
    low = panel['low']
    volume = panel['volume']

3. 構建因子
~~~~~~~~~~~

簡單動量因子::

    momentum_20 = (close / close.ts_delay(20)) - 1
    factor = rank(momentum_20)

或均值回歸因子::

    relative_low = (close - ts_min(low, 30)) / (ts_max(high, 30) - ts_min(low, 30))
    factor = rank(relative_low)

4. 因子中性化（可選）
~~~~~~~~~~~~~~~~~~~~~

移除因子與成交量的相關性::

    factor_neutral = vector_neut(factor, rank(-volume))

5. 回測
~~~~~~~

::

    result = backtest(
        entry_price_factor=open,
        strategy_factor=factor,
        transaction_cost=(0.0003, 0.0003)
    )

    # 查看結果
    result.print_summary()
    result.plot_equity()

輸出示例::

    ================================
    Backtest Summary
    ================================
    Total Return:        125.3%
    Annual Return:       35.2%
    Sharpe Ratio:        1.45
    Max Drawdown:        -18.2%
    ================================

更多例子
--------

參考 :doc:`examples/01_basic_factor` 和 :doc:`examples/02_combine_factors`。

