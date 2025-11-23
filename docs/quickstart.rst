快速開始
========

3 分鐘上手 Phandas。

基本步驟
--------

1. 導入和加載數據
~~~~~~~~~~~~~~~~~

::

    from phandas import Factor

    # 使用內置加密貨幣 1 日線數據
    f = Factor.load_crypto_1d()

2. 定義因子
~~~~~~~~~~~

動量因子（排名）::

    momentum = f.ts_rank(f.close)

均值回歸因子::

    deviation = f.close - f.ts_mean(f.close, 20)

3. 回測
~~~~~~~

::

    result = momentum.backtest()
    
    # 查看結果
    print(result.summary())
    print(result.stats)

輸出示例::

    ================================
    Factor Backtest Summary
    ================================
    Total Return:        125.3%
    Annual Return:       35.2%
    Sharpe Ratio:        1.45
    Max Drawdown:        -18.2%
    ================================

4. 查看詳細結果
~~~~~~~~~~~~~~~~

::

    # 日收益曲線
    print(result.daily_return)
    
    # 月度收益表
    print(result.monthly_return)
    
    # 每個資產的貢獻度
    print(result.contribution_by_symbol)

更多例子
--------

參考 :doc:`examples/01_basic_factor` 和 :doc:`examples/02_combine_factors`。

