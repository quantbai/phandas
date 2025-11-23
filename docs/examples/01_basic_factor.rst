基本因子：動量
==============

本教學展示最簡單的量化因子。

概念
----

**動量因子** = 股票/加密貨幣價格上升趨勢的強度

原理：排名最近的收盤價，價格上升最快的資產排名最高。

代碼實現
--------

::

    from phandas import Factor
    
    # 1. 加載數據（自動從 CSV 或 API）
    f = Factor.load_crypto_1d()
    
    # 2. 定義因子（閉盤價排名）
    momentum = f.ts_rank(f.close)
    
    # 3. 回測（自動計算收益、夏普比、回撤）
    result = momentum.backtest()
    
    # 4. 查看結果
    print(result.summary())

輸出::

    ================================
    Factor Backtest Summary
    ================================
    Total Return:          125.3%
    Annual Return:         35.2%
    Sharpe Ratio:          1.45
    Max Drawdown:          -18.2%
    ================================

深入理解
--------

``ts_rank()`` 做什麼？

- 對每根 K 線的收盤價進行排名
- 返回 0-1 之間的值（最低價 = 0，最高價 = 1）
- 簡單的排名因子

改進版本：20 日動量
---------------------

使用 20 日動量（最近 20 天的上升趨勢）::

    from phandas import Factor
    
    f = Factor.load_crypto_1d()
    
    # 計算 20 日的對數收益
    log_return_20d = f.log(f.close) - f.ts_delay(f.log(f.close), 20)
    
    # 排名
    momentum_20d = f.ts_rank(log_return_20d)
    
    # 回測
    result = momentum_20d.backtest()
    print(result.summary())

這個版本通常會有更好的風險調整收益（Sharpe Ratio）。

