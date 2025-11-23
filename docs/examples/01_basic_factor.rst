基本因子：動量
==============

本教學展示最簡單的量化因子。

概念
----

**動量因子** = 加密貨幣價格上升趨勢的強度

原理：20 日動量 = (當日收盤價 / 20 天前收盤價) - 1

例如：
- 如果現在 $100，20 天前 $80，動量 = 25%（上漲）
- 如果現在 $80，20 天前 $100，動量 = -20%（下跌）

代碼實現
--------

::

    from phandas import *
    
    # 1. 加載數據
    panel = Panel.from_csv('crypto_1d.csv')
    
    # 2. 提取收盤價
    close = panel['close']
    open = panel['open']
    
    # 3. 計算 20 日動量
    momentum_20 = (close / close.ts_delay(20)) - 1
    
    # 4. 排名轉換
    factor = rank(momentum_20)
    
    # 5. 回測
    result = backtest(
        entry_price_factor=open,
        strategy_factor=factor,
        transaction_cost=(0.0003, 0.0003)
    )
    
    # 查看結果
    result.print_summary()
    result.plot_equity()

輸出::

    ================================
    Backtest Summary
    ================================
    Total Return:          125.3%
    Annual Return:         35.2%
    Sharpe Ratio:          1.45
    Max Drawdown:          -18.2%
    ================================

核心概念
--------

- ``ts_delay(close, 20)``：將收盤價延後 20 期
- ``close / close.ts_delay(20)`` ：計算 20 日收益率
- ``rank()``：將因子排名轉換為 0-1 的值
  - 排名最低 = 0（看空）
  - 排名最高 = 1（看多）

改進版本：添加成交量中性化
----------------------------

移除動量因子與成交量的相關性::

    from phandas import *
    
    panel = Panel.from_csv('crypto_1d.csv')
    close = panel['close']
    open = panel['open']
    volume = panel['volume']
    
    # 計算動量
    momentum_20 = (close / close.ts_delay(20)) - 1
    momentum_factor = rank(momentum_20)
    
    # 移除成交量偏差
    volume_factor = rank(-volume)
    factor = vector_neut(momentum_factor, volume_factor)
    
    # 回測改進後的因子
    result = backtest(
        entry_price_factor=open,
        strategy_factor=factor,
        transaction_cost=(0.0003, 0.0003)
    )
    
    result.print_summary()

這個版本通常有更高的夏普比，因為移除了成交量噪聲的影響。

