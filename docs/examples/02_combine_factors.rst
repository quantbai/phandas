組合因子
========

組合多個因子以降低風險。

概念
----

單個因子容易過擬合。組合多個獨立因子提高穩定性。

例子：組合動量 + 均值回歸
---------------------------

::

    from phandas import Factor
    
    f = Factor.load_crypto_1d()
    
    # 因子 1：動量（價格上升趨勢）
    momentum = f.ts_rank(f.close)
    
    # 因子 2：均值回歸（價格偏離均線）
    mean = f.ts_mean(f.close, 20)
    deviation = (f.close - mean) / mean
    mean_reversion = f.ts_rank(deviation)
    
    # 中性化：移除動量和均值回歸的相關性
    factor = f.vector_neut(momentum, mean_reversion)
    
    # 回測組合因子
    result = factor.backtest()
    print(result.summary())

``vector_neut()`` 做什麼？

- 移除兩個因子之間的相關性
- 例：momentum 獨立於 mean_reversion
- 提高因子的多樣性

三因子組合
----------

::

    from phandas import Factor
    
    f = Factor.load_crypto_1d()
    
    # 因子 1：動量
    momentum = f.ts_rank(f.close)
    
    # 因子 2：成交量動量
    volume_momentum = f.ts_rank(f.volume)
    
    # 因子 3：波動率
    volatility = f.ts_std(f.close, 20)
    
    # 組合（加權平均）
    factor = (momentum + volume_momentum - f.ts_rank(volatility)) / 3
    
    result = factor.backtest()
    print(result.summary())

最佳實踐
--------

組合因子時考慮：

1. **因子獨立性**：因子之間相關性越低越好
2. **權重分配**：平均權重通常是好的起點
3. **回測檢驗**：組合後的夏普比應該比單個因子更高

