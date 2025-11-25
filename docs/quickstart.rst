快速開始
========

5 分鐘上手 Phandas - 從數據下載到策略回測。

完整工作流
---------

第一步：下載數據與存檔
~~~~~~~~~~~~~~~~~~~~

下載加密貨幣歷史數據並保存到本地::

    from phandas import *

    # 下載數據
    panel = fetch_data(
        symbols=['ETH', 'SOL', 'ARB', 'OP', 'POL', 'SUI'], 
        start_date='2022-01-01',
        sources=['binance']
    )

    # 保存到 CSV（避免重複下載）
    panel.to_csv('crypto_1d.csv')

.. note::
   使用 ``to_csv()`` 保存數據後，下次可以直接用 ``from_csv()`` 讀取，不需要重複下載。

第二步：載入數據
~~~~~~~~~~~~~~

從本地 CSV 文件讀取數據::

    # 載入數據
    panel = Panel.from_csv('crypto_1d.csv')

第三步：提取數據
~~~~~~~~~~~~~~

提取 OHLCV 數據，使用 ``.show()`` 查看因子值::

    close = panel['close']
    close.show()  # 查看收盤價數據

.. tip::
   使用 ``.show()`` 可以查看任何因子的具體數值，方便調試和驗證。

第四步：計算因子
~~~~~~~~~~~~~~

使用算子構建 Alpha 因子::

    # 提取數據
    high = panel['high']
    low = panel['low']
    volume = panel['volume']
    
    # 計算反轉因子
    n = 30
    relative_low = (close - ts_min(high, n)) / (ts_max(low, n) - ts_min(high, n))
    vol_ma = ts_mean(volume, n)
    vol_deviation = volume / vol_ma
    factor = relative_low * (1 + 0.5*(1 - vol_deviation))
    
    # 設置因子名稱
    factor.name = "Reversion Alpha"

第五步：回測策略
~~~~~~~~~~~~~~

將因子放入 ``strategy_factor`` 進行回測::

    bt_results = backtest(
        entry_price_factor=open, # 進場價格
        strategy_factor=factor, # 策略因子
        transaction_cost=(0.0003, 0.0003),  # 進出場手續費 0.03%
        full_rebalance=False,  # 是否每天全倉模式（預設關閉）
    )

.. important::
   - ``transaction_cost=(0.0003, 0.0003)`` 是最常見設定，代表進出場各 0.03% 手續費
   - ``full_rebalance=False`` 是預設值，設為 ``True`` 則每天全倉重新平衡

第六步：查看結果
~~~~~~~~~~~~~~

繪製權益曲線::

    bt_results.plot_equity()

完整代碼示例
~~~~~~~~~~~~

以下是完整的可執行代碼，整合上述所有步驟::

    from phandas import *

    # 1. 下載數據
    panel = fetch_data(
        symbols=['ETH', 'SOL', 'ARB', 'OP', 'POL', 'SUI'], 
        start_date='2022-01-01',
        sources=['binance']
    )

    # 2. 提取數據
    open = panel['open']
    close = panel['close']
    high = panel['high']
    low = panel['low']
    volume = panel['volume']

    # 3. 計算因子
    n = 30
    relative_low = (close - ts_min(high, n)) / (ts_max(low, n) - ts_min(high, n))
    vol_ma = ts_mean(volume, n)
    vol_deviation = volume / vol_ma
    factor = relative_low * (1 + 0.5*(1 - vol_deviation))

    # 4. 回測
    bt_results = backtest(
        entry_price_factor=open,
        strategy_factor=factor,
        transaction_cost=(0.0003, 0.0003),
    )
    bt_results.plot_equity()


下一步
-----

- 了解更多算子：參考 :doc:`guide/operators_guide`
