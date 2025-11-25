算子 MCP
=========

Phandas 提供 MCP (Model Context Protocol) 集成，讓 AI IDE（如 Cursor）可以直接調用 Phandas 的算子和回測功能。

什麼是 MCP？
-----------

MCP 是一個標準協議，讓 AI 助手可以訪問外部工具和數據源。通過 MCP，Cursor 中的 AI 可以：

- 直接獲取加密貨幣市場數據
- 瀏覽所有 50+ 因子算子
- 查看函數源代碼
- 執行因子回測

安裝步驟
--------

1. 安裝 Phandas
~~~~~~~~~~~~~~~

::

    pip install phandas

2. 配置 Cursor
~~~~~~~~~~~~~~

1. 打開 Cursor
2. 進入 **Settings** → **Tools & MCP** → **New MCP Server**
3. 貼上以下 JSON 配置：

::

    {
      "mcpServers": {
        "phandas": {
          "command": "python",
          "args": ["-m", "phandas.mcp_server"]
        }
      }
    }

4. 保存並重啟 Cursor

驗證安裝
~~~~~~~~

重啟 Cursor 後，在聊天中詢問 AI::

    列出 phandas 所有可用的算子

如果 AI 能夠回應並列出算子，表示 MCP 配置成功。

可用工具
--------

MCP 服務器提供 4 個工具函數：

fetch_market_data
~~~~~~~~~~~~~~~~~

獲取加密貨幣 OHLCV 數據。

**參數**：

- ``symbols``: 交易對列表（例如 ['BTC', 'ETH']）
- ``timeframe``: 時間間隔（'1d', '1h', '15m' 等）
- ``limit``: 返回最近 N 條數據（預設：5）
- ``start_date``: 開始日期（YYYY-MM-DD）
- ``end_date``: 結束日期（YYYY-MM-DD）
- ``sources``: 數據源（預設：['binance']）

**示例**::

    獲取 ETH 和 SOL 最近 10 天的日線數據

list_operators
~~~~~~~~~~~~~~

列出所有可用的因子算子。

返回所有算子的名稱、函數簽名和說明文檔。

**示例**::

    列出所有時間序列算子

read_source
~~~~~~~~~~~

查看任何 Phandas 函數或類的源代碼。

**參數**：

- ``object_path``: 對象路徑（例如 'phandas.operators.ts_mean'）

**示例**::

    查看 ts_mean 函數的源代碼

execute_factor_backtest
~~~~~~~~~~~~~~~~~~~~~~~

執行自定義因子回測。

**參數**：

- ``factor_code``: 計算因子的 Python 代碼
- ``symbols``: 交易代幣列表（預設：['ETH','SOL','ARB','OP','POL','SUI']）
- ``start_date``: 開始日期（預設：'2022-01-01'）
- ``transaction_cost``: 手續費率（預設：0.0003 = 0.03%）
- ``full_rebalance``: 是否全倉重新平衡（預設：False）

**預定義變量**：

- ``close``, ``open``, ``high``, ``low``, ``volume``
- 所有 Phandas 算子（``ts_rank()``, ``ts_mean()``, ``log()``, ``rank()``, ``vector_neut()`` 等）

**注意**：代碼必須將結果賦值給變量 ``factor``

**示例**::

    幫我回測一個 20 日動量因子，對成交量中性化

使用示例
--------

常見使用場景
~~~~~~~~~~~~

**查詢算子**
    詢問 AI 列出所有可用的時間序列算子，AI 會調用 ``list_operators()`` 並過濾相關結果。

**獲取市場數據**
    請求獲取特定代幣的歷史數據，AI 會調用 ``fetch_market_data()`` 並返回 OHLCV 數據。

**執行因子回測**
    描述策略邏輯，AI 會自動生成因子代碼並調用 ``execute_factor_backtest()`` 進行回測。

**查看源代碼**
    詢問特定函數的實現細節，AI 會使用 ``read_source()`` 顯示源代碼。

優勢
----

使用 MCP 集成的優勢：

- **無需編碼**：用自然語言描述策略，AI 自動生成代碼
- **快速迭代**：快速測試不同的因子組合
- **學習工具**：查看源代碼學習算子實現
- **數據探索**：輕鬆獲取和分析市場數據

下一步
------

- 返回 :doc:`installation` 查看基本安裝
- 查看 :doc:`quickstart` 學習手動編寫策略
- 參考 :doc:`guide/operators_guide` 了解所有算子
