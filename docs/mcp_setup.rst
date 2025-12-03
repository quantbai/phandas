MCP Integration
===============

Phandas provides MCP (Model Context Protocol) integration, allowing AI IDEs (like Cursor) to directly call Phandas operators and backtesting functions.

What is MCP?
------------

MCP is a standard protocol that lets AI assistants access external tools and data sources. Through MCP, AI in Cursor can:

- Directly fetch cryptocurrency market data
- Browse all 50+ factor operators
- View function source code
- Execute factor backtests

Installation Steps
------------------

1. Install Phandas
~~~~~~~~~~~~~~~~~~

::

    pip install phandas

2. Configure Cursor
~~~~~~~~~~~~~~~~~~~

1. Open Cursor
2. Go to **Settings** → **Tools & MCP** → **New MCP Server**
3. Paste the following JSON configuration:

::

    {
      "mcpServers": {
        "phandas": {
          "command": "python",
          "args": ["-m", "phandas.mcp_server"]
        }
      }
    }

4. Save and restart Cursor

Verify Installation
~~~~~~~~~~~~~~~~~~~

After restarting Cursor, ask the AI in chat::

    List all available phandas operators

If the AI responds with a list of operators, MCP configuration is successful.

Available Tools
---------------

The MCP server provides 4 tool functions:

fetch_market_data
~~~~~~~~~~~~~~~~~

Fetch cryptocurrency OHLCV data.

**Parameters**:

- ``symbols``: List of trading pairs (e.g., ['BTC', 'ETH'])
- ``timeframe``: Time interval ('1d', '1h', '15m', etc.)
- ``limit``: Return last N data points (default: 5)
- ``start_date``: Start date (YYYY-MM-DD)
- ``end_date``: End date (YYYY-MM-DD)
- ``sources``: Data sources (default: ['binance'])

**Example**::

    Fetch the last 10 days of daily data for ETH and SOL

list_operators
~~~~~~~~~~~~~~

List all available factor operators.

Returns names, function signatures, and documentation for all operators.

**Example**::

    List all time series operators

read_source
~~~~~~~~~~~

View source code for any Phandas function or class.

**Parameters**:

- ``object_path``: Object path (e.g., 'phandas.operators.ts_mean')

**Example**::

    Show the source code for ts_mean function

execute_factor_backtest
~~~~~~~~~~~~~~~~~~~~~~~

Execute custom factor backtests.

**Parameters**:

- ``factor_code``: Python code to calculate factor
- ``symbols``: List of trading tokens (default: ['ETH','SOL','ARB','OP','POL','SUI'])
- ``start_date``: Start date (default: '2022-01-01')
- ``transaction_cost``: Transaction fee rate (default: 0.0003 = 0.03%)
- ``full_rebalance``: Whether to fully rebalance (default: False)

**Pre-defined variables**:

- ``close``, ``open``, ``high``, ``low``, ``volume``
- All Phandas operators (``ts_rank()``, ``ts_mean()``, ``log()``, ``rank()``, ``vector_neut()``, etc.)

**Note**: Code must assign result to variable named ``factor``

**Example**::

    Backtest a 20-day momentum factor neutralized against volume

Usage Examples
--------------

Common Use Cases
~~~~~~~~~~~~~~~~

**Query operators**
    Ask AI to list all available time series operators. AI will call ``list_operators()`` and filter relevant results.

**Fetch market data**
    Request historical data for specific tokens. AI will call ``fetch_market_data()`` and return OHLCV data.

**Execute factor backtest**
    Describe strategy logic. AI will auto-generate factor code and call ``execute_factor_backtest()`` for backtesting.

**View source code**
    Ask about implementation details of specific functions. AI will use ``read_source()`` to display source code.

Benefits
--------

Benefits of using MCP integration:

- **No coding required**: Describe strategies in natural language, AI auto-generates code
- **Fast iteration**: Quickly test different factor combinations
- **Learning tool**: View source code to learn operator implementations
- **Data exploration**: Easily fetch and analyze market data

Next Steps
----------

- Return to :doc:`installation` for basic installation
- See :doc:`quickstart` to learn writing strategies manually
- Refer to :doc:`guide/operators_guide` for all operators
