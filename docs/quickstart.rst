Quick Start
===========

Get started with Phandas in 5 minutes - from data download to strategy backtesting.

Complete Workflow
-----------------

Step 1: Download and Save Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download cryptocurrency historical data and save locally::

    from phandas import *

    # Download data
    panel = fetch_data(
        symbols=['ETH', 'SOL', 'ARB', 'OP', 'POL', 'SUI'], 
        start_date='2022-01-01',
        sources=['binance']
    )

    # Save to CSV (avoid repeated downloads)
    panel.to_csv('crypto_1d.csv')

.. note::
   After saving data with ``to_csv()``, you can load it directly with ``from_csv()`` next time without re-downloading.

Step 2: Load Data
~~~~~~~~~~~~~~~~~

Read data from local CSV file::

    # Load data
    panel = Panel.from_csv('crypto_1d.csv')

Step 3: Extract Data
~~~~~~~~~~~~~~~~~~~~

Extract OHLCV data, use ``.show()`` to view factor values::

    close = panel['close']
    close.show()  # View close price data

.. tip::
   Use ``.show()`` to view any factor's actual values for debugging and verification.

Step 4: Calculate Factor
~~~~~~~~~~~~~~~~~~~~~~~~

Build alpha factors using operators::

    # Extract data
    high = panel['high']
    low = panel['low']
    volume = panel['volume']
    
    # Calculate reversion factor
    n = 30
    relative_low = (close - ts_min(high, n)) / (ts_max(low, n) - ts_min(high, n))
    vol_ma = ts_mean(volume, n)
    vol_deviation = volume / vol_ma
    factor = relative_low * (1 + 0.5*(1 - vol_deviation))
    
    # Set factor name
    factor.name = "Reversion Alpha"

Step 5: Backtest Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~

Pass the factor to ``backtest`` for backtesting::

    bt_results = backtest(
        entry_price_factor=open,  # Entry price
        strategy_factor=factor,   # Strategy factor
        transaction_cost=(0.0003, 0.0003),  # Entry/exit fee 0.03%
        full_rebalance=False,  # Full rebalance mode (default off)
    )

.. important::
   - ``transaction_cost=(0.0003, 0.0003)`` is the most common setting, representing 0.03% fee for both entry and exit
   - ``full_rebalance=False`` is the default; set to ``True`` for daily full portfolio rebalancing

Step 6: View Results
~~~~~~~~~~~~~~~~~~~~

Plot equity curve::

    bt_results.plot_equity()

Complete Code Example
~~~~~~~~~~~~~~~~~~~~~

Here's the complete executable code combining all steps above::

    from phandas import *

    # 1. Download data
    panel = fetch_data(
        symbols=['ETH', 'SOL', 'ARB', 'OP', 'POL', 'SUI'], 
        start_date='2022-01-01',
        sources=['binance']
    )

    # 2. Extract data
    open = panel['open']
    close = panel['close']
    high = panel['high']
    low = panel['low']
    volume = panel['volume']

    # 3. Calculate factor
    n = 30
    relative_low = (close - ts_min(high, n)) / (ts_max(low, n) - ts_min(high, n))
    vol_ma = ts_mean(volume, n)
    vol_deviation = volume / vol_ma
    factor = relative_low * (1 + 0.5*(1 - vol_deviation))

    # 4. Backtest
    bt_results = backtest(
        entry_price_factor=open,
        strategy_factor=factor,
        transaction_cost=(0.0003, 0.0003),
    )
    bt_results.plot_equity()


Next Steps
----------

- Learn more operators: see :doc:`guide/operators_guide`
