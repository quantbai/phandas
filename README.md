# Phandas

[English](README.md) | [繁體中文](README_zh-TW.md)

Hedge fund-grade multi-factor quantitative trading framework for cryptocurrency markets.

## Overview

Phandas is a professional toolkit for multi-factor quantitative trading, designed for systematic portfolio construction and risk management. Built with a pandas-like API, it provides efficient implementations of cross-sectional and time-series operations essential for factor investing and statistical arbitrage strategies.

## Key Features

- **Data management**: Automated OHLCV data fetching with validation and quality checks
- **Factor operations**: Extensive library of time-series and cross-sectional operators
- **Neutralization**: Vector projection and regression-based factor neutralization
- **Backtesting**: Performance evaluation with transaction cost modeling
- **Visualization**: Price charts and equity curves with automatic gap detection

## Installation

```bash
pip install phandas
```

## Quick Start

```python
from phandas import fetch_data, load_factor, backtest
from phandas.operators import vector_neut

# Fetch market data
data = fetch_data(
    symbols=['BNB', 'ETH', 'SOL', 'MATIC', 'ARB', 'OP'],
    timeframe='1d',
    start_date='2023-01-01'
)

# Load factors
close = load_factor(data, 'close')
volume = load_factor(data, 'volume')
open_price = load_factor(data, 'open')

# Construct momentum factor
def momentum(close, delay):
    return (close / close.ts_delay(delay)) - 1

factor = (momentum(close, 14) + 
          momentum(close, 21) + 
          momentum(close, 30))

# Neutralize against volume
neutralized_factor = vector_neut(factor, -volume)

# Backtest strategy
result = backtest(
    price_factor=open_price, 
    strategy_factor=neutralized_factor,
    transaction_cost=(0.0003, 0.0003)
)

result.plot_equity()
```

---

Developed by Phantom Management.
