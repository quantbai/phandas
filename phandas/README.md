<div align="center">

<img src="https://raw.githubusercontent.com/quantbai/phandas/main/assets/PHANDAS.png" alt="Phandas" width="500">

</div>

---

<div align="center">

[![en](https://img.shields.io/badge/lang-en-yellow.svg)](#english)
[![zh-TW](https://img.shields.io/badge/lang-繁體中文-green.svg)](#繁體中文)

</div>


## English

A multi-factor quantitative trading framework for cryptocurrency markets.

### Overview

Phandas is a quantitative analysis framework designed for systematic portfolio construction and risk management. It provides high-performance data structures and financial analysis tools for factor investing and statistical arbitrage strategies.

### Key Features

- **Data management**: Automated OHLCV data fetching with validation and quality checks
- **Factor operations**: Extensive library of time-series and cross-sectional operators
- **Neutralization**: Vector projection and regression-based factor neutralization
- **Backtesting**: Performance evaluation with transaction cost modeling
- **Visualization**: Price charts and equity curves with automatic gap detection

### Installation

```bash
pip install phandas
```

### Quick Start

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

---

## 繁體中文

一個專為加密貨幣市場設計的多因子量化交易框架。

### 概述

Phandas 是一個為系統化投資組合構建與風險管理而設計的量化分析框架。它為因子投資與統計套利策略提供高效能的資料結構與金融分析工具。

### 核心功能

- **資料管理**：自動化 OHLCV 資料獲取，包含驗證與品質檢查
- **因子運算**：豐富的時間序列與橫截面運算子庫
- **中性化**：基於向量投影與迴歸的因子中性化
- **回測**：包含交易成本建模的績效評估
- **視覺化**：價格圖表與權益曲線，自動檢測資料缺口

### 安裝

```bash
pip install phandas
```

### 快速開始

```python
from phandas import fetch_data, load_factor, backtest
from phandas.operators import vector_neut

# 獲取市場資料
data = fetch_data(
    symbols=['BNB', 'ETH', 'SOL', 'MATIC', 'ARB', 'OP'],
    timeframe='1d',
    start_date='2023-01-01'
)

# 載入因子
close = load_factor(data, 'close')
volume = load_factor(data, 'volume')
open_price = load_factor(data, 'open')

# 構建動量因子
def momentum(close, delay):
    return (close / close.ts_delay(delay)) - 1

factor = (momentum(close, 14) + 
          momentum(close, 21) + 
          momentum(close, 30))

# 對成交量進行中性化
neutralized_factor = vector_neut(factor, -volume)

# 回測策略
result = backtest(
    price_factor=open_price, 
    strategy_factor=neutralized_factor,
    transaction_cost=(0.0003, 0.0003)
)

result.plot_equity()
```

---

由 Phantom Management 開發。
