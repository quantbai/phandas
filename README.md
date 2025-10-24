<div align="center">

<img src="https://raw.githubusercontent.com/quantbai/phandas/main/assets/PHANDAS2.png" alt="Phandas" width="500">

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
- **Backtesting**: Dollar-neutral portfolio construction with dynamic rebalancing
- **Performance Analytics**: Total Return, Annual Return, Sharpe Ratio, Max Drawdown, Turnover

### Installation

```bash
pip install phandas
```

### Quick Start

```python
from phandas import *

# Fetch market data
panel = fetch_data(
    symbols=['BNB', 'ETH', 'SOL', 'MATIC', 'ARB', 'OP'],
    timeframe='1d',
    start_date='2023-01-01'
)

# Extract factors
close = panel['close']
volume = panel['volume']
open_price = panel['open']

# Construct momentum factor
momentum_14 = (close / close.ts_delay(14)) - 1
momentum_21 = (close / close.ts_delay(21)) - 1
momentum_30 = (close / close.ts_delay(30)) - 1
factor = momentum_14 + momentum_21 + momentum_30

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
- **回測**：美元中性投組構建、動態調倉
- **績效分析**：年化收益、夏普比率、最大回撤、換手率

### 安裝

```bash
pip install phandas
```

### 快速開始

```python
from phandas import *

# 獲取市場資料
panel = fetch_data(
    symbols=['BNB', 'ETH', 'SOL', 'MATIC', 'ARB', 'OP'],
    timeframe='1d',
    start_date='2023-01-01'
)

# 提取因子
close = panel['close']
volume = panel['volume']
open_price = panel['open']

# 構建動量因子
momentum_14 = (close / close.ts_delay(14)) - 1
momentum_21 = (close / close.ts_delay(21)) - 1
momentum_30 = (close / close.ts_delay(30)) - 1
factor = momentum_14 + momentum_21 + momentum_30

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


