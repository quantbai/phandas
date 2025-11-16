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
    symbols=['ETH', 'SOL', 'ARB', 'OP', 'POL', 'SUI'],
    timeframe='1d',
    start_date='2023-01-01',
    sources=['binance', 'benchmark', 'calendar'],
)

# Extract factors
close = panel['close']
volume = panel['volume']
open = panel['open']

# Construct momentum factor
momentum_20 = (close / close.ts_delay(20)) - 1

# Neutralize against volume
neutralized_factor = vector_neut(rank(momentum_20), rank(-volume))

# Backtest strategy
result = backtest(
    price_factor=open, 
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
    symbols=['ETH', 'SOL', 'ARB', 'OP', 'POL', 'SUI'],
    timeframe='1d',
    start_date='2023-01-01',
    sources=['binance', 'benchmark', 'calendar'],
)

# 提取因子
close = panel['close']
volume = panel['volume']
open = panel['open']

# 構建動量因子
momentum_20 = (close / close.ts_delay(20)) - 1

# 對成交量進行中性化
neutralized_factor = vector_neut(rank(momentum_20), rank(-volume))

# 回測策略
result = backtest(
    price_factor=open, 
    strategy_factor=neutralized_factor,
    transaction_cost=(0.0003, 0.0003)
)

result.plot_equity()
```

---

由 Phantom Management 開發。

## Community & Support | 社群與支持

- **Discord**: [Join our community](https://discord.gg/TcPHTSGMdH)
- **GitHub Issues**: [Report bugs or request features](https://github.com/quantbai/phandas/issues)

## License

This project is licensed under the BSD 3-Clause License - see [LICENSE](LICENSE) file for details.


