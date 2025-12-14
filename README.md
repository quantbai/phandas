<div align="center">

<img src="https://raw.githubusercontent.com/quantbai/phandas/main/assets/PHANDAS2.png" alt="Phandas" width="500">

[![en](https://img.shields.io/badge/lang-en-yellow.svg)](#english) &nbsp; [![zh-TW](https://img.shields.io/badge/lang-繁體中文-green.svg)](#繁體中文)

</div>

## English

A multi-factor quantitative trading framework for cryptocurrency markets.

### Overview

Phandas is a streamlined toolkit for alpha factor research and backtesting in cryptocurrency markets. Design factors with 60+ operators, test with dollar-neutral backtesting, and analyze with professional metrics.

### Try it now

[**Web Demo**](https://phandas.streamlit.app/) - Experience Phandas directly in your browser. No installation required.

### Key Features

- **Data Fetching**: Multi-source OHLCV data (Binance, OKX)
- **Factor Engine**: 60+ time-series and cross-sectional operators
- **Neutralization**: Vector projection & regression-based orthogonalization
- **Backtesting**: Dollar-neutral strategies with full/partial rebalancing
- **Performance Metrics**: Sharpe, Sortino, Calmar, Max Drawdown, VaR, PSR
- **Factor Analysis**: IC, IR, correlation, coverage, turnover
- **MCP Integration**: AI agents (Claude) can directly access Phandas

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
    sources=['binance'],
)

# Extract factors
close = panel['close']
volume = panel['volume']
open = panel['open']

# Construct momentum factor
momentum_20 = (close / close.ts_delay(20)) - 1

# Neutralize against volume
factor = vector_neut(rank(momentum_20), rank(-volume))

# Backtest strategy
result = backtest(
    entry_price_factor=open, 
    strategy_factor=factor,
    transaction_cost=(0.0003, 0.0003)
)

result.plot_equity()
```

### AI Integration via MCP

Use Phandas with AI IDEs (Cursor, Claude Desktop) directly—no coding required.

**Setup for Cursor (Recommended)**

1. `pip install phandas`
2. Open Cursor → Settings → Tools & MCP → **New MCP Server**
3. Paste the JSON config below, save and restart

```json
{
  "mcpServers": {
    "phandas": {
      "command": "python",
      "args": ["-m", "phandas.mcp_server"]
    }
  }
}
```

**Available Tools (4 Functions)**

- `fetch_market_data`: Get OHLCV data for symbols
- `list_operators`: Browse all 50+ factor operators
- `read_source`: View source code of any function
- `execute_factor_backtest`: Backtest custom factor expressions

---

## 繁體中文

一個專為加密貨幣市場設計的多因子量化交易框架。

### 概述

Phandas 是一個精簡的加密貨幣因子研究與回測工具。提供 60+ 運算子設計因子、美元中性回測、專業績效指標分析。

### 立即體驗

[**網頁演示**](https://phandas.streamlit.app/) - 直接在瀏覽器中體驗 Phandas，無需安裝。

### 核心功能

- **資料獲取**：多源 OHLCV 資料（Binance、OKX）
- **因子引擎**：60+ 時間序列與橫截面運算子
- **因子中性化**：向量投影與迴歸正交化
- **回測引擎**：美元中性策略、全/部分調倉
- **績效指標**：夏普比、Sortino、Calmar、最大回撤、VaR、PSR
- **因子分析**：IC、IR、相關性、覆蓋率、換手率
- **MCP 集成**：AI 代理（Claude）可直接調用 Phandas

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
    sources=['binance'],
)

# 提取因子
close = panel['close']
volume = panel['volume']
open = panel['open']

# 構建動量因子
momentum_20 = (close / close.ts_delay(20)) - 1

# 對成交量進行中性化
factor = vector_neut(rank(momentum_20), rank(-volume))

# 回測策略
result = backtest(
    entry_price_factor=open, 
    strategy_factor=factor,
    transaction_cost=(0.0003, 0.0003)
)

result.plot_equity()
```

### AI 集成（MCP 支援）

在 AI IDE（Cursor、Claude Desktop）中直接使用 Phandas—無需編碼。

**Cursor 設定（推薦）**

1. `pip install phandas`
2. 開啟 Cursor → Settings → Tools & MCP → **New MCP Server**
3. 貼上下方 JSON 配置，儲存並重啟

```json
{
  "mcpServers": {
    "phandas": {
      "command": "python",
      "args": ["-m", "phandas.mcp_server"]
    }
  }
}
```

**可用工具（4 個函數）**

- `fetch_market_data`: 獲取代幣 OHLCV 資料
- `list_operators`: 瀏覽 50+ 因子運算子
- `read_source`: 查看任何函數的源代碼
- `execute_factor_backtest`: 回測自訂因子表達式

---

## Documentation | 文檔

- [Full Docs](https://phandas.readthedocs.io/) - Complete API reference
- [Operators Guide](https://phandas.readthedocs.io/guide/operators_guide.html) - 50+ operators
- [MCP Setup](https://phandas.readthedocs.io/mcp_setup.html) - AI IDE integration

---

## Community & Support | 社群與支持

- **Discord**: [Join us - Phantom Management](https://discord.gg/TcPHTSGMdH)
- **GitHub Issues**: [Report bugs or request features](https://github.com/quantbai/phandas/issues)

## License

This project is licensed under the BSD 3-Clause License - see [LICENSE](LICENSE) file for details.


