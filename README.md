# Phandas

**Phandas** is a quantitative analysis and backtesting toolkit for cryptocurrency markets, developed by **Phantom Management**.

## Features

- 🚀 **One-click data download**: Fetch crypto OHLCV data from major exchanges
- 📊 **Smart visualization**: Plot price charts (line/candlestick) with data gap detection  
- 🔍 **Data quality checks**: Comprehensive data validation and anomaly detection
- 🎯 **Special handling**: Automatic support for renamed tokens (e.g., MATIC→POL)
- ⚡ **Factor generation**: Alpha101-style factor expressions with built-in parser
- 📈 **Professional backtesting**: Complete backtesting engine with performance metrics

## Installation

```bash
pip install phandas
```

## Quick Start

```python
from phandas import fetch_data, load_factor, backtest, ts_corr

# 下載數據
raw_data = fetch_data(
    symbols=['BTC', 'ETH', 'SOL', 'MATIC', 'ARB', 'OP'],
    timeframe='1d',
    start_date='2023-01-01'
)

# 創建因子
open = load_factor(raw_data, 'open')
volume = load_factor(raw_data, 'volume')
price = load_factor(raw_data, 'open')

# Alpha#6: (-1 * ts_corr(open, volume, 10))
alpha006 = -ts_corr(open, volume, 10)

# 回測
result = backtest(price, alpha006, transaction_cost=0.001)
result.summary()
```

Perfect for crypto quantitative research and strategy development! 📈
