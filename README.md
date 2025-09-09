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

Here is a complete example showing how to build a momentum strategy and neutralize it against trading volume.

```python
from phandas import fetch_data, load_factor, backtest, Factor
from phandas.operators import vector_neutralize

# 1. Download Data
raw_data = fetch_data(
    symbols=['BNB', 'ETH', 'SOL', 'MATIC', 'ARB', 'OP'],
    timeframe='1d',
    start_date='2023-01-01'
)

# 2. Load and Calculate Factors
close = load_factor(raw_data, 'close')
volume = load_factor(raw_data, 'volume')
open_price = load_factor(raw_data, 'open')

# Define a momentum factor function
def momentum_factor(close: Factor, delay: int) -> Factor:
    return (close / close.ts_delay(delay)) - 1

# Combine multiple momentum factors
momentum = (momentum_factor(close, 14) + 
            momentum_factor(close, 21) + 
            momentum_factor(close, 30))

# 3. Factor Neutralization
# Neutralize the momentum factor against volume to remove its influence
neutralized_momentum = vector_neutralize(momentum, -volume)

# 4. Backtest
result = backtest(
    price_factor=open_price, 
    strategy_factor=neutralized_momentum,
    transaction_cost=(0.0003, 0.0003) # Set separate buy/sell cost rates
)

# 5. Analyze Results
summary_output = result.summary()
result.plot_equity(summary_text=summary_output)
```

Perfect for crypto quantitative research and strategy development! 📈
