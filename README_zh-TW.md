# Phandas

[English](README.md) | [繁體中文](README_zh-TW.md)

對沖基金級多因子量化交易框架，專為加密貨幣市場打造。

## 概述

Phandas 是專業的多因子量化交易工具，專為系統化投資組合構建與風險管理設計。採用類 pandas 的 API，提供因子投資與統計套利策略所需的橫截面與時間序列運算高效實現。

## 核心功能

- **資料管理**：自動化 OHLCV 資料獲取，包含驗證與品質檢查
- **因子運算**：豐富的時間序列與橫截面運算子庫
- **中性化**：基於向量投影與迴歸的因子中性化
- **回測**：包含交易成本建模的績效評估
- **視覺化**：價格圖表與權益曲線，自動檢測資料缺口

## 安裝

```bash
pip install phandas
```

## 快速開始

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

