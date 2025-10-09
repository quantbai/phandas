# 0. pip install phandas

from phandas import Panel, backtest, analyze_ic, vector_neut, rank

# 1. 載入數據
panel = Panel.from_csv('crypto_1d.csv')

# 2. 萃取因子
open = panel['open']
close = panel['close']
volume = panel['volume']

# 3. 計算 Momentum 因子
def momentum_factor(close, delay: int):
    return (close / close.ts_delay(delay)) - 1

mom_20d = momentum_factor(close, 20)
mom_20d = vector_neut(rank(mom_20d), rank(volume))
mom_20d.name = "20-Day Momentum (Volume-Neutral)"

# 4. IC 分析
ic_results = analyze_ic(
    factor=mom_20d,
    price=close,
    periods=1,
    method='spearman'
)

# 5. 回測
bt_results = backtest(
    price_factor=open,
    strategy_factor=mom_20d,
    transaction_cost=(0.0003, 0.0003)
)

bt_results.plot_equity()