# 0. pip install phandas

from phandas import Panel, backtest, analyze_ic, rank, log, backtest_layer, simulate_trade_replay

# 1. 載入數據
panel = Panel.from_csv('crypto_1d.csv')

# 2. 萃取因子
open = panel['open']
close = panel['close']
volume = panel['volume']

# 3. 計算 Size 因子（小市值效應）
size_factor = -rank(volume)
size_factor.name = "Small Size Factor"

# 4. IC 分析
ic_results = analyze_ic(
    factor=size_factor,
    price=close,
    periods=1,
    method='spearman'
)

# 5. 回測
bt_results = backtest(
    price_factor=open,
    strategy_factor=size_factor,
    transaction_cost=(0.0003, 0.0003)
)

bt_results.plot_equity()

# 6. 多空分層回測 (Long Top 20%, Short Bottom 20%)
layer_bt_results = backtest_layer(
    price_factor=open,
    strategy_factor=size_factor,
    transaction_cost=(0.0003, 0.0003),
    long_top_n=3,
    short_bottom_n=3,
)

layer_bt_results.plot_equity()

# 7. 交易回放模擬
simulate_trade_replay(
    price_factor=open,
    strategy_factor=size_factor,
    n_days=60
)
