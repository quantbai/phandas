# 0. pip install phandas

from phandas import Panel, backtest, ts_mean, ts_delay, log, power, ts_sum, vector_neut, analyze_ic, backtest_layer, simulate_trade_replay

# 1. 載入數據
panel = Panel.from_csv('crypto_1d.csv')

# 2. 萃取因子
open = panel['open']
close = panel['close']
volume = panel['volume']

# 3. 計算 Skewness 因子（偏度因子）
n = 60

# 計算對數收益率
r_it = log(close) - ts_delay(log(close), window=1)

# 計算收益率與其均值之差
diff = r_it - ts_mean(r_it, window=n)

# 計算偏度分子與分母
numerator = ((n * (n - 1)) ** 1.5) * ts_sum(power(diff, 3), window=n)
denominator = ((n - 1) * (n - 2)) * power(ts_sum(power(diff, 2), window=n), 1.5)

# 偏度因子（成交量中性化）
skewness = numerator / denominator
skewness= vector_neut(skewness.rank(), -volume.rank()).normalize()
skewness.name = "Volume-Neutral Skewness Factor"

# 4. IC 分析
ic_results = analyze_ic(
    factor=skewness,
    price=close,
    periods=1,
    method='spearman'
)

# 5. 回測
bt_results = backtest(
    price_factor=open,
    strategy_factor=skewness,
    transaction_cost=(0.0003, 0.0003)
)

bt_results.plot_equity()

# 6. 多空分層回測
layer_bt_results = backtest_layer(
    price_factor=open,
    strategy_factor=skewness,
    transaction_cost=(0.0003, 0.0003),
    long_top_n=3,
    short_bottom_n=3,
)

layer_bt_results.plot_equity()

# 7. 交易回放模擬
simulate_trade_replay(
    price_factor=open,
    strategy_factor=skewness,
    n_days=60
)