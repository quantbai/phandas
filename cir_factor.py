from phandas import Panel, backtest, analyze_ic, backtest_layer, simulate_trade_replay
from phandas.operators import log, ts_delay, vector_neut, rank, ts_regression

# 1. 載入數據
panel = Panel.from_csv('crypto_1d.csv')

# 2. 提取因子
close = panel['close']
volume = panel['volume']

# ============================================================
# Alpha Factor: G. Mean Reversion (Ornstein-Uhlenbeck)
#
# 模型定性：
# 這是一個典型的「均值回歸」因子，其理論基礎為 Ornstein-Uhlenbeck (OU)
# 隨機過程。它旨在捕捉資產短期報酬率回歸至其長期均衡水平的現象。
#
# 核心理念：
# 如同一根被拉長的橡皮筋會縮回原狀，資產的報酬率在短期內若大幅偏離
# 其歷史均值，也存在一股力量將其拉回。本策略旨在系統性地利用此現象，
# 在報酬率過低時買入，過高時賣出。
#
# 數學推導與實作：
# 1. 理論模型 (連續時間):
#    報酬率 r(t) 的動態遵循 OU 過程：dr(t) = a(b - r(t))dt + σ'dW(t)，
#    其中 'a' 為回歸速度，'b' 為長期均值。
#
# 2. 實作模型 (離散時間):
#    上述隨機微分方程 (SDE) 可離散化為一個 AR(1) 自迴歸模型：
#    r(t) = α + β*r(t-1) + ε(t)。
#
# 3. 參數估計:
#    透過在一個滾動的時間窗口 (n天) 上進行線性回歸，我們可以從歷史數據
#    中估計出 α 和 β。
#
# 4. 因子構建:
#    根據估計出的 α 和 β，可以推導出模型的長期均值 b = α / (1 - β)。
#    最終的交易訊號 (因子值) 被定義為預期回歸的幅度：
#    Factor = long_term_mean - current_return
# ============================================================

# 3. 計算因子
# 共用變數
# 計算連續複利報酬率 r
# 這裡運用了對數的巧妙特性：
# r(t) ≈ log(S(t)) - log(S(t-1)) = log(S(t) / S(t-1))
n = 30
r = log(close) - ts_delay(log(close), 1)
r_lag = ts_delay(r, 1)

# 滾動回歸估計參數
# r_t = alpha + beta * r_{t-1}
alpha = ts_regression(r, r_lag, window=n, rettype=1)
beta = ts_regression(r, r_lag, window=n, rettype=2)

# 計算均值回歸參數
# 回歸速度 a = 1 - beta
# 長期均值 b = alpha / (1 - beta)
reversion_speed = 1 - beta
long_term_mean = alpha / reversion_speed

# 因子： -(當前觀測 - 長期均值)
mean_reversion_factor = long_term_mean - r
mean_reversion_factor_neutralized = vector_neut(rank(mean_reversion_factor), rank(volume))
mean_reversion_factor_neutralized.name = "G_MeanReversion_OU"

# 4. IC 分析
ic_results = analyze_ic(
    factor=mean_reversion_factor_neutralized,
    price=close,
    periods=1,
    method='spearman'
)

# 5. 回測
bt_results = backtest(
    price_factor=panel['open'],
    strategy_factor=mean_reversion_factor_neutralized,
    transaction_cost=(0.0003, 0.0003)
)

bt_results.plot_equity()

# 6. 多空分層回測
layer_bt_results = backtest_layer(
    price_factor=panel['open'],
    strategy_factor=mean_reversion_factor_neutralized,
    transaction_cost=(0.0003, 0.0003),
    long_top_n=3,
    short_bottom_n=3,
)

layer_bt_results.plot_equity()

