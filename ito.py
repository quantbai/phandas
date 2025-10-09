# 0. pip install phandas

from phandas import Panel, backtest, analyze_ic, vector_neut, rank, ts_mean, ts_std_dev, ts_sum, log, ts_delay

# 1. 載入數據
panel = Panel.from_csv('crypto_1d.csv')

# 2. 萃取因子
open = panel['open']
close = panel['close']
volume = panel['volume']

# 3. 計算 Momentum 因子
from phandas import Panel, ts_std_dev, ts_sum, log, ts_delay, vector_neut, rank

panel = Panel.from_csv('crypto_1d.csv')
close = panel['close']
volume = panel['volume']

n = 20
r = log(close) - ts_delay(log(close), 1)
sigma = ts_std_dev(r, window=60)

# detect jumps
is_jump = (r.abs() > 4 * sigma)

# ✅ 正確使用 where：保留非 jump，jump 改成 0
r_nojump = r.where(is_jump == False , other=0.0)

# compute momentum on non-jump series
mom_nojump = ts_sum(r_nojump, window=n) / n

# volume-neutralize and normalize

mom_nojump.name = "20d_Momentum_NoJump"

# 5. 回測
bt_results = backtest(
    price_factor=open,
    strategy_factor=mom_nojump,
    transaction_cost=(0.0003, 0.0003)
)

bt_results.plot_equity()