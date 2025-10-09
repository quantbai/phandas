from phandas import Panel, backtest, Factor
from phandas.operators import ts_mean, vector_neut, ts_rank, log, ts_std_dev
from phandas.utils import save_factor

# 1. 下載數據
# meme = [ 'DOGE', 'SHIB', 'PEPE', 'PENGU', 'BONK','WIF','FLOKI','TURBO']
# mainnet = ['BNB', 'ETH', 'SOL', 'MATIC', 'ARB', 'OP']
# panel = fetch_data(
#     symbols=mainnet,
#     timeframe='1d',
#     start_date='2023-01-01'
# )
panel = Panel.from_csv('crypto_1d.csv')

# 2. 提取因子
open = panel['open']
close = panel['close']
volume = panel['volume']
high = panel['high']
low = panel['low']


# 3. 計算因子
from phandas import Panel, ts_mean, ts_std_dev, ts_delay, ts_sum, log, vector_neut, rank

# 載入數據
panel = Panel.from_csv('crypto_1d.csv')
close = panel['close']
volume = panel['volume']

# 共用變數
n = 60
log_price = log(close)
r = log(close) - ts_delay(log(close), 1)
vol = ts_std_dev(r, n)

r_mean = ts_mean(r, 10)
r_vol = ts_std_dev(r, 10)
short_rev = -(r - r_mean) / r_vol
short_rev = vector_neut(rank(short_rev), rank(volume))
short_rev.name = "E_ShortTerm_Reversion"

# ============================================================
# F. Ito Drift Estimator (隨機微積分推導)
# 理念：直接估 drift ≈ ΔlogS - ½σ²，用於反向交易
# ============================================================
drift_est = r - 0.5 * (vol ** 2)
ito_reversion = -drift_est
ito_reversion = vector_neut(ito_reversion, volume)
ito_reversion.name = "F_Ito_Drift_Reversion"

factor = short_rev
# 4. 回測
result = backtest(
    price_factor=open, 
    strategy_factor=factor, # 將策略因子替換為中性化後的因子
    transaction_cost=(0.0003, 0.0003) # 使用進場/出場分開的費率
)

# 5. 結果分析
result.plot_equity()