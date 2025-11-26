import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import traceback
import sys
from io import StringIO

st.set_page_config(page_title="Phandas Alpha Research", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for professional look
st.markdown("""
<style>
    .header-container {
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 2rem;
    }
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .editor-label {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.75rem;
    }
    .results-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <div class="main-header">Phandas Alpha Research</div>
    <div class="sub-header">量化因子回測分析平台</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration (Collapsible)
with st.sidebar:
    st.header("設定")
    
    with st.expander("回測參數", expanded=True):
        transaction_cost = st.number_input("交易成本 (%)", min_value=0.0, max_value=1.0, value=0.03, step=0.01) / 100
        full_rebalance = st.checkbox("完全調倉", value=False)
        factor_name = st.text_input("因子名稱", value="alpha")
    
    with st.expander("數據參考"):
        st.info("""
**預載入因子**:
- close
- open
- high
- low
- volume
""")
    
    with st.expander("資源"):
        st.markdown("""
**算子手冊**:  
[算子手冊](https://phandas.readthedocs.io/zh-tw/latest/guide/operators_guide.html)

**原始碼**:  
[GitHub 專案](https://github.com/quantbai/phandas)
""")

# Default factor code template
DEFAULT_CODE = """# 在下方編寫您的因子表達式
# 範例：20日動量
alpha = rank(close / ts_delay(close, 20))
"""

# Main Layout: Full Width
st.markdown('<div class="editor-label">因子表達式</div>', unsafe_allow_html=True)

user_code = st.text_area(
    "代碼編輯器",
    value=DEFAULT_CODE,
    height=250,
    help="使用 phandas 算子編寫您的因子表達式",
    label_visibility="collapsed"
)

run_button = st.button("執行回測", type="primary", use_container_width=True, key="backtest_button")

st.markdown("---")

# Results section
results_container = st.container()

# Execute backtest when button is clicked
if run_button:
    with results_container:
        with st.spinner("執行回測中..."):
            try:
                # Pre-execution setup
                exec_globals = {}
                setup_code = """
import warnings
warnings.filterwarnings('ignore')

from phandas import *
import matplotlib.pyplot as plt

# Fetch market data
panel = fetch_data(
    symbols=['ETH', 'SOL', 'ARB', 'OP', 'POL', 'SUI'],
    start_date='2022-01-01',
    sources=['binance']
)

# Extract price/volume factors
close = panel['close']
open = panel['open']
high = panel['high']
low = panel['low']
volume = panel['volume']
"""
                
                exec(setup_code, exec_globals)
                
                # Execute user code
                exec(user_code, exec_globals)
                
                # Check if alpha is defined
                if 'alpha' not in exec_globals:
                    st.error("錯誤：您的代碼必須定義名為 'alpha' 的變數")
                else:
                    alpha = exec_globals['alpha']
                    
                    # Set factor name
                    alpha.name = factor_name
                    
                    # Run backtest (same as skewness.py)
                    backtest_code = f"""
bt_results = backtest(
    entry_price_factor=open,
    strategy_factor=alpha,
    transaction_cost=({transaction_cost}, {transaction_cost}),
    full_rebalance={full_rebalance}
)
"""
                    exec(backtest_code, exec_globals)
                    bt_results = exec_globals['bt_results']
                    
                    # Display success
                    st.success("回測完成")
                    
                    # Plot equity curve (full width, larger)
                    st.markdown('<div class="results-label">回測結果</div>', unsafe_allow_html=True)
                    
                    # Use phandas built-in plot_equity() method
                    try:
                        # Clear any existing figures
                        plt.close('all')
                        
                        # Call plot_equity with larger figsize
                        bt_results.plot_equity(figsize=(14, 8))
                        
                        # Get the current figure and display it full width
                        fig = plt.gcf()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception as plot_error:
                        st.error(f"產生圖表時發生錯誤：{plot_error}")
                        st.code(traceback.format_exc(), language="python")
                    
            except Exception as e:
                st.error("執行回測時發生錯誤：")
                st.code(traceback.format_exc(), language="python")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; padding: 1rem; font-size: 0.9rem;'>
    Powered by <a href='https://github.com/quantbai/phandas' target='_blank' style='color: #666; text-decoration: none;'>Phandas</a>
</div>
""", unsafe_allow_html=True)
