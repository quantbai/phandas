import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import traceback
import sys
import warnings
import os
from datetime import datetime
import phandas


st.set_page_config(
    page_title="Phandas Alpha Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)


def inject_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
            
            :root {
                /* 使用 Streamlit 系統主題變數，自動適應 light/dark 模式 */
                --bg-primary: var(--background-color);
                --bg-secondary: var(--secondary-background-color);
                --bg-tertiary: color-mix(in srgb, var(--secondary-background-color) 80%, var(--text-color) 5%);
                --bg-card: var(--secondary-background-color);
                --accent-primary: var(--primary-color, #00d4ff);
                --accent-secondary: color-mix(in srgb, var(--primary-color, #00d4ff) 80%, #0ea5e9 20%);
                --accent-glow: color-mix(in srgb, var(--primary-color, #00d4ff) 20%, transparent 80%);
                --accent-gold: #fbbf24;
                --text-primary: var(--text-color);
                --text-secondary: color-mix(in srgb, var(--text-color) 60%, transparent 40%);
                --text-muted: color-mix(in srgb, var(--text-color) 40%, transparent 60%);
                --border-subtle: color-mix(in srgb, var(--text-color) 10%, transparent 90%);
                --border-accent: color-mix(in srgb, var(--primary-color, #00d4ff) 25%, transparent 75%);
                --positive: #10b981;
                --negative: #ef4444;
            }
            
            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                color: var(--text-primary);
                letter-spacing: 0.01em;
            }
            
            .stApp {
                background: var(--bg-primary);
            }
            
            .stApp::before {
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: 
                    radial-gradient(ellipse 100% 80% at 50% -30%, var(--accent-glow), transparent 60%),
                    radial-gradient(ellipse 50% 50% at 100% 100%, color-mix(in srgb, var(--accent-secondary) 3%, transparent 97%), transparent);
                pointer-events: none;
                z-index: 0;
            }
            
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                letter-spacing: -0.01em;
                color: var(--text-primary);
            }
            
            .block-container {
                padding: 1.5rem 2.5rem 2rem 2.5rem;
                max-width: 100%;
            }
            
            .header-bar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 1.25rem 0;
                margin-bottom: 1.75rem;
                border-bottom: 1px solid var(--border-subtle);
            }
            
            .brand-section {
                display: flex;
                align-items: baseline;
                gap: 0.75rem;
            }
            
            .brand-title {
                font-family: 'Inter', sans-serif;
                font-size: 1.5rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                color: var(--text-primary);
                margin: 0;
            }
            
            .brand-accent {
                color: var(--accent-primary);
            }
            
            .version-tag {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.65rem;
                font-weight: 500;
                color: var(--text-muted);
                background: var(--bg-tertiary);
                padding: 0.15rem 0.5rem;
                border-radius: 3px;
                border: 1px solid var(--border-subtle);
            }
            
            .header-links {
                display: flex;
                gap: 1.5rem;
                font-size: 0.8rem;
            }
            
            .header-links a {
                color: var(--text-muted);
                text-decoration: none;
                transition: color 0.2s;
            }
            
            .header-links a:hover {
                color: var(--accent-primary);
            }
            
            .editor-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.75rem;
            }
            
            .section-label {
                font-size: 0.7rem;
                font-weight: 600;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: var(--text-muted);
            }
            
            .stButton button {
                background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
                color: #000 !important;
                border: none !important;
                border-radius: 6px !important;
                font-weight: 600 !important;
                font-size: 0.8rem !important;
                letter-spacing: 0.05em !important;
                padding: 0.65rem 1.5rem !important;
                transition: all 0.2s ease !important;
                text-transform: uppercase !important;
            }
            
            .stButton button:hover {
                box-shadow: 0 4px 16px -2px var(--accent-glow) !important;
                transform: translateY(-1px) !important;
            }
            
            .stTextArea textarea {
                background-color: var(--bg-secondary) !important;
                border: 1px solid var(--border-subtle) !important;
                color: var(--text-primary) !important;
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 13px !important;
                line-height: 1.7 !important;
                border-radius: 8px !important;
                padding: 1rem !important;
            }
            
            .stTextArea textarea:focus {
                border-color: var(--accent-primary) !important;
                box-shadow: 0 0 0 2px var(--accent-glow) !important;
            }
            
            .stTextInput input, .stNumberInput input {
                background-color: var(--bg-tertiary) !important;
                border: 1px solid var(--border-subtle) !important;
                color: var(--text-primary) !important;
                font-family: 'JetBrains Mono', monospace !important;
                border-radius: 6px !important;
            }
            
            div[data-testid="metric-container"] {
                background: var(--bg-secondary);
                border: 1px solid var(--border-subtle);
                border-left: 2px solid var(--accent-primary);
                padding: 0.875rem 1rem;
                border-radius: 0 8px 8px 0;
            }
            
            div[data-testid="metric-container"] label {
                font-size: 0.65rem !important;
                font-weight: 600 !important;
                letter-spacing: 0.1em !important;
                text-transform: uppercase !important;
                color: var(--text-muted) !important;
            }
            
            div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 1.35rem !important;
                font-weight: 600 !important;
                color: var(--text-primary) !important;
                letter-spacing: -0.01em !important;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
                background: var(--bg-secondary);
                padding: 3px;
                border-radius: 8px;
                border: 1px solid var(--border-subtle);
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: transparent !important;
                border: none !important;
                border-radius: 5px !important;
                color: var(--text-muted) !important;
                font-size: 0.75rem !important;
                font-weight: 500 !important;
                letter-spacing: 0.04em !important;
                padding: 0.5rem 1rem !important;
            }
            
            .stTabs [aria-selected="true"] {
                background: var(--bg-tertiary) !important;
                color: var(--text-primary) !important;
            }
            
            .stDataFrame {
                background-color: var(--bg-secondary) !important;
                border-radius: 6px !important;
            }
            
            table {
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 0.8rem !important;
            }
            
            .stExpander {
                background: var(--bg-secondary) !important;
                border: 1px solid var(--border-subtle) !important;
                border-radius: 6px !important;
            }

            .stExpander [data-testid="stExpanderDetails"] {
                padding-bottom: 1rem !important;
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
            
            ::-webkit-scrollbar {
                width: 5px;
                height: 5px;
            }
            
            ::-webkit-scrollbar-track {
                background: var(--bg-primary);
            }
            
            ::-webkit-scrollbar-thumb {
                background: var(--bg-tertiary);
                border-radius: 3px;
            }
            
            .footer-text {
                text-align: center;
                color: var(--text-muted);
                font-size: 0.75rem;
                padding: 1.5rem 0;
                margin-top: 2rem;
                border-top: 1px solid var(--border-subtle);
            }
            
            .footer-text a {
                color: var(--accent-primary);
                text-decoration: none;
            }
            
        </style>
    """, unsafe_allow_html=True)



inject_custom_css()


st.markdown("""
    <div class="header-bar">
        <div class="brand-section">
            <span class="brand-title">PHANDAS <span class="brand-accent">ALPHA LAB</span></span>
            <span class="version-tag">v0.18.0</span>
        </div>
        <div class="header-links">
            <a href="https://phandas.readthedocs.io/guide/operators_guide.html" target="_blank">Documentation</a>
            <a href="https://github.com/quantbai/phandas" target="_blank">GitHub</a>
        </div>
    </div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Settings")
    
    with st.expander("Backtest Parameters", expanded=True):
        factor_name = st.text_input("Factor Name", value="alpha", help="Identifier for your factor")
        transaction_cost = st.number_input("Transaction Cost (%)", min_value=0.0, max_value=1.0, value=0.03, step=0.01) / 100
        full_rebalance = st.checkbox("Full Rebalance", value=False)
    
    with st.expander("Data Reference"):
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.08) 0%, rgba(14, 165, 233, 0.04) 100%);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 8px;
            padding: 12px 14px;
            margin: 0.25rem 0;
            margin-bottom: 0.5rem;
        ">
            <div style="
                font-family: 'Inter', sans-serif;
                font-size: 0.65rem;
                font-weight: 600;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: var(--text-muted, #64748b);
                margin-bottom: 0.6rem;
            ">Available Factors</div>
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                color: var(--text-primary, #f1f5f9);
                display: flex;
                flex-direction: column;
                gap: 4px;
            ">
                <span style="color: #00d4ff;">close</span>
                <span style="color: #00d4ff;">open</span>
                <span style="color: #00d4ff;">high</span>
                <span style="color: #00d4ff;">low</span>
                <span style="color: #00d4ff;">volume</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("Resources"):
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.08) 0%, rgba(14, 165, 233, 0.04) 100%);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 8px;
            padding: 12px 14px;
            margin: 0.25rem 0;
            margin-bottom: 0.5rem;
        ">
            <div style="margin-bottom: 0.8rem;">
                <div style="
                    font-family: 'Inter', sans-serif;
                    font-size: 0.65rem;
                    font-weight: 600;
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    color: var(--text-muted, #64748b);
                    margin-bottom: 0.3rem;
                ">Operators Guide</div>
                <a href="https://phandas.readthedocs.io/guide/operators_guide.html" target="_blank" style="
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 0.8rem;
                    color: #00d4ff;
                    text-decoration: none;
                    transition: opacity 0.2s;
                    display: block;
                ">Documentation</a>
            </div>
            <div>
                <div style="
                    font-family: 'Inter', sans-serif;
                    font-size: 0.65rem;
                    font-weight: 600;
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    color: var(--text-muted, #64748b);
                    margin-bottom: 0.3rem;
                ">Source Code</div>
                <a href="https://github.com/quantbai/phandas" target="_blank" style="
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 0.8rem;
                    color: #00d4ff;
                    text-decoration: none;
                    display: block;
                ">GitHub Repository</a>
            </div>
        </div>
        """, unsafe_allow_html=True)


col_left, col_right = st.columns([35, 65], gap="medium")

with col_left:
    st.markdown('<div class="section-label">Strategy Editor</div>', unsafe_allow_html=True)
    
    default_code = """alpha = rank(close / ts_delay(close, 20))
"""
    factor_code = st.text_area(
        "code",
        value=default_code,
        height=420,
        label_visibility="collapsed"
    )
    
    run_bt = st.button("EXECUTE", type="primary", use_container_width=True)


with col_right:
    st.markdown('<div class="section-label">Performance Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 0.5rem"></div>', unsafe_allow_html=True)
    
    result_container = st.container()


if run_bt:
    with result_container:
        # 先在 spinner 內完成所有計算
        results_ready = False
        error_info = None
        
        with st.spinner("Processing..."):
            try:
                csv_path = os.path.join(os.path.dirname(__file__), 'crypto_1d.csv')
                if not os.path.exists(csv_path):
                    error_info = f"Data file not found: {csv_path}"
                else:
                    exec_globals = vars(phandas).copy()
                    exec_globals.update({
                        'csv_path': csv_path,
                        'plt': plt,
                        'pd': pd,
                        'warnings': sys.modules['warnings']
                    })
                    
                    setup_code = """
import warnings
warnings.filterwarnings('ignore')
import signal
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

try:
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError("Timeout")))
    signal.alarm(60)
except:
    pass

panel = Panel.from_csv(csv_path)

close = panel['close']
open = panel['open']
high = panel['high']
low = panel['low']
volume = panel['volume']
"""
                    exec(setup_code, exec_globals)
                    
                    try:
                        exec(factor_code, exec_globals)
                    finally:
                        try:
                            import signal
                            signal.alarm(0)
                        except:
                            pass
                    
                    if 'alpha' not in exec_globals:
                        error_info = "Error: Your code must define a variable named 'alpha'"
                    else:
                        alpha = exec_globals['alpha']
                        alpha.name = factor_name
                        
                        close_price = exec_globals['close']
                        
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
                        m = bt_results.metrics
                        
                        turnover_df = bt_results.turnover
                        avg_turnover = turnover_df['turnover'].mean() if not turnover_df.empty else 0.0
                        
                        # 偵測當前 Streamlit 主題
                        theme_base = st.get_option("theme.base")
                        is_dark_mode = theme_base == "dark" or theme_base is None
                        
                        # 根據主題設定顏色（背景透明，會自動跟隨頁面）
                        if is_dark_mode:
                            text_color = '#94a3b8'
                            grid_color = '#475569'
                            spine_color = '#334155'
                            line_alpha = 0.3
                            plt.style.use('dark_background')
                        else:
                            text_color = '#374151'
                            grid_color = '#d1d5db'
                            spine_color = '#9ca3af'
                            line_alpha = 0.5
                            plt.style.use('default')
                        
                        accent_color = '#00d4ff'
                        
                        plt.rcParams['figure.dpi'] = 150
                        plt.rcParams['savefig.dpi'] = 150
                        
                        # 預先生成圖表
                        import numpy as np
                        from matplotlib.colors import LinearSegmentedColormap
                        from matplotlib.patches import Polygon
                        
                        equity = bt_results.equity
                        fig = plt.figure(figsize=(14, 5))
                        ax = fig.add_subplot(111)
                        
                        x = np.arange(len(equity))
                        y = equity.values
                        
                        ax.plot(x, y, color=accent_color, linewidth=2.5, alpha=1.0, zorder=3)
                        
                        ylim_min = y.min() * 0.98
                        ylim_max = y.max() * 1.02
                        ax.set_ylim(ylim_min, ylim_max)
                        ax.set_xlim(x.min(), x.max())
                        
                        gradient_alpha = 0.3 if is_dark_mode else 0.15
                        gradient_colors = [(0, 0.83, 1, 0), (0, 0.83, 1, gradient_alpha)]
                        cmap = LinearSegmentedColormap.from_list('cyan_gradient', gradient_colors)
                        
                        Z = np.linspace(0, 1, 256).reshape(-1, 1)
                        Z = np.hstack((Z, Z))
                        
                        im = ax.imshow(Z, aspect='auto', cmap=cmap,
                                     extent=[x.min(), x.max(), ylim_min, ylim_max],
                                     origin='lower', zorder=1)
                        
                        verts = [(x.min(), ylim_min)] + list(zip(x, y)) + [(x.max(), ylim_min)]
                        poly = Polygon(verts, facecolor='none')
                        ax.add_patch(poly)
                        im.set_clip_path(poly)
                        
                        baseline_color = '#ffffff' if is_dark_mode else '#000000'
                        ax.axhline(y=equity.iloc[0], color=baseline_color, linewidth=1, linestyle='--', alpha=line_alpha, zorder=2)
                        
                        ax.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color=grid_color)
                        
                        for spine in ['top', 'right']:
                            ax.spines[spine].set_visible(False)
                        for spine in ['bottom', 'left']:
                            ax.spines[spine].set_color(spine_color)
                            ax.spines[spine].set_linewidth(0.8)
                        
                        ax.tick_params(axis='both', colors=text_color, labelsize=10, width=0.8, length=4)
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, p: f'{val:,.0f}'))
                        
                        tick_positions = np.linspace(0, len(equity)-1, 8, dtype=int)
                        ax.set_xticks(tick_positions)
                        ax.set_xticklabels([equity.index[i].strftime('%Y-%m') for i in tick_positions], fontsize=9)
                        
                        fig.patch.set_facecolor('none')
                        fig.patch.set_alpha(0)
                        ax.set_facecolor('none')
                        ax.patch.set_alpha(0)
                        
                        plt.tight_layout(pad=1.0)
                        
                        # 預先將圖表轉成 PNG buffer，加速渲染
                        import io
                        fig_buffer = io.BytesIO()
                        fig.savefig(fig_buffer, format='png', transparent=True, 
                                  facecolor='none', edgecolor='none', bbox_inches='tight')
                        fig_buffer.seek(0)
                        plt.close(fig)
                        
                        # 預先準備 IC 數據
                        ic_data = None
                        ic_error = None
                        try:
                            from phandas import FactorAnalyzer
                            analyzer = FactorAnalyzer([alpha], close_price, horizons=[1, 7, 30])
                            ic_results = analyzer.ic()
                            factor_ic = ic_results.get(factor_name, {})
                            ic_data = []
                            for h in [1, 7, 30]:
                                h_data = factor_ic.get(h, {})
                                ic_data.append({
                                    "Horizon": f"{h}D",
                                    "IC Mean": f"{h_data.get('ic_mean', 0):.4f}",
                                    "IC Std": f"{h_data.get('ic_std', 0):.4f}",
                                    "IR": f"{h_data.get('ir', 0):.4f}",
                                    "T-Stat": f"{h_data.get('t_stat', 0):.2f}"
                                })
                        except Exception as e:
                            ic_error = str(e)
                        
                        results_ready = True
                        
            except Exception as e:
                error_info = traceback.format_exc()
        
        # Spinner 結束後，一次性渲染所有 UI
        if error_info:
            if "Error:" in str(error_info):
                st.error(error_info)
            else:
                st.error("Execution error:")
                st.code(error_info, language="python")
        elif results_ready:
            # 指標
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Return", f"{m['total_return']:+.2%}")
            k2.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
            k3.metric("Max Drawdown", f"{m['max_drawdown']:.2%}")
            k4.metric("Linearity", f"{m['linearity']:.4f}")
            
            st.markdown('<div style="height: 1.25rem"></div>', unsafe_allow_html=True)
            
            # Equity Curve (使用預先生成的 PNG)
            st.image(fig_buffer, use_container_width=True)
            
            st.markdown('<div style="height: 0.75rem"></div>', unsafe_allow_html=True)
            
            # Tabs
            tab1, tab2 = st.tabs(["Risk Metrics", "IC Analysis"])
            
            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Risk Profile**")
                    risk_df = pd.DataFrame([
                        ["Sortino Ratio", f"{m['sortino_ratio']:.2f}"],
                        ["Calmar Ratio", f"{m['calmar_ratio']:.2f}"],
                        ["VaR 95%", f"{m['var_95']:.2%}"],
                        ["CVaR", f"{m['cvar']:.2%}"],
                        ["Avg Turnover", f"{avg_turnover:.2%}"],
                    ], columns=["Metric", "Value"])
                    st.dataframe(risk_df, use_container_width=True, hide_index=True)
                
                with c2:
                    st.markdown("**Drawdown Periods**")
                    if 'drawdown_periods' in m and m['drawdown_periods']:
                        dd_data = []
                        for dd in m['drawdown_periods'][:5]:
                            dd_data.append({
                                "Depth": f"{dd['depth']:.2%}",
                                "Duration": f"{dd['duration_days']}d",
                                "End": str(dd['end']).split(' ')[0]
                            })
                        st.dataframe(pd.DataFrame(dd_data), use_container_width=True, hide_index=True)
                    else:
                        st.info("No significant drawdowns.")
            
            with tab2:
                st.markdown("**Information Coefficient**")
                if ic_data:
                    st.dataframe(pd.DataFrame(ic_data), use_container_width=True, hide_index=True)
                elif ic_error:
                    st.warning(f"IC calculation failed: {ic_error}")


st.markdown("""
    <div class="footer-text">
        Powered by <a href="https://github.com/quantbai/phandas" target="_blank">Phandas</a>
    </div>
""", unsafe_allow_html=True)