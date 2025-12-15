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
                --bg-primary: #06080c;
                --bg-secondary: #0c0f14;
                --bg-tertiary: #14171f;
                --bg-card: rgba(14, 17, 23, 0.9);
                --accent-primary: #00d4ff;
                --accent-secondary: #0ea5e9;
                --accent-glow: rgba(0, 212, 255, 0.2);
                --accent-gold: #fbbf24;
                --text-primary: #f1f5f9;
                --text-secondary: #94a3b8;
                --text-muted: #64748b;
                --border-subtle: rgba(148, 163, 184, 0.1);
                --border-accent: rgba(0, 212, 255, 0.25);
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
                    radial-gradient(ellipse 100% 80% at 50% -30%, rgba(0, 212, 255, 0.06), transparent 60%),
                    radial-gradient(ellipse 50% 50% at 100% 100%, rgba(14, 165, 233, 0.03), transparent);
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
        st.info("""
**Available Factors:**
- close
- open
- high
- low
- volume
""")
    
    with st.expander("Resources"):
        st.markdown("""
**Operators Guide:**  
[Documentation](https://phandas.readthedocs.io/guide/operators_guide.html)

**Source Code:**  
[GitHub Repository](https://github.com/quantbai/phandas)
""")


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
        with st.spinner("Processing..."):
            try:
                csv_path = os.path.join(os.path.dirname(__file__), 'crypto_1d.csv')
                if not os.path.exists(csv_path):
                    st.error(f"Data file not found: {csv_path}")
                    st.stop()
                
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
                    st.error("Error: Your code must define a variable named 'alpha'")
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
                    
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total Return", f"{m['total_return']:+.2%}")
                    k2.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
                    k3.metric("Max Drawdown", f"{m['max_drawdown']:.2%}")
                    k4.metric("Linearity", f"{m['linearity']:.4f}")
                    
                    st.markdown('<div style="height: 1.25rem"></div>', unsafe_allow_html=True)
                    
                    plt.style.use('dark_background')
                    plt.rcParams['figure.dpi'] = 150
                    plt.rcParams['savefig.dpi'] = 150
                    
                    fig = plt.figure(figsize=(14, 5))
                    ax = fig.add_subplot(111)
                    
                    equity = bt_results.equity
                    

                    # Modern Crypto-style Equity Curve
                    
                    import numpy as np
                    
                    x = np.arange(len(equity))
                    y = equity.values
                    
                    # 1. Main Line
                    ax.plot(x, y, color='#00d4ff', linewidth=2.5, alpha=1.0, zorder=3)
                    
                    # 2. Vertical Gradient Fill
                    from matplotlib.colors import LinearSegmentedColormap
                    from matplotlib.patches import Polygon
                    
                    # Calculate limits
                    ylim_min = y.min() * 0.98  # Slightly tighter padding
                    ylim_max = y.max() * 1.02
                    ax.set_ylim(ylim_min, ylim_max)
                    ax.set_xlim(x.min(), x.max())
                    
                    # Create gradient from transparent to semi-opaque cyan
                    # (0, 212, 255) is #00d4ff
                    gradient_colors = [(0, 0.83, 1, 0), (0, 0.83, 1, 0.3)] 
                    cmap = LinearSegmentedColormap.from_list('cyan_gradient', gradient_colors)
                    
                    # Create vertical gradient array
                    # We want the gradient to map to the Y-axis range
                    # imshow plots a 2D array. We create a vertical gradient (0 at bottom, 1 at top)
                    Z = np.linspace(0, 1, 256).reshape(-1, 1)
                    Z = np.hstack((Z, Z)) # Duplicate to make it 2D
                    
                    # Plot gradient image covering the whole plot area
                    im = ax.imshow(Z, aspect='auto', cmap=cmap, 
                                 extent=[x.min(), x.max(), ylim_min, ylim_max], 
                                 origin='lower', zorder=1)
                    
                    # Create a polygon to clip the image to the area under the curve
                    # Vertices: bottom-left -> all (x,y) points -> bottom-right
                    verts = [(x.min(), ylim_min)] + list(zip(x, y)) + [(x.max(), ylim_min)]
                    poly = Polygon(verts, facecolor='none')
                    ax.add_patch(poly) # Necessary to attach to axes
                    im.set_clip_path(poly)
                    
                    # 3. Dashed line for initial capital
                    ax.axhline(y=equity.iloc[0], color='#ffffff', linewidth=1, linestyle='--', alpha=0.3, zorder=2)
                    
                    ax.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color='#475569')
                    
                    for spine in ['top', 'right']:
                        ax.spines[spine].set_visible(False)
                    for spine in ['bottom', 'left']:
                        ax.spines[spine].set_color('#334155')
                        ax.spines[spine].set_linewidth(0.8)
                    
                    ax.tick_params(axis='both', colors='#94a3b8', labelsize=10, width=0.8, length=4)
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, p: f'{val:,.0f}'))
                    
                    tick_positions = np.linspace(0, len(equity)-1, 8, dtype=int)
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels([equity.index[i].strftime('%Y-%m') for i in tick_positions], fontsize=9)
                    
                    fig.patch.set_facecolor('#0a0a0f')
                    ax.set_facecolor('#0a0a0f')
                    
                    plt.tight_layout(pad=1.0)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    
                    st.markdown('<div style="height: 0.75rem"></div>', unsafe_allow_html=True)
                    
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
                            
                            st.dataframe(pd.DataFrame(ic_data), use_container_width=True, hide_index=True)
                            
                        except Exception as ic_error:
                            st.warning(f"IC calculation failed: {ic_error}")
                    
            except Exception as e:
                st.error("Execution error:")
                st.code(traceback.format_exc(), language="python")


st.markdown("""
    <div class="footer-text">
        Powered by <a href="https://github.com/quantbai/phandas" target="_blank">Phandas</a>
    </div>
""", unsafe_allow_html=True)
