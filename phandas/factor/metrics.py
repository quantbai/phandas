"""Factor metrics operators for alpha research.

Provides metrics that are computed as Factor-to-Factor transformations,
enabling seamless integration with phandas operator chains and .show() visualization.
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import FactorBase


class MetricsMixin:
    """Mixin providing factor metrics (IC, crowding, turnover, etc.).
    
    All metrics return Factor objects, enabling method chaining and
    visualization via .show().
    """
    
    def crowding(self: 'FactorBase', volume: 'FactorBase', close: 'FactorBase',
                 lookback: int = 20) -> 'FactorBase':
        """Factor crowding based on abnormal volume correlation.
        
        Measures whether large factor signal changes coincide with
        abnormally high trading volume, indicating crowded positioning.
        
        Parameters
        ----------
        volume : Factor
            Trading volume factor
        close : Factor
            Close price factor
        lookback : int, default 20
            Lookback period for average dollar volume
        
        Returns
        -------
        Factor
            Daily cross-sectional correlation between |delta_signal| and
            abnormal dollar volume. Values > 0.3 sustained suggest crowding.
        
        Notes
        -----
        Algorithm:
        1. Compute |signal_t - signal_{t-1}| (signal change magnitude)
        2. Compute abnormal_vol = dollar_vol / MA(dollar_vol, lookback) - 1
        3. Cross-sectional rank correlation of the above
        """
        from . import Factor
        
        delta_signal = self - self.ts_delay(1)
        delta_signal_abs = delta_signal.abs()
        
        dollar_vol = volume * close
        avg_dollar_vol = dollar_vol.ts_mean(lookback)
        abnormal_vol = dollar_vol / avg_dollar_vol - 1
        
        delta_ranked = delta_signal_abs.rank()
        vol_ranked = abnormal_vol.rank()
        
        merged = pd.merge(
            delta_ranked.data, vol_ranked.data,
            on=['timestamp', 'symbol'],
            suffixes=('_sig', '_vol')
        )
        
        def cs_corr(group):
            x = group['factor_sig']
            y = group['factor_vol']
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                return pd.Series(np.nan, index=group.index)
            corr = x[valid].corr(y[valid])
            return pd.Series(corr, index=group.index)
        
        merged['factor'] = merged.groupby('timestamp', group_keys=False).apply(
            cs_corr, include_groups=False
        ).values
        
        result = merged[['timestamp', 'symbol', 'factor']].drop_duplicates()
        return Factor(result, f"crowding({self.name},{lookback})")
    
    def turnover(self: 'FactorBase', lag: int = 1) -> 'FactorBase':
        """Factor turnover as absolute rank change.
        
        Parameters
        ----------
        lag : int, default 1
            Lag period for comparison
        
        Returns
        -------
        Factor
            Absolute change in cross-sectional rank, broadcast to all symbols.
            Higher values indicate more factor churn.
        """
        from . import Factor
        
        ranked = self.rank()
        lagged = ranked.ts_delay(lag)
        
        diff = ranked - lagged
        diff_abs = diff.abs()
        
        result = diff_abs.data.copy()
        result['factor'] = result.groupby('timestamp')['factor'].transform('mean')
        
        return Factor(result, f"turnover({self.name},{lag})")
    
    def autocorr(self: 'FactorBase', lag: int = 1) -> 'FactorBase':
        """Factor autocorrelation time series.
        
        Computes cross-sectional correlation between factor_t and factor_{t-lag}.
        
        Parameters
        ----------
        lag : int, default 1
            Lag period
        
        Returns
        -------
        Factor
            Autocorrelation at each timestamp, broadcast to all symbols.
        """
        from . import Factor
        
        current = self.rank()
        lagged = current.ts_delay(lag)
        
        merged = pd.merge(
            current.data, lagged.data,
            on=['timestamp', 'symbol'],
            suffixes=('_cur', '_lag')
        )
        
        def cs_corr(group):
            x = group['factor_cur']
            y = group['factor_lag']
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                return np.nan
            return x[valid].corr(y[valid])
        
        corr_by_ts = merged.groupby('timestamp').apply(cs_corr, include_groups=False)
        
        result = self.data.copy()
        result['factor'] = result['timestamp'].map(corr_by_ts)
        
        return Factor(result, f"autocorr({self.name},{lag})")
    
    def ic(self: 'FactorBase', close: 'FactorBase', 
           method: str = 'spearman') -> 'FactorBase':
        """Information Coefficient time series.
        
        Cross-sectional correlation between factor_{t-1} and return_t.
        
        Parameters
        ----------
        close : Factor
            Close price factor
        method : {'spearman', 'pearson'}, default 'spearman'
            Correlation method
        
        Returns
        -------
        Factor
            Daily IC values, broadcast to all symbols.
        """
        from . import Factor
        
        prev_close = close.ts_delay(1)
        forward_returns = (close - prev_close) / prev_close
        
        lagged_factor = self.ts_delay(1)
        
        if method == 'spearman':
            f_data = lagged_factor.rank()
            r_data = forward_returns.rank()
        else:
            f_data = lagged_factor
            r_data = forward_returns
        
        merged = pd.merge(
            f_data.data, r_data.data,
            on=['timestamp', 'symbol'],
            suffixes=('_f', '_r')
        )
        
        def cs_corr(group):
            x = group['factor_f']
            y = group['factor_r']
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                return np.nan
            return x[valid].corr(y[valid])
        
        ic_by_ts = merged.groupby('timestamp').apply(cs_corr, include_groups=False)
        
        result = self.data.copy()
        result['factor'] = result['timestamp'].map(ic_by_ts)
        
        return Factor(result, f"ic({self.name},{method})")
    
    def pnl(self: 'FactorBase', close: 'FactorBase') -> 'FactorBase':
        """Factor PnL contribution per symbol (full-rebalance, no transaction cost).
        
        Uses close-to-close returns with previous day's signal as weights.
        
        Parameters
        ----------
        close : Factor
            Close price factor
        
        Returns
        -------
        Factor
            Daily PnL per symbol = weight_{t-1} * (close_t - close_{t-1}) / close_{t-1}
            Sum across symbols at each timestamp = portfolio return
        """
        weights = self.signal().ts_delay(1)
        prev_close = close.ts_delay(1)
        returns = (close - prev_close) / prev_close
        result = weights * returns
        result.name = f"pnl({self.name})"
        return result
    
    def rolling_sharpe(self: 'FactorBase', close: 'FactorBase', 
                       window: int = 60) -> 'FactorBase':
        """Rolling Sharpe ratio of factor PnL.
        
        Parameters
        ----------
        close : Factor
            Close price factor
        window : int, default 60
            Rolling window for Sharpe calculation
        
        Returns
        -------
        Factor
            Annualized rolling Sharpe ratio, broadcast to all symbols.
        """
        from . import Factor
        
        daily_pnl = self.pnl(close)
        
        pnl_sum = daily_pnl.data.groupby('timestamp')['factor'].sum()
        pnl_sum = pnl_sum.to_frame('pnl').reset_index()
        
        rolling_mean = pnl_sum['pnl'].rolling(window, min_periods=window).mean()
        rolling_std = pnl_sum['pnl'].rolling(window, min_periods=window).std()
        sharpe = (rolling_mean / rolling_std) * np.sqrt(365)
        
        pnl_sum['sharpe'] = sharpe
        ts_sharpe = pnl_sum.set_index('timestamp')['sharpe']
        
        result = self.data.copy()
        result['factor'] = result['timestamp'].map(ts_sharpe)
        
        return Factor(result, f"rolling_sharpe({self.name},{window})")
    
    def drawdown(self: 'FactorBase', close: 'FactorBase') -> 'FactorBase':
        """Portfolio drawdown from high water mark.
        
        Parameters
        ----------
        close : Factor
            Close price factor
        
        Returns
        -------
        Factor
            Drawdown (negative values), broadcast to all symbols.
            0 = at high water mark, -0.1 = 10% below peak.
        """
        from . import Factor
        
        daily_pnl = self.pnl(close)
        
        pnl_sum = daily_pnl.data.groupby('timestamp')['factor'].sum()
        equity = (1 + pnl_sum).cumprod()
        hwm = equity.expanding().max()
        dd = equity / hwm - 1
        
        dd_df = dd.to_frame('dd').reset_index()
        dd_df.columns = ['timestamp', 'dd']
        ts_dd = dd_df.set_index('timestamp')['dd']
        
        result = self.data.copy()
        result['factor'] = result['timestamp'].map(ts_dd)
        
        return Factor(result, f"drawdown({self.name})")
