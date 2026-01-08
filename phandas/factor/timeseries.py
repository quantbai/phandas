"""Time-series operators for Factor."""

import numpy as np
import pandas as pd
from typing import Union, List, TYPE_CHECKING
from scipy.stats import norm, uniform, cauchy

if TYPE_CHECKING:
    from .base import FactorBase


class TimeSeriesMixin:
    """Mixin providing time-series operators (ts_*).
    
    Time-series operators compute statistics over rolling windows
    within each symbol's time series.
    """
    
    def ts_rank(self: 'FactorBase', window: int) -> 'FactorBase':
        """Where does today's value rank in the last N days? (0-1)
        
        1.0 = highest in window, 0 = lowest in window.
        
        Parameters
        ----------
        window : int
            Number of past days to look at
        
        Returns
        -------
        Factor
            Percentile rank within the window
        """
        from . import Factor
        
        self._validate_window(window)
        
        result = self.data.copy()
        
        def rank_window(w):
            if np.isnan(w).any() or len(np.unique(w)) == 1:
                return np.nan
            sorted_idx = np.argsort(w)
            rank_array = np.empty_like(sorted_idx, dtype=float)
            rank_array[sorted_idx] = np.arange(1, len(w) + 1)
            return rank_array[-1] / len(w)
        
        result['factor'] = result.groupby('symbol')['factor'].rolling(
            window, min_periods=window
        ).apply(rank_window, raw=True).reset_index(level=0, drop=True)
        
        return Factor(result, f"ts_rank({self.name},{window})")
    
    def ts_sum(self: 'FactorBase', window: int) -> 'FactorBase':
        """Add up all values in the last N days.
        
        Parameters
        ----------
        window : int
            Number of past days
        
        Returns
        -------
        Factor
            Rolling sum
        """
        from . import Factor
        
        self._validate_window(window)
        
        def safe_sum(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.sum()
        
        result = self._apply_rolling(safe_sum, window)
        return Factor(result, f"ts_sum({self.name},{window})")

    def ts_product(self: 'FactorBase', window: int) -> 'FactorBase':
        """Multiply all values in the last N days.
        
        Parameters
        ----------
        window : int
            Number of past days
        
        Returns
        -------
        Factor
            Rolling product
        """
        from . import Factor
        
        self._validate_window(window)
        
        def safe_prod(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.prod()
        
        result = self._apply_rolling(safe_prod, window)
        return Factor(result, f"ts_product({self.name},{window})")

    def ts_mean(self: 'FactorBase', window: int) -> 'FactorBase':
        """Average of the last N days.
        
        Parameters
        ----------
        window : int
            Number of past days to average
        
        Returns
        -------
        Factor
            Rolling average
        """
        from . import Factor
        
        self._validate_window(window)
        
        def safe_mean(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.mean()
        
        result = self._apply_rolling(safe_mean, window)
        return Factor(result, f"ts_mean({self.name},{window})")
        
    def ts_median(self: 'FactorBase', window: int) -> 'FactorBase':
        """Middle value of the last N days.
        
        Parameters
        ----------
        window : int
            Number of past days
        
        Returns
        -------
        Factor
            Rolling median
        """
        from . import Factor
        
        self._validate_window(window)
        result = self._apply_rolling(lambda x: x.median() if not x.isna().all() else np.nan, window)
        return Factor(result, f"ts_median({self.name},{window})")
    
    def ts_std_dev(self: 'FactorBase', window: int) -> 'FactorBase':
        """How spread out are values in the last N days?
        
        Higher = more volatile, lower = more stable.
        
        Parameters
        ----------
        window : int
            Number of past days
        
        Returns
        -------
        Factor
            Rolling standard deviation
        """
        from . import Factor
        
        self._validate_window(window)
        
        def safe_std(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.std()
        
        result = self._apply_rolling(safe_std, window)
        return Factor(result, f"ts_std_dev({self.name},{window})")
    
    def ts_min(self: 'FactorBase', window: int) -> 'FactorBase':
        """Rolling minimum over window.
        
        Parameters
        ----------
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            Rolling minimum, NaN if window incomplete
        """
        from . import Factor
        
        self._validate_window(window)
        
        def safe_min(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.min()
        
        result = self._apply_rolling(safe_min, window)
        return Factor(result, f"ts_min({self.name},{window})")
    
    def ts_max(self: 'FactorBase', window: int) -> 'FactorBase':
        """Rolling maximum over window.
        
        Parameters
        ----------
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            Rolling maximum, NaN if window incomplete
        """
        from . import Factor
        
        self._validate_window(window)
        
        def safe_max(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.max()
        
        result = self._apply_rolling(safe_max, window)
        return Factor(result, f"ts_max({self.name},{window})")
    
    def ts_arg_max(self: 'FactorBase', window: int) -> 'FactorBase':
        """Relative index of maximum in window (0=oldest).
        
        Parameters
        ----------
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            Index of max value in window
        """
        from . import Factor
        
        self._validate_window(window)
        
        def safe_arg_max(s):
            return np.nan if s.isna().all() else (len(s) - 1) - s.argmax()
        
        result = self._apply_rolling(safe_arg_max, window)
        return Factor(result, f"ts_arg_max({self.name},{window})")
        
    def ts_arg_min(self: 'FactorBase', window: int) -> 'FactorBase':
        """Relative index of minimum in window (0=oldest).
        
        Parameters
        ----------
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            Index of min value in window
        """
        from . import Factor
        
        self._validate_window(window)
        
        def safe_arg_min(s):
            return np.nan if s.isna().all() else (len(s) - 1) - s.argmin()
        
        result = self._apply_rolling(safe_arg_min, window)
        return Factor(result, f"ts_arg_min({self.name},{window})")

    def ts_count_nans(self: 'FactorBase', window: int) -> 'FactorBase':
        """Count NaN values in rolling window.
        
        Parameters
        ----------
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            Count of NaN values in window
        """
        from . import Factor
        
        self._validate_window(window)
        
        result = self.data.copy()
        result['factor'] = (result.groupby('symbol')['factor']
                           .rolling(window, min_periods=1)
                           .apply(lambda x: x.isna().sum(), raw=False)
                           .reset_index(level=0, drop=True))
        
        return Factor(result, f"ts_count_nans({self.name},{window})")
    
    def ts_av_diff(self: 'FactorBase', window: int) -> 'FactorBase':
        """Deviation from rolling mean.
        
        Parameters
        ----------
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            Deviation from rolling mean
        """
        from . import Factor
        
        self._validate_window(window)
        mean_factor = self.ts_mean(window)
        result = self.subtract(mean_factor)
        return Factor(result.data, f"ts_av_diff({self.name},{window})")

    def ts_scale(self: 'FactorBase', window: int, constant: float = 0) -> 'FactorBase':
        """Rolling min-max scale.
        
        Parameters
        ----------
        window : int
            Window size (periods)
        constant : float, default 0
            Offset constant added after scaling
        
        Returns
        -------
        Factor
            Scaled factor: (x-min)/(max-min) + constant
        """
        from . import Factor
        
        self._validate_window(window)

        min_factor = self.ts_min(window)
        max_factor = self.ts_max(window)
        
        result = (self - min_factor) / (max_factor - min_factor)
        result.data['factor'] = self._replace_inf(result.data['factor'])
        result = result + constant

        return Factor(result.data, f"ts_scale({self.name},{window},{constant})")
    
    def ts_zscore(self: 'FactorBase', window: int) -> 'FactorBase':
        """How unusual is today's value compared to recent history?
        
        Formula: (today - average) / std_dev
        Higher absolute value = more unusual.
        
        Parameters
        ----------
        window : int
            Number of past days for comparison
        
        Returns
        -------
        Factor
            Rolling z-score
        """
        from . import Factor
        
        self._validate_window(window)
        
        mean = self.ts_mean(window)
        std = self.ts_std_dev(window)
        
        result = (self - mean) / std
        result.data['factor'] = self._replace_inf(result.data['factor'])
        
        return Factor(result.data, f"ts_zscore({self.name},{window})")

    def ts_quantile(self: 'FactorBase', window: int, driver: str = "gaussian") -> 'FactorBase':
        """Rolling quantile transform (normal/uniform/Cauchy).
        
        Parameters
        ----------
        window : int
            Window size (periods)
        driver : {'gaussian', 'uniform', 'cauchy'}, default 'gaussian'
            Quantile transform distribution
        
        Returns
        -------
        Factor
            Rolling quantile-transformed factor
        """
        from . import Factor
        
        self._validate_window(window)

        valid_drivers = {
            "gaussian": norm.ppf,
            "uniform": uniform.ppf,
            "cauchy": cauchy.ppf,
        }

        if driver not in valid_drivers:
            raise ValueError(f"Invalid driver: {driver}. Must be one of {list(valid_drivers.keys())}")

        ppf_func = valid_drivers[driver]
        ranked_factor = self.ts_rank(window)

        result_data = ranked_factor.data.copy()
        epsilon = 1e-6
        result_data['factor'] = result_data['factor'].clip(lower=epsilon, upper=1-epsilon).apply(ppf_func)

        return Factor(result_data, f"ts_quantile({self.name},{window},{driver})")

    def ts_kurtosis(self: 'FactorBase', window: int) -> 'FactorBase':
        """Rolling excess kurtosis.
        
        Parameters
        ----------
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            Rolling excess kurtosis: E[(x-mean)^4]/std^4 - 3
        """
        from . import Factor
        
        self._validate_window(window)
        
        result = self.data.copy()
        
        def kurtosis_vectorized(group):
            vals = group.values
            n = len(vals)
            kurt_vals = np.full(n, np.nan)
            
            for i in range(window - 1, n):
                window_vals = vals[i-window+1:i+1]
                
                if np.isnan(window_vals).any():
                    continue
                
                if len(np.unique(window_vals)) < 2:
                    continue
                
                mean_val = np.mean(window_vals)
                std_val = np.std(window_vals, ddof=0)
                
                if std_val < 1e-10:
                    continue
                
                deviations = window_vals - mean_val
                kurt = np.mean(deviations**4) / (std_val**4) - 3
                kurt_vals[i] = kurt
            
            return pd.Series(kurt_vals, index=group.index)
        
        result['factor'] = result.groupby('symbol', group_keys=False)['factor'].apply(
            kurtosis_vectorized
        )
        
        return Factor(result, f"ts_kurtosis({self.name},{window})")

    def ts_skewness(self: 'FactorBase', window: int) -> 'FactorBase':
        """Is the distribution lopsided in the last N days?
        
        Positive = more extreme high values, negative = more extreme low values.
        Zero = symmetric distribution.
        
        Parameters
        ----------
        window : int
            Number of past days
        
        Returns
        -------
        Factor
            Rolling skewness
        """
        self._validate_window(window)
        
        n = window
        mean_val = self.ts_mean(window)
        diff = self - mean_val
        
        sum_cube_dev = diff.power(3).ts_sum(window)
        sum_sq_dev = diff.power(2).ts_sum(window)
        
        numerator = sum_cube_dev * n
        denominator = sum_sq_dev.power(1.5) * ((n - 1) * (n - 2))
        
        skew = numerator / denominator
        skew.name = f"ts_skewness({self.name},{window})"
        
        return skew

    def ts_backfill(self: 'FactorBase', window: int, k: int = 1) -> 'FactorBase':
        """Backfill NaN with k-th most recent non-NaN in window.
        
        Parameters
        ----------
        window : int
            Window size (periods)
        k : int, default 1
            Which recent non-NaN to use (1=most recent, 2=second-most, ...)
        
        Returns
        -------
        Factor
            Factor with NaN backfilled
        """
        from . import Factor
        
        self._validate_window(window)
        if k <= 0:
            raise ValueError("k must be a positive integer")

        result = self.data.copy()
        
        def backfill_func(s):
            if pd.isna(s.iloc[-1]):
                non_nan = s.dropna()
                if len(non_nan) >= k:
                    return non_nan.iloc[-k]
            return s.iloc[-1]

        result['factor'] = (result.groupby('symbol')['factor']
                           .rolling(window, min_periods=1)
                           .apply(backfill_func, raw=False)
                           .reset_index(level=0, drop=True))
        
        return Factor(result, f"ts_backfill({self.name},{window},{k})")
    
    def ts_decay_exp_window(self: 'FactorBase', window: int, factor: float = 1.0, nan: bool = True) -> 'FactorBase':
        """Exponentially weighted rolling average (recent heavier).
        
        Parameters
        ----------
        window : int
            Window size (periods)
        factor : float, default 1.0
            Decay factor (must be between 0 and 1)
        nan : bool, default True
            Whether to skip NaN values
        
        Returns
        -------
        Factor
            Exponentially weighted rolling average
        """
        from . import Factor
        
        self._validate_window(window)
        if not (0 < factor < 1):
            raise ValueError("Factor must be between 0 and 1 (exclusive)")
        
        result = self.data.copy()
        
        def vectorized_exp_decay(group):
            vals = group.values
            n = len(vals)
            out = np.full(n, np.nan)
            
            for i in range(n):
                window_vals = vals[max(0, i - window + 1):i + 1]
                
                if nan:
                    valid_mask = ~np.isnan(window_vals)
                    if not valid_mask.any():
                        continue
                    weights = np.array([factor ** j for j in range(len(window_vals) - 1, -1, -1)])
                    weighted_sum = np.dot(window_vals[valid_mask], weights[valid_mask])
                    weight_sum = weights[valid_mask].sum()
                else:
                    filled_vals = np.where(np.isnan(window_vals), 0.0, window_vals)
                    weights = np.array([factor ** j for j in range(len(window_vals) - 1, -1, -1)])
                    weighted_sum = np.dot(filled_vals, weights)
                    weight_sum = weights.sum()
                
                out[i] = weighted_sum / weight_sum if weight_sum > 0 else (np.nan if nan else 0.0)
            
            return pd.Series(out, index=group.index)
        
        result['factor'] = result.groupby('symbol', group_keys=False)['factor'].apply(vectorized_exp_decay)
        return Factor(result, f"ts_decay_exp_window({self.name},{window},{factor},{nan})")
    
    def ts_decay_linear(self: 'FactorBase', window: int, dense: bool = False) -> 'FactorBase':
        """Linearly weighted rolling average (recent heavier).
        
        Parameters
        ----------
        window : int
            Window size (periods)
        dense : bool, default False
            Whether to use dense weighting (skip NaN values)
        
        Returns
        -------
        Factor
            Linearly weighted rolling average
        """
        from . import Factor
        
        self._validate_window(window)
        
        result = self.data.copy()
        
        def vectorized_linear_decay(group):
            vals = group.values
            n = len(vals)
            out = np.full(n, np.nan)
            
            for i in range(n):
                window_vals = vals[max(0, i - window + 1):i + 1]
                
                if dense:
                    valid_mask = ~np.isnan(window_vals)
                    if not valid_mask.any():
                        continue
                    weights = np.arange(len(window_vals), 0, -1, dtype=np.float64)
                    weighted_sum = np.dot(window_vals[valid_mask], weights[valid_mask])
                    weight_sum = weights[valid_mask].sum()
                else:
                    filled_vals = np.where(np.isnan(window_vals), 0.0, window_vals)
                    if np.isnan(window_vals).all():
                        continue
                    weights = np.arange(len(window_vals), 0, -1, dtype=np.float64)
                    weighted_sum = np.dot(filled_vals, weights)
                    weight_sum = weights.sum()
                
                out[i] = weighted_sum / weight_sum if weight_sum > 0 else np.nan
            
            return pd.Series(out, index=group.index)
        
        result['factor'] = result.groupby('symbol', group_keys=False)['factor'].apply(vectorized_linear_decay)
        return Factor(result, f"ts_decay_linear({self.name},{window},dense={dense})")
    
    def ts_step(self: 'FactorBase', start: int = 1) -> 'FactorBase':
        """Time step counter (1, 2, 3, ...) per symbol.
        
        Parameters
        ----------
        start : int, default 1
            Starting value
        
        Returns
        -------
        Factor
            Incrementing time steps per symbol
        """
        from . import Factor
        
        result = self.data.copy()
        result['factor'] = result.groupby('symbol').cumcount() + start
        return Factor(result, f"ts_step({start})")
    
    def ts_delay(self: 'FactorBase', window: int) -> 'FactorBase':
        """Lag factor by window periods.
        
        Parameters
        ----------
        window : int
            Lag periods
        
        Returns
        -------
        Factor
            Lagged factor
        """
        from . import Factor
        
        result = self.data.copy()
        result['factor'] = result.groupby('symbol')['factor'].shift(window)
        return Factor(result, f"ts_delay({self.name},{window})")
        
    def ts_delta(self: 'FactorBase', window: int) -> 'FactorBase':
        """Difference between current and lagged value.
        
        Parameters
        ----------
        window : int
            Lag periods
        
        Returns
        -------
        Factor
            First difference (x - lag(x, window))
        """
        from . import Factor
        
        result = self.data.copy()
        result['factor'] = result.groupby('symbol')['factor'].diff(window)
        return Factor(result, f"ts_delta({self.name},{window})")

    def ts_corr(self: 'FactorBase', other: 'FactorBase', window: int) -> 'FactorBase':
        """Rolling Pearson correlation over window.
        
        Parameters
        ----------
        other : Factor
            Second factor for correlation
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            Rolling correlation, NaN if window incomplete
        """
        from . import Factor
        
        self._validate_window(window)
        self._validate_factor(other, "ts_corr")
        
        merged = pd.merge(self.data, other.data,
                         on=['timestamp', 'symbol'],
                         suffixes=('_x', '_y'))
        
        if merged.empty:
            raise ValueError("No common data between factors")
        
        def safe_corr(group):
            x = group['factor_x']
            y = group['factor_y']
            
            valid_mask = x.notna() & y.notna()
            if valid_mask.sum() < 2:
                return pd.Series(np.nan, index=group.index)
            
            if x[valid_mask].std() == 0 or y[valid_mask].std() == 0:
                return pd.Series(np.nan, index=group.index)
            
            corr_result = group[['factor_x', 'factor_y']].rolling(
                window, min_periods=window
            ).corr().iloc[0::2, -1]
            
            return corr_result
        
        result = merged.copy()
        result['factor'] = merged.groupby('symbol', group_keys=False).apply(
            safe_corr, include_groups=False
        ).values
        
        result = result[['timestamp', 'symbol', 'factor']]
        return Factor(result, f"ts_corr({self.name},{other.name},{window})")
    
    def ts_covariance(self: 'FactorBase', other: 'FactorBase', window: int) -> 'FactorBase':
        """Rolling covariance over window.
        
        Parameters
        ----------
        other : Factor
            Second factor for covariance
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            Rolling covariance, NaN if window incomplete
        """
        from . import Factor
        
        self._validate_window(window)
        self._validate_factor(other, "ts_covariance")
        
        merged = pd.merge(self.data, other.data,
                         on=['timestamp', 'symbol'],
                         suffixes=('_x', '_y'))
        
        if merged.empty:
            raise ValueError("No common data between factors")
        
        def safe_cov(group):
            x = group['factor_x']
            y = group['factor_y']
            
            valid_mask = x.notna() & y.notna()
            if valid_mask.sum() < 2:
                return pd.Series(np.nan, index=group.index)
            
            cov_result = group[['factor_x', 'factor_y']].rolling(
                window, min_periods=window
            ).cov().iloc[0::2, -1]
            
            return cov_result
        
        result = merged.copy()
        result['factor'] = merged.groupby('symbol', group_keys=False).apply(
            safe_cov, include_groups=False
        ).values

        result = result[['timestamp', 'symbol', 'factor']]
        return Factor(result, f"ts_covariance({self.name},{other.name},{window})")

    def ts_regression(self: 'FactorBase', x_factor, window: int, lag: int = 0, rettype: int = 0) -> 'FactorBase':
        """Rolling OLS regression with multiple return types.
        
        Parameters
        ----------
        x_factor : Factor or List[Factor]
            Independent variable(s)
        window : int
            Window size (periods)
        lag : int, default 0
            Lag applied to x_factor
        rettype : int, default 0
            Return type: 0=residual, 1=alpha, 2=beta,
            3=predicted, 4=SSE, 5=SST, 6=R-squared, 7=MSE,
            8=SE(beta), 9=SE(alpha), 100+=beta_i
        
        Returns
        -------
        Factor
            Regression result of specified type
        """
        from . import Factor
        from .base import FactorBase
        
        self._validate_window(window)
        if lag < 0:
            raise ValueError("Lag must be non-negative")
        
        is_multi = isinstance(x_factor, list)
        if is_multi:
            if not all(isinstance(f, FactorBase) for f in x_factor):
                raise TypeError("x_factor list must contain only Factor objects")
            x_factors = x_factor
        else:
            if not isinstance(x_factor, FactorBase):
                raise TypeError("x_factor must be Factor or list of Factors")
            if rettype >= 100:
                raise ValueError("rettype 100+ only for multivariate mode")
            x_factors = [x_factor]
        
        y_data = self.data.rename(columns={'factor': 'y'})
        merged = y_data.copy()
        
        for i, xf in enumerate(x_factors):
            x_data = xf.data.rename(columns={'factor': f'x{i}'})
            if lag > 0:
                x_data[f'x{i}'] = x_data.groupby('symbol')[f'x{i}'].shift(lag)
            merged = pd.merge(merged, x_data, on=['timestamp', 'symbol'])
        
        if merged.empty:
            raise ValueError("No common data for regression")
        
        merged = merged.sort_values(['symbol', 'timestamp'])
        x_cols = [f'x{i}' for i in range(len(x_factors))]
        
        def rolling_regression(group):
            y = group['y'].values
            X = group[x_cols].values
            n, m = len(y), len(x_cols)
            results = np.full(n, np.nan)
            
            for i in range(window - 1, n):
                y_win = y[i-window+1:i+1]
                X_win = X[i-window+1:i+1]
                
                valid = ~(np.isnan(y_win) | np.isnan(X_win).any(axis=1))
                if valid.sum() < m + 2:
                    continue
                
                y_v = y_win[valid]
                X_v = X_win[valid]
                X_mat = np.column_stack([np.ones(len(y_v)), X_v])
                
                try:
                    if np.linalg.cond(X_mat.T @ X_mat) > 1e10:
                        continue
                    
                    params = np.linalg.lstsq(X_mat, y_v, rcond=None)[0]
                    residuals = y_v - X_mat @ params
                    SSE = (residuals ** 2).sum()
                    SST = ((y_v - y_v.mean()) ** 2).sum()
                    
                    if rettype == 0:
                        results[i] = residuals[-1]
                    elif rettype == 1:
                        results[i] = params[0]
                    elif rettype == 2:
                        results[i] = params[1]
                    elif rettype == 3:
                        results[i] = (X_mat @ params)[-1]
                    elif rettype == 4:
                        results[i] = SSE
                    elif rettype == 5:
                        results[i] = SST
                    elif rettype == 6:
                        results[i] = 1 - SSE / SST if SST > 0 else np.nan
                    elif rettype == 7:
                        df = len(y_v) - m - 1
                        results[i] = SSE / df if df > 0 else np.nan
                    elif rettype in [8, 9]:
                        df = len(y_v) - m - 1
                        MSE = SSE / df if df > 0 else 0
                        if MSE > 0:
                            var_covar = MSE * np.linalg.inv(X_mat.T @ X_mat)
                            idx = 2 if rettype == 8 else 0
                            results[i] = np.sqrt(var_covar[idx, idx])
                    elif rettype >= 100:
                        idx = rettype - 100 + 1
                        if idx < len(params):
                            results[i] = params[idx]
                except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                    continue
            
            return pd.Series(results, index=group.index)
        
        merged['factor'] = merged.groupby('symbol', group_keys=False).apply(
            rolling_regression, include_groups=False
        ).values
        
        result = merged[['timestamp', 'symbol', 'factor']]
        x_name = ','.join([f.name for f in x_factors]) if is_multi else x_factors[0].name
        return Factor(result, name=f"ts_regression({self.name},{x_name},{window},lag={lag},rettype={rettype})")

    def ts_cv(self: 'FactorBase', window: int) -> 'FactorBase':
        """Rolling coefficient of variation.
        
        Parameters
        ----------
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            Rolling CV: std / abs(mean)
        """
        from . import Factor
        
        self._validate_window(window)
        
        result = self.data.copy()
        
        def cv_vectorized(group):
            vals = group.values
            n = len(vals)
            out = np.full(n, np.nan)
            
            for i in range(window - 1, n):
                w = vals[i - window + 1:i + 1]
                
                if np.isnan(w).any() or len(w) < window:
                    continue
                
                mean_val = np.mean(w)
                std_val = np.std(w, ddof=1)
                
                out[i] = std_val / (abs(mean_val) + 1e-10)
            
            return pd.Series(out, index=group.index)
        
        result['factor'] = result.groupby('symbol', group_keys=False)['factor'].apply(cv_vectorized)
        result['factor'] = self._replace_inf(result['factor'])
        return Factor(result, f"ts_cv({self.name},{window})")

    def ts_jumpiness(self: 'FactorBase', window: int) -> 'FactorBase':
        """Measure of path jumpiness (total abs diff / range).
        
        Parameters
        ----------
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            Jumpiness measure
        """
        from . import Factor
        
        self._validate_window(window)
        diff = self.ts_delta(1).abs()
        total_jump = diff.ts_sum(window)
        range_val = self.ts_max(window) - self.ts_min(window)
        result = total_jump / (range_val + 1e-10)
        result.data['factor'] = self._replace_inf(result.data['factor'])
        return Factor(result.data, f"ts_jumpiness({self.name},{window})")

    def ts_trend_strength(self: 'FactorBase', window: int) -> 'FactorBase':
        """Linear trend strength (R-squared of regression on time).
        
        Parameters
        ----------
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            R-squared from linear regression
        """
        from . import Factor
        
        self._validate_window(window)
        time_step = self.ts_step()
        result = self.ts_regression(time_step, window, rettype=6)
        return Factor(result.data, f"ts_trend_strength({self.name},{window})")

    def ts_vr(self: 'FactorBase', window: int, k: int = 2) -> 'FactorBase':
        """Variance ratio test statistic.
        
        Parameters
        ----------
        window : int
            Window size (periods)
        k : int, default 2
            Ratio of periods
        
        Returns
        -------
        Factor
            Variance ratio: Var(k-diff) / (k * Var(1-diff))
        """
        from . import Factor
        
        self._validate_window(window)
        if k <= 0:
            raise ValueError("k must be positive")
        k_diff = self.ts_delta(k)
        one_diff = self.ts_delta(1)
        var_k = k_diff.ts_std_dev(window) ** 2
        var_1 = one_diff.ts_std_dev(window) ** 2
        result = var_k / (k * var_1 + 1e-10)
        result.data['factor'] = self._replace_inf(result.data['factor'])
        return Factor(result.data, f"ts_vr({self.name},{window},{k})")

    def ts_autocorr(self: 'FactorBase', window: int, lag: int = 1) -> 'FactorBase':
        """Rolling autocorrelation.
        
        Parameters
        ----------
        window : int
            Window size (periods)
        lag : int, default 1
            Lag for autocorrelation
        
        Returns
        -------
        Factor
            Autocorrelation at specified lag
        """
        from . import Factor
        
        self._validate_window(window)
        if lag <= 0:
            raise ValueError("lag must be positive")
        lagged = self.ts_delay(lag)
        result = self.ts_corr(lagged, window)
        return Factor(result.data, f"ts_autocorr({self.name},{window},{lag})")

    def ts_reversal_count(self: 'FactorBase', window: int) -> 'FactorBase':
        """Count of sign reversals in rolling window.
        
        Parameters
        ----------
        window : int
            Window size (periods)
        
        Returns
        -------
        Factor
            Fraction of sign reversals
        """
        from . import Factor
        
        self._validate_window(window)
        
        def count_reversals(s):
            if len(s) < 3:
                return np.nan
            diff = np.diff(s)
            if len(diff) < 2:
                return np.nan
            valid_diff = diff[~np.isnan(diff)]
            if len(valid_diff) < 2:
                return np.nan
            sign_changes = ((valid_diff[1:] * valid_diff[:-1]) < 0).sum()
            return sign_changes / (len(valid_diff) - 1)
        
        result = self.data.copy()
        result['factor'] = (result.groupby('symbol')['factor']
                           .rolling(window, min_periods=3)
                           .apply(count_reversals, raw=True)
                           .reset_index(level=0, drop=True))
        
        return Factor(result, f"ts_reversal_count({self.name},{window})")
