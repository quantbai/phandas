"""
Core factor computation engine for quantitative analysis.

Provides efficient factor matrix operations with pandas-like API.

The Factor class supports:
- Time-series, cross-sectional, and mathematical operations
- Neutralization (group, vector, regression)
- Method chaining
- Boolean operations (including ~ NOT operator)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Union, Optional, Callable, List, Dict
from scipy.stats import rankdata, norm, uniform, cauchy


class Factor:
    """
    Professional factor matrix for quantitative analysis.
    
    Internal format: 3-column DataFrame [timestamp, symbol, factor].
    Optimized for fast time-series operations.
    """
    
    
    def __init__(self, data: Union[pd.DataFrame, str], name: Optional[str] = None):
        """
        Initialize factor matrix.
        
        Parameters
        ----------
        data : DataFrame or str
            Factor data or CSV path with [timestamp, symbol, factor] columns.
        name : str, optional
            Factor name for identification.
        """
        if isinstance(data, str):
            df = pd.read_csv(data, parse_dates=['timestamp'])
        else:
            df = data.copy()
        
        # Standardize data format to 3 columns: [timestamp, symbol, factor]
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        
        if len(df.columns) == 3 and 'factor' not in df.columns:
            df.columns = ['timestamp', 'symbol', 'factor']
        elif 'factor' not in df.columns:
            factor_cols = [col for col in df.columns 
                          if col not in ['timestamp', 'symbol']]
            if not factor_cols:
                raise ValueError("No factor column found")
            df = df[['timestamp', 'symbol', factor_cols[0]]]
            df.columns = ['timestamp', 'symbol', 'factor']
        
        self.data = df[['timestamp', 'symbol', 'factor']].copy()
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        self.name = name or 'factor'

    def rank(self) -> 'Factor':
        """Cross-sectional rank within each timestamp."""
        result = self.data.copy()
        result['factor'] = pd.to_numeric(result['factor'], errors='coerce')
        
        if result['factor'].isna().all():
            raise ValueError("All factor values are NaN")
        
        def safe_rank(group):
            valid_mask = group.notna()
            if valid_mask.sum() == 0:
                return group
            
            ranked = group.copy()
            ranked[valid_mask] = group[valid_mask].rank(method='min', pct=True)
            return ranked
        
        result['factor'] = result.groupby('timestamp')['factor'].transform(safe_rank)
        return Factor(result, f"rank({self.name})")

    def ts_rank(self, window: int) -> 'Factor':
        """Rolling time-series rank within window."""
        if window <= 0:
            raise ValueError("Window must be positive")
        
        def safe_ts_rank(x: pd.Series) -> float:
            if x.notna().sum() < window:
                return np.nan
            ranks = rankdata(x.to_numpy(), method='min')
            return ranks[-1] / len(ranks)
        
        result = self._apply_rolling(safe_ts_rank, window)
        return Factor(result, f"ts_rank({self.name},{window})")
    
    def ts_sum(self, window: int) -> 'Factor':
        """Rolling sum over window."""
        result = self._apply_rolling('sum', window)
        return Factor(result, f"ts_sum({self.name},{window})")

    def ts_product(self, window: int) -> 'Factor':
        """Returns product of x for the past d days."""
        if window <= 0:
            raise ValueError("Window must be positive")
        result = self._apply_rolling(lambda x: x.prod(), window)
        return Factor(result, f"ts_product({self.name},{window})")

    def ts_quantile(self, window: int, driver: str = "gaussian") -> 'Factor':
        """Calculates ts_rank and applies an inverse cumulative density function."""
        if window <= 0:
            raise ValueError("Window must be positive")

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

    def ts_count_nans(self, window: int) -> 'Factor':
        """Returns the number of NaN values in x for the past d days."""
        if window <= 0:
            raise ValueError("Window must be positive")
        result = self._apply_rolling(lambda x: x.isna().sum(), window)
        return Factor(result, f"ts_count_nans({self.name},{window})")
        
    def ts_mean(self, window: int) -> 'Factor':
        """Returns average value of x for the past d days."""
        result = self._apply_rolling('mean', window)
        return Factor(result, f"ts_mean({self.name},{window})")
        
    def ts_median(self, window: int) -> 'Factor':
        """Returns median value of x for the past d days."""
        if window <= 0:
            raise ValueError("Window must be positive")
        result = self._apply_rolling(lambda x: x.median() if not x.isna().all() else np.nan, window)
        return Factor(result, f"ts_median({self.name},{window})")
    
    def ts_std_dev(self, window: int) -> 'Factor':
        """Returns standard deviation of x for the past d days."""
        result = self._apply_rolling('std', window)
        return Factor(result, f"ts_std_dev({self.name},{window})")
    
    def ts_min(self, window: int) -> 'Factor':
        """Rolling minimum over window."""
        result = self._apply_rolling('min', window)
        return Factor(result, f"ts_min({self.name},{window})")
    
    def ts_max(self, window: int) -> 'Factor':
        """Rolling maximum over window."""
        result = self._apply_rolling('max', window)
        return Factor(result, f"ts_max({self.name},{window})")
    
    def ts_arg_max(self, window: int) -> 'Factor':
        """Returns relative index of max value in time series for past d days."""
        if window <= 0:
            raise ValueError("Window must be positive")
        
        def safe_arg_max(s):
            if s.isna().all():
                return np.nan
            return (len(s) - 1) - s.argmax()
        
        result = self._apply_rolling(safe_arg_max, window)
        return Factor(result, f"ts_arg_max({self.name},{window})")
        
    def ts_arg_min(self, window: int) -> 'Factor':
        """Returns relative index of min value in time series for past d days."""
        if window <= 0:
            raise ValueError("Window must be positive")
        
        def safe_arg_min(s):
            if s.isna().all():
                return np.nan
            return (len(s) - 1) - s.argmin()
        
        result = self._apply_rolling(safe_arg_min, window)
        return Factor(result, f"ts_arg_min({self.name},{window})")
    
    def ts_av_diff(self, window: int) -> 'Factor':
        """Returns x - ts_mean(x, d), ignoring NaNs during mean computation."""
        if window <= 0:
            raise ValueError("Window must be positive")
        mean_factor = self.ts_mean(window)
        result = self.subtract(mean_factor)
        return Factor(result.data, f"ts_av_diff({self.name},{window})")

    def ts_scale(self, window: int, constant: float = 0) -> 'Factor':
        """Returns (x - ts_min(x, d)) / (ts_max(x, d) - ts_min(x, d)) + constant."""
        if window <= 0:
            raise ValueError("Window must be positive")

        min_factor = self.ts_min(window)
        max_factor = self.ts_max(window)
        
        result = (self - min_factor) / (max_factor - min_factor)
        result.data['factor'] = result.data['factor'].replace([np.inf, -np.inf], np.nan)
        result = result + constant

        return Factor(result.data, f"ts_scale({self.name},{window},{constant})")
    
    def ts_zscore(self, window: int) -> 'Factor':
        """Returns Z-score: (x - ts_mean(x,d)) / ts_std_dev(x,d)."""
        if window <= 0:
            raise ValueError("Window must be positive")
        
        mean = self.ts_mean(window)
        std = self.ts_std_dev(window)
        
        result = (self - mean) / std
        result.data['factor'] = result.data['factor'].replace([np.inf, -np.inf], np.nan)
        
        return Factor(result.data, f"ts_zscore({self.name},{window})")

    def quantile(self, driver: str = "gaussian", sigma: float = 1.0) -> 'Factor':
        """Cross-sectional quantile transformation with specified driver and scale."""
        valid_drivers = {
            "gaussian": norm.ppf,
            "uniform": uniform.ppf,
            "cauchy": cauchy.ppf,
        }

        if driver not in valid_drivers:
            raise ValueError(f"Invalid driver: {driver}. Must be one of {list(valid_drivers.keys())}")

        ppf_func = valid_drivers[driver]
        result = self.data.copy()

        def apply_quantile(group: pd.Series):
            ranked_group = group.rank(method='average', pct=True)
            
            if ranked_group.isna().all() or len(ranked_group.dropna()) < 2:
                return pd.Series(np.nan, index=group.index)

            N = len(ranked_group.dropna())
            if N <= 1:
                return pd.Series(np.nan, index=group.index)
            
            epsilon = 1e-6
            shifted_rank = 1/N + ranked_group.clip(lower=epsilon, upper=1-epsilon) * (1 - 2/N)
            shifted_rank = shifted_rank.clip(lower=epsilon, upper=1-epsilon)

            transformed_group = shifted_rank.apply(ppf_func)
            if sigma != 1.0:
                transformed_group *= sigma
            
            return transformed_group

        result['factor'] = result.groupby('timestamp')['factor'].transform(apply_quantile)
        return Factor(result, f"quantile({self.name},driver={driver},sigma={sigma})")

    def scale(self, scale: float = 1.0, longscale: float = -1.0, shortscale: float = -1.0) -> 'Factor':
        """Scales input to booksize, with optional separate scaling for long and short positions."""
        result = self.data.copy()

        def apply_scale(group: pd.Series):
            if longscale != -1.0 or shortscale != -1.0:
                scaled_group = group.copy()
                
                long_mask = group > 0
                short_mask = group < 0
                
                if long_mask.any() and longscale > 0:
                    long_abs_sum = group[long_mask].abs().sum()
                    if long_abs_sum > 0:
                        scaled_group[long_mask] = (group[long_mask] / long_abs_sum) * longscale
                
                if short_mask.any() and shortscale > 0:
                    short_abs_sum = group[short_mask].abs().sum()
                    if short_abs_sum > 0:
                        scaled_group[short_mask] = (group[short_mask] / short_abs_sum) * shortscale
                
                return scaled_group
            else:
                abs_sum = group.abs().sum()
                if abs_sum == 0:
                    return group
                return (group / abs_sum) * scale

        result['factor'] = result.groupby('timestamp')['factor'].transform(apply_scale)
        return Factor(result, f"scale({self.name},scale={scale},longscale={longscale},shortscale={shortscale})")

    def zscore(self) -> 'Factor':
        """Computes the cross-sectional Z-score of the factor."""
        return self.normalize(useStd=True)

    def group_neutralize(self, group_data: 'Factor') -> 'Factor':
        """Neutralizes the factor against specified groups (e.g., industry)."""
        if not isinstance(group_data, Factor):
            raise TypeError("group_data must be a Factor object.")

        merged = pd.merge(self.data, group_data.data, 
                         on=['timestamp', 'symbol'],
                         suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data between factor and group data.")

        merged['factor'] = merged.groupby(['timestamp', 'factor_group'])['factor'].transform(
            lambda x: x - x.mean()
        )
        
        result_data = merged[['timestamp', 'symbol', 'factor']]
        return Factor(result_data, name=f"group_neutralize({self.name},{group_data.name})")

    def vector_neut(self, other: 'Factor') -> 'Factor':
        """
        Performs cross-sectional vector neutralization (orthogonalization)
        of the current factor against another factor.

        This projects the current factor (`self`) onto the neutralization factor 
        (`other`) within each timestamp and returns the residual (the component 
        orthogonal to `other`). This is equivalent to removing the linear
        component of `other` from `self`.

        Formula: Residual = self - (self dot other / other dot other) * other

        Parameters
        ----------
        other : Factor
            The factor to neutralize against (the vector to project onto).

        Returns
        -------
        Factor
            A new Factor object representing the neutralized factor.
        
        Notes
        -----
        - Operates cross-sectionally (within each timestamp).
        - Handles NaN values and ensures numerical stability by checking for 
          constant neutralization vectors or near-zero norms.
        """
        if not isinstance(other, Factor):
            raise TypeError("other must be a Factor object.")

        merged = pd.merge(self.data, other.data,
                        on=['timestamp', 'symbol'],
                        suffixes=('_target', '_neut'))

        if merged.empty:
            raise ValueError("No common data for vector neutralization.")

        def neutralize_single_date(group):
            """
            Perform vector neutralization for a single timestamp.
            
            Handles edge cases:
            1. NaN values - only use valid pairs
            2. Constant neutralization factor - return original
            3. Near-zero norm - numerical stability
            4. Insufficient observations - need at least 2 valid pairs
            """
            x = group['factor_target'].values
            y = group['factor_neut'].values
            
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            n_valid = valid_mask.sum()
            
            if n_valid < 2:
                return pd.Series(x, index=group.index)
            
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            y_std = np.std(y_valid)
            y_mean = np.abs(np.mean(y_valid))
            
            if y_std < max(1e-10, 1e-4 * y_mean):
                return pd.Series(x, index=group.index)
            
            y_norm_sq = np.dot(y_valid, y_valid)
            
            if y_norm_sq < 1e-10:
                return pd.Series(x, index=group.index)
            
            xy_dot = np.dot(x_valid, y_valid)
            projection_coef = xy_dot / y_norm_sq
            
            result = x.copy()
            
            result[valid_mask] = x_valid - projection_coef * y_valid
            
            return pd.Series(result, index=group.index)

        merged['factor'] = merged.groupby('timestamp', group_keys=False).apply(
            neutralize_single_date, include_groups=False
        )
        
        result_data = merged[['timestamp', 'symbol', 'factor']]
        return Factor(result_data, name=f"vector_neut({self.name},{other.name})")

    def regression_neut(self, neut_factors: Union['Factor', List['Factor']]) -> 'Factor':
        """Neutralizes the factor against one or more factors using OLS regression."""
        if isinstance(neut_factors, Factor):
            neut_factors = [neut_factors]
        if not all(isinstance(f, Factor) for f in neut_factors):
            raise TypeError("neut_factors must be a Factor or a list of Factor objects.")

        merged = self.data.rename(columns={'factor': self.name or 'target'})
        neut_names = []
        
        for i, f in enumerate(neut_factors):
            neut_name = f.name or f'neut_{i}'
            neut_names.append(neut_name)
            merged = pd.merge(merged, f.data.rename(columns={'factor': neut_name}),
                              on=['timestamp', 'symbol'], how='inner')

        if merged.empty:
            raise ValueError("No common data for regression neutralization.")
        
        def get_residuals(group):
            Y = group[self.name or 'target']
            X = group[neut_names]
            
            valid_idx = Y.notna() & X.notna().all(axis=1)
            if valid_idx.sum() < 2:
                group['factor'] = np.nan
                return group
                
            Y_valid = Y[valid_idx]
            X_valid = X[valid_idx]

            X_const = sm.add_constant(X_valid)
            if np.linalg.cond(X_const) > 1e10:
                group['factor'] = np.nan
                return group

            model = sm.OLS(Y_valid, X_const).fit()
            group['factor'] = np.nan
            group.loc[valid_idx, 'factor'] = model.resid
            
            return group

        result = merged.groupby('timestamp', group_keys=False).apply(get_residuals)
        result_data = result[['timestamp', 'symbol', 'factor']]

        neut_factors_str = ",".join([f.name for f in neut_factors])
        return Factor(result_data, name=f"regression_neut({self.name},[{neut_factors_str}])")

    def normalize(self, useStd: bool = False, limit: float = 0.0) -> 'Factor':
        """Cross-sectional normalization: (x - mean) / std (optional), then limit (optional)."""
        result = self.data.copy()

        def apply_normalize(group: pd.Series):
            mean_val = group.mean()
            normalized_group = group - mean_val

            if useStd:
                std_val = group.std()
                if std_val == 0 or pd.isna(std_val):
                    return pd.Series(np.nan, index=group.index)
                normalized_group /= std_val
            
            if limit != 0.0:
                normalized_group = normalized_group.clip(lower=-limit, upper=limit)
            
            return normalized_group

        result['factor'] = result.groupby('timestamp')['factor'].transform(apply_normalize)
        return Factor(result, f"normalize({self.name},useStd={useStd},limit={limit})")

    def ts_backfill(self, window: int, k: int = 1) -> 'Factor':
        """Backfills NaN values with the k-th most recent non-NaN value."""
        if window <= 0:
            raise ValueError("Window must be positive")
        if k <= 0:
            raise ValueError("k must be a positive integer")

        def backfill_func(s):
            if pd.isna(s.iloc[-1]):
                non_nan = s.dropna()
                if len(non_nan) >= k:
                    return non_nan.iloc[-k]
            return s.iloc[-1]

        result = self._apply_rolling(backfill_func, window)
        return Factor(result, f"ts_backfill({self.name},{window},{k})")
    
    def ts_decay_exp_window(self, window: int, factor: float = 1.0, nan: bool = True) -> 'Factor':
        """Returns exponential decay weighted average over rolling window."""
        if window <= 0:
            raise ValueError("Window must be positive")
        if not (0 < factor < 1):
            raise ValueError("Factor must be between 0 and 1 (exclusive)")
        
        def decay_func(s):
            if s.isna().all():
                return np.nan if nan else 0.0
            
            valid_s = s.dropna() if nan else s.fillna(0)
            
            if len(valid_s) == 0:
                 return np.nan if nan else 0.0
            
            weights = np.array([factor**i for i in range(len(valid_s))])[::-1]
            weight_sum = weights.sum()
            
            if weight_sum == 0:
                return np.nan if nan else 0.0
            
            return (valid_s * weights).sum() / weight_sum
        
        result = self._apply_rolling(decay_func, window)
        return Factor(result, f"ts_decay_exp_window({self.name},{window},{factor},{nan})")
    
    def ts_decay_linear(self, window: int, dense: bool = False) -> 'Factor':
        """Returns linear decay weighted average over rolling window."""
        if window <= 0:
            raise ValueError("Window must be positive")
            
        def linear_decay_func(s):
            current_s = s.copy()
            if not dense:
                current_s = current_s.fillna(0)
            
            if current_s.isna().all():
                return np.nan
                
            weights = np.arange(len(current_s), 0, -1)
            
            if dense:
                valid_mask = current_s.notna()
                if not valid_mask.any():
                    return np.nan
                weighted_sum = (current_s[valid_mask] * weights[valid_mask]).sum()
                weight_sum = weights[valid_mask].sum()
            else:
                weighted_sum = (current_s * weights).sum()
                weight_sum = weights.sum()
                
            return weighted_sum / weight_sum if weight_sum > 0 else np.nan
            
        result = self._apply_rolling(linear_decay_func, window)
        return Factor(result, f"ts_decay_linear({self.name},{window},dense={dense})")
    
    def ts_delay(self, window: int) -> 'Factor':
        """Returns x value d days ago."""
        result = self.data.copy()
        result['factor'] = result.groupby('symbol')['factor'].shift(window)
        return Factor(result, f"ts_delay({self.name},{window})")
        
    def ts_delta(self, window: int) -> 'Factor':
        """Returns x - ts_delay(x, d)."""
        result = self.data.copy()
        result['factor'] = result.groupby('symbol')['factor'].diff(window)
        return Factor(result, f"ts_delta({self.name},{window})")
        
    def returns(self, periods: int = 1) -> 'Factor':
        """Percentage returns over periods."""
        result = self.data.copy()
        result['factor'] = result.groupby('symbol')['factor'].pct_change(periods)
        return Factor(result, f"returns({self.name},{periods})")
    
    def add(self, other: Union['Factor', float], filter_nan: bool = False) -> 'Factor':
        """Addition with factor or scalar, with optional NaN filtering."""
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                              on=['timestamp', 'symbol'],
                              suffixes=('_x', '_y'), how='outer' if filter_nan else 'inner')
            
            if filter_nan:
                merged['factor_x'] = merged['factor_x'].fillna(0)
                merged['factor_y'] = merged['factor_y'].fillna(0)
            
            merged['factor'] = merged['factor_x'] + merged['factor_y']
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"({self.name}+{other.name})")
        else:
            result = self.data.copy()
            if filter_nan:
                result['factor'] = result['factor'].fillna(0) + other
            else:
                result['factor'] += other
            return Factor(result, f"({self.name}+{other})")
    
    def subtract(self, other: Union['Factor', float], filter_nan: bool = False) -> 'Factor':
        """Subtraction with factor or scalar, with optional NaN filtering."""
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                              on=['timestamp', 'symbol'],
                              suffixes=('_x', '_y'), how='outer' if filter_nan else 'inner')
            
            if filter_nan:
                merged['factor_x'] = merged['factor_x'].fillna(0)
                merged['factor_y'] = merged['factor_y'].fillna(0)
            
            merged['factor'] = merged['factor_x'] - merged['factor_y']
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"({self.name}-{other.name})")
        else:
            result = self.data.copy()
            if filter_nan:
                result['factor'] = result['factor'].fillna(0) - other
            else:
                result['factor'] -= other
            return Factor(result, f"({self.name}-{other})")
    
    def multiply(self, other: Union['Factor', float], filter_nan: bool = False) -> 'Factor':
        """Multiplication with factor or scalar, with optional NaN filtering."""
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                              on=['timestamp', 'symbol'],
                              suffixes=('_x', '_y'), how='outer' if filter_nan else 'inner')
            
            if filter_nan:
                merged['factor_x'] = merged['factor_x'].fillna(1)
                merged['factor_y'] = merged['factor_y'].fillna(1)
            
            merged['factor'] = merged['factor_x'] * merged['factor_y']
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"({self.name}*{other.name})")
        else:
            result = self.data.copy()
            if filter_nan:
                result['factor'] = result['factor'].fillna(1) * other
            else:
                result['factor'] *= other
            return Factor(result, f"({self.name}*{other})")
    
    def log(self, base: Optional[float] = None) -> 'Factor':
        """
        Logarithm of factor with optional base.
        
        Safely handles non-positive values by returning NaN.
        Only computes log for positive values.
        """
        result = self.data.copy()
        
        if base is None:
            # 自然對數：只對正數計算
            result['factor'] = np.where(
                result['factor'] > 0,
                np.log(result['factor']),
                np.nan
            )
            return Factor(result, f"log({self.name})")
        else:
            # 指定底數：檢查底數有效性
            if base <= 0 or base == 1:
                raise ValueError(f"Invalid log base: {base}. Base must be positive and not equal to 1.")
            
            result['factor'] = np.where(
                result['factor'] > 0,
                np.log(result['factor']) / np.log(base),
                np.nan
            )
            return Factor(result, f"log({self.name},base={base})")
    
    def ln(self) -> 'Factor':
        """
        Natural logarithm of factor.
        
        Safely handles non-positive values by returning NaN.
        Equivalent to log() without base parameter.
        """
        return self.log()  # 重用 log 方法，保持一致性
    
    def sqrt(self) -> 'Factor':
        """
        Square root of factor values.
        
        Safely handles negative values by returning NaN.
        """
        result = self.data.copy()
        result['factor'] = np.where(
            result['factor'] >= 0,
            np.sqrt(result['factor']),
            np.nan
        )
        return Factor(result, f"sqrt({self.name})")
    
    def s_log_1p(self) -> 'Factor':
        """Confine factor values using sign(x) * log(1 + abs(x))."""
        result = self.data.copy()
        result['factor'] = np.sign(result['factor']) * np.log1p(np.abs(result['factor']))
        return Factor(result, f"s_log_1p({self.name})")
    
    def sign(self) -> 'Factor':
        """Returns the sign of factor values."""
        result = self.data.copy()
        result['factor'] = np.sign(result['factor'])
        return Factor(result, f"sign({self.name})")
    
    def signed_power(self, exponent: Union['Factor', float]) -> 'Factor':
        """
        x raised to the power of y preserving sign of x.
        
        Computes: sign(x) * |x|^y
        
        Safely handles edge cases and preserves the sign of the base.
        """
        if isinstance(exponent, Factor):
            merged = pd.merge(self.data, exponent.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            
            # 分離符号和絕對值
            sign = np.sign(merged['factor_x'])
            abs_val = np.abs(merged['factor_x'])
            
            # 計算 |x|^y，然後恢復符號
            with np.errstate(invalid='ignore', divide='ignore'):
                result_val = sign * (abs_val ** merged['factor_y'])
            
            # 清理 inf
            merged['factor'] = result_val.replace([np.inf, -np.inf], np.nan)
            
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"signed_power({self.name},{exponent.name})")
        else:
            result = self.data.copy()
            
            # 分離符號和絕對值
            sign = np.sign(result['factor'])
            abs_val = np.abs(result['factor'])
            
            # 計算 |x|^y，然後恢復符號
            with np.errstate(invalid='ignore', divide='ignore'):
                result_val = sign * (abs_val ** exponent)
            
            # 清理 inf
            result['factor'] = result_val.replace([np.inf, -np.inf], np.nan)
            
            return Factor(result, f"signed_power({self.name},{exponent})")
    
    def power(self, exponent: Union['Factor', float]) -> 'Factor':
        """
        Factor to the power of exponent: factor ** exponent.
        
        Safely handles edge cases like negative base with non-integer exponent.
        """
        if isinstance(exponent, Factor):
            merged = pd.merge(self.data, exponent.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            
            # 使用 numpy 的錯誤抑制，讓非法運算自然產生 NaN
            with np.errstate(invalid='ignore', divide='ignore'):
                merged['factor'] = merged['factor_x'] ** merged['factor_y']
            
            # 清理 inf（轉為 NaN）
            merged['factor'] = merged['factor'].replace([np.inf, -np.inf], np.nan)
            
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"({self.name}**{exponent.name})")
        else:
            result = self.data.copy()
            
            with np.errstate(invalid='ignore', divide='ignore'):
                result['factor'] = result['factor'] ** exponent
            
            # 清理 inf
            result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
            
            return Factor(result, f"({self.name}**{exponent})")
    
    def abs(self) -> 'Factor':
        """Absolute value of factor."""
        result = self.data.copy()
        result['factor'] = np.abs(result['factor'])
        return Factor(result, f"abs({self.name})")
    
    def inverse(self) -> 'Factor':
        """
        Inverse: 1 / factor.
        
        Safely handles division by zero by returning NaN.
        """
        result = self.data.copy()
        result['factor'] = np.where(
            result['factor'] != 0,
            1 / result['factor'],
            np.nan
        )
        return Factor(result, f"inverse({self.name})")

    def where(self, cond: 'Factor', other: Union['Factor', float] = np.nan) -> 'Factor':
        """
        Replace values where the condition is False.
        
        Parameters
        ----------
        cond : Factor
            Boolean condition factor (will be converted to bool)
        other : Factor or float, default np.nan
            Value to use where condition is False
            
        Returns
        -------
        Factor
            New factor with values replaced where condition is False
            
        Examples
        --------
        >>> # Keep only positive values, replace negative with 0
        >>> factor.where(factor > 0, 0)
        
        >>> # Keep only top 50% ranked values
        >>> factor.where(factor.rank() > 0.5, np.nan)
        
        Notes
        -----
        This follows pandas/numpy convention:
        - where(True) -> keep original value
        - where(False) -> replace with 'other'
        """
        if not isinstance(cond, Factor):
            raise TypeError("cond must be a Factor object.")

        merged = pd.merge(self.data, cond.data,
                         on=['timestamp', 'symbol'],
                         suffixes=('', '_cond'))
        
        cond_bool = merged['factor_cond'].fillna(False).astype(bool)
        
        if isinstance(other, Factor):
            merged = pd.merge(merged, other.data.rename(columns={'factor': 'factor_other'}),
                            on=['timestamp', 'symbol'])
            merged['factor'] = np.where(cond_bool, merged['factor'], merged['factor_other'])
        else:
            merged['factor'] = np.where(cond_bool, merged['factor'], other)
        
        result = merged[['timestamp', 'symbol', 'factor']]
        return Factor(result, name=f"where({self.name})")
    
    def __neg__(self) -> 'Factor':
        """Unary negation: -factor"""
        return self.multiply(-1)
    
    def __invert__(self) -> 'Factor':
        """
        Bitwise NOT operator: ~factor
        Converts factor to boolean and inverts (NOT operation).
        
        Returns
        -------
        Factor
            Factor with boolean NOT applied (True->False, False->True)
            Non-zero values become 0, zero/NaN values become 1
            
        Examples
        --------
        >>> # Invert a boolean condition
        >>> positive = factor > 0  # 1 where positive, 0 otherwise
        >>> negative = ~positive   # 1 where negative or zero, 0 where positive
        
        >>> # Combine with other conditions
        >>> result = factor.where(~(factor.rank() < 0.2), 0)  # Keep only top 80%
        """
        result = self.data.copy()
        bool_values = result['factor'].fillna(0).astype(bool)
        result['factor'] = (~bool_values).astype(int)
        return Factor(result, f"~{self.name}")
    
    def reverse(self) -> 'Factor':
        """Reverse sign of factor: -factor"""
        return self.__neg__()
    
    def __add__(self, other: Union['Factor', float]) -> 'Factor':
        """Addition: factor + other"""
        return self.add(other)
    
    def __radd__(self, other: Union['Factor', float]) -> 'Factor':
        """Right addition: other + factor"""
        return self.add(other)
    
    def __abs__(self) -> 'Factor':
        """Absolute value: abs(factor)"""
        return self.abs()
    
    def __sub__(self, other: Union['Factor', float]) -> 'Factor':
        """Subtraction: factor - other"""
        return self.subtract(other)
    
    def __rsub__(self, other: Union['Factor', float]) -> 'Factor':
        """Right subtraction: other - factor"""
        if isinstance(other, Factor):
            return other.subtract(self)
        else:
            result = self.data.copy()
            result['factor'] = other - result['factor']
            return Factor(result, f"({other}-{self.name})")
    
    def __mul__(self, other: Union['Factor', float]) -> 'Factor':
        """Multiplication: factor * other"""
        return self.multiply(other)
    
    def __rmul__(self, other: Union['Factor', float]) -> 'Factor':
        """Right multiplication: other * factor"""
        return self.multiply(other)
    
    def __pow__(self, other: Union['Factor', float]) -> 'Factor':
        """Power: factor ** other (uses the safe power method)."""
        return self.power(other)
    
    def __rpow__(self, other: Union['Factor', float]) -> 'Factor':
        """
        Right power: other ** factor.
        
        Safely handles edge cases.
        """
        if isinstance(other, Factor):
            return other.power(self)
        else:
            result = self.data.copy()
            
            with np.errstate(invalid='ignore', divide='ignore'):
                result['factor'] = other ** result['factor']
            
            result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
            
            return Factor(result, f"({other}**{self.name})")
    
    def __truediv__(self, other: Union['Factor', float]) -> 'Factor':
        """
        Division: factor / other.
        
        Safely handles division by zero by returning NaN.
        """
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            
            # 安全除法：除數為0時返回 NaN
            merged['factor'] = np.where(
                merged['factor_y'] != 0,
                merged['factor_x'] / merged['factor_y'],
                np.nan
            )
            
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"({self.name}/{other.name})")
        else:
            result = self.data.copy()
            
            if other == 0:
                # 除以標量0：全部返回 NaN
                result['factor'] = np.nan
            else:
                result['factor'] = result['factor'] / other
            
            return Factor(result, f"({self.name}/{other})")
    
    def __rtruediv__(self, other: Union['Factor', float]) -> 'Factor':
        """
        Right division: other / factor.
        
        Safely handles division by zero by returning NaN.
        """
        if isinstance(other, Factor):
            return other.__truediv__(self)
        else:
            result = self.data.copy()
            
            # 安全除法：被除數為0時返回 NaN
            result['factor'] = np.where(
                result['factor'] != 0,
                other / result['factor'],
                np.nan
            )
            
            return Factor(result, f"({other}/{self.name})")

    def __lt__(self, other: Union['Factor', float]) -> 'Factor':
        """Less than: factor < other"""
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            merged['factor'] = (merged['factor_x'] < merged['factor_y']).astype(int)
            result = merged[['timestamp', 'symbol', 'factor']]
        else:
            result = self.data.copy()
            result['factor'] = (result['factor'] < other).astype(int)
        return Factor(result, f"({self.name}<{getattr(other, 'name', other)})")

    def __le__(self, other: Union['Factor', float]) -> 'Factor':
        """Less than or equal to: factor <= other"""
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            merged['factor'] = (merged['factor_x'] <= merged['factor_y']).astype(int)
            result = merged[['timestamp', 'symbol', 'factor']]
        else:
            result = self.data.copy()
            result['factor'] = (result['factor'] <= other).astype(int)
        return Factor(result, f"({self.name}<={getattr(other, 'name', other)})")

    def __gt__(self, other: Union['Factor', float]) -> 'Factor':
        """Greater than: factor > other"""
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            merged['factor'] = (merged['factor_x'] > merged['factor_y']).astype(int)
            result = merged[['timestamp', 'symbol', 'factor']]
        else:
            result = self.data.copy()
            result['factor'] = (result['factor'] > other).astype(int)
        return Factor(result, f"({self.name}>{getattr(other, 'name', other)})")

    def __ge__(self, other: Union['Factor', float]) -> 'Factor':
        """Greater than or equal to: factor >= other"""
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            merged['factor'] = (merged['factor_x'] >= merged['factor_y']).astype(int)
            result = merged[['timestamp', 'symbol', 'factor']]
        else:
            result = self.data.copy()
            result['factor'] = (result['factor'] >= other).astype(int)
        return Factor(result, f"({self.name}>={getattr(other, 'name', other)})")

    def __eq__(self, other: Union['Factor', float]) -> 'Factor':
        """Equal to: factor == other"""
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            merged['factor'] = (merged['factor_x'] == merged['factor_y']).astype(int)
            result = merged[['timestamp', 'symbol', 'factor']]
        else:
            result = self.data.copy()
            result['factor'] = (result['factor'] == other).astype(int)
        return Factor(result, f"({self.name}=={getattr(other, 'name', other)})")

    def __ne__(self, other: Union['Factor', float]) -> 'Factor':
        """Not equal to: factor != other"""
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            merged['factor'] = (merged['factor_x'] != merged['factor_y']).astype(int)
            result = merged[['timestamp', 'symbol', 'factor']]
        else:
            result = self.data.copy()
            result['factor'] = (result['factor'] != other).astype(int)
        return Factor(result, f"({self.name}!={getattr(other, 'name', other)})")
    
    def maximum(self, other: Union['Factor', float]) -> 'Factor':
        """Element-wise maximum between this factor and another factor or scalar."""
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            merged['factor'] = np.maximum(merged['factor_x'], merged['factor_y'])
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"max({self.name},{other.name})")
        else:
            result = self.data.copy()
            result['factor'] = np.maximum(result['factor'], other)
            return Factor(result, f"max({self.name},{other})")
    
    def minimum(self, other: Union['Factor', float]) -> 'Factor':
        """Element-wise minimum between this factor and another factor or scalar."""
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            merged['factor'] = np.minimum(merged['factor_x'], merged['factor_y'])
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"min({self.name},{other.name})")
        else:
            result = self.data.copy()
            result['factor'] = np.minimum(result['factor'], other)
            return Factor(result, f"min({self.name},{other})")
    
    def ts_corr(self, other: 'Factor', window: int) -> 'Factor':
        """
        Returns Pearson correlation of two factors for the past d days.
        
        Handles edge cases:
        - Returns NaN when either series is constant (std=0)
        - Returns NaN when window has insufficient non-NaN data
        """
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if not isinstance(other, Factor):
            raise TypeError("Other must be a Factor object")
        
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
    
    def ts_covariance(self, other: 'Factor', window: int) -> 'Factor':
        """
        Returns covariance of self and other for the past d days.
        
        Handles edge cases:
        - Returns NaN when window has insufficient non-NaN data
        """
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if not isinstance(other, Factor):
            raise TypeError("Other must be a Factor object")
            
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

    def ts_regression(self, x_factor: 'Factor', window: int, lag: int = 0, rettype: int = 0) -> 'Factor':
        """
        Rolling time-series linear regression: Y = alpha + beta * X
        
        Parameters
        ----------
        x_factor : Factor
            Independent variable (X)
        window : int
            Rolling window size
        lag : int, default 0
            Lag to apply to X factor
        rettype : int, default 0
            Return type:
            0: Error Term (residual at current point)
            1: Alpha (y-intercept)
            2: Beta (slope)
            3: Y-estimate (predicted value at current point)
            4: SSE (Sum of Squares Error)
            5: SST (Sum of Squares Total)
            6: R-Square
            7: MSE (Mean Square Error)
            8: Std Error of Beta
            9: Std Error of Alpha
            
        Returns
        -------
        Factor
            Selected regression statistic
            
        Notes
        -----
        - Returns NaN when X is constant (no variation)
        - Returns NaN for insufficient data or multicollinearity
        """
        if window <= 0:
            raise ValueError("Window must be positive")
        if lag < 0:
            raise ValueError("Lag must be non-negative")
        if not 0 <= rettype <= 9:
            raise ValueError("rettype must be between 0 and 9")
        
        y_data = self.data.rename(columns={'factor': 'y'})
        x_data = x_factor.data.rename(columns={'factor': 'x'})

        if lag > 0:
            x_data['x'] = x_data.groupby('symbol')['x'].shift(lag)

        merged = pd.merge(y_data, x_data, on=['timestamp', 'symbol'])

        if merged.empty:
            raise ValueError("No common data for regression.")

        merged = merged.sort_values(['symbol', 'timestamp'])

        def rolling_regression(group):
            results = []
            
            for i in range(len(group)):
                if i < window - 1:
                    results.append(np.nan)
                    continue
                
                window_data = group.iloc[i-window+1:i+1]
                clean_data = window_data.dropna(subset=['y', 'x'])
                
                if len(clean_data) < 2:
                    results.append(np.nan)
                    continue
                
                Y = clean_data['y'].values
                X = clean_data['x'].values
                
                try:
                    # Check if X is constant
                    if X.std() == 0:
                        results.append(np.nan)
                        continue
                    
                    # Add constant
                    X_with_const = np.column_stack([np.ones(len(X)), X])
                    
                    # Check for multicollinearity
                    if np.linalg.cond(X_with_const) > 1e10:
                        results.append(np.nan)
                        continue
                    
                    # OLS calculation
                    XtX = X_with_const.T @ X_with_const
                    Xty = X_with_const.T @ Y
                    params = np.linalg.solve(XtX, Xty)
                    
                    alpha = params[0]
                    beta = params[1]
                    
                    # Predictions & residuals
                    y_pred = X_with_const @ params
                    residuals = Y - y_pred
                    
                    # Last window value
                    y_estimate = y_pred[-1]
                    error_term = residuals[-1]
                    
                    # Statistics
                    SSE = np.sum(residuals ** 2)
                    SST = np.sum((Y - Y.mean()) ** 2)
                    
                    R_squared = 1 - (SSE / SST) if SST > 0 else np.nan
                    df_resid = len(Y) - 2
                    MSE = SSE / df_resid if df_resid > 0 else np.nan
                    
                    # Standard errors
                    if MSE > 0:
                        var_covar = MSE * np.linalg.inv(XtX)
                        std_err_alpha = np.sqrt(var_covar[0, 0])
                        std_err_beta = np.sqrt(var_covar[1, 1])
                    else:
                        std_err_alpha = np.nan
                        std_err_beta = np.nan
                    
                    result_values = [
                        error_term,      # 0: Error Term
                        alpha,           # 1: y-intercept
                        beta,            # 2: slope
                        y_estimate,      # 3: y-estimate
                        SSE,             # 4: Sum of Squares Error
                        SST,             # 5: Sum of Squares Total
                        R_squared,       # 6: R-Square
                        MSE,             # 7: Mean Square Error
                        std_err_beta,    # 8: Std Error of Beta
                        std_err_alpha    # 9: Std Error of Alpha
                    ]
                    
                    results.append(result_values[rettype])
                    
                except Exception:
                    results.append(np.nan)
            
            return results

        merged['factor'] = merged.groupby('symbol', group_keys=False).apply(
            lambda g: pd.Series(rolling_regression(g), index=g.index),
            include_groups=False
        ).values

        result = merged[['timestamp', 'symbol', 'factor']]
        return Factor(result, name=f"ts_regression({self.name},{x_factor.name},{window},lag={lag},rettype={rettype})")

    def _apply_rolling(self, func: Union[str, Callable], window: int) -> pd.DataFrame:
        """Apply rolling function by symbol - OPTIMIZED for 3-column format."""
        result = self.data.copy()
        
        if isinstance(func, str):
            
            result['factor'] = (result.groupby('symbol')['factor']
                               .rolling(window, min_periods=window)
                               .agg(func)
                               .reset_index(level=0, drop=True))
        else:
            
            result['factor'] = (result.groupby('symbol')['factor']
                               .rolling(window, min_periods=window)
                               .apply(func, raw=False)
                               .reset_index(level=0, drop=True))
        
        return result
    
    # ==================== Data Access and Information ====================

    def to_weights(self, date: Optional[Union[str, pd.Timestamp]] = None) -> dict:
        """
        Convert factor to dollar-neutral portfolio weights.
        
        Returns demeaned and normalized weights where sum(abs(weights)) = 1.
        Multiply by portfolio value to get target dollar holdings.
        """
        if date is None:
            target_date = self.data['timestamp'].max()
        else:
            target_date = pd.to_datetime(date)
        
        date_data = self.data[self.data['timestamp'] == target_date]
        if date_data.empty:
            return {}
        
        factors = date_data.set_index('symbol')['factor'].dropna()
        if factors.empty:
            return {}
        
        demeaned = factors - factors.mean()
        abs_sum = np.abs(demeaned).sum()
        
        if abs_sum < 1e-10:
            return {}
        
        weights = demeaned / abs_sum
        return weights.to_dict()

    def to_csv(self, path: str) -> str:
        """Save to CSV file."""
        self.data.to_csv(path, index=False)
        return path
    
    def info(self) -> dict:
        """Get factor information."""
        return {
            'name': self.name,
            'shape': self.data.shape,
            'time_range': (self.data['timestamp'].min(), self.data['timestamp'].max()),
            'symbols': sorted(self.data['symbol'].unique()),
            'valid_ratio': self.data['factor'].notna().mean()
        }
    
    def __repr__(self):
        n_obs = len(self.data)
        n_symbols = self.data['symbol'].nunique()
        valid_ratio = self.data['factor'].notna().mean()
        time_range = f"{self.data['timestamp'].min().strftime('%Y-%m-%d')} to {self.data['timestamp'].max().strftime('%Y-%m-%d')}"
        return (f"Factor(name={self.name}, obs={n_obs}, symbols={n_symbols}, "
               f"valid={valid_ratio:.1%}, period={time_range})")
    
    def __str__(self):
        """User-friendly string representation."""
        n_symbols = self.data['symbol'].nunique()
        return f"Factor({self.name}): {len(self.data)} obs, {n_symbols} symbols"