"""
Core factor computation engine.

Factor: Transform raw data (close, open, volume, etc.) into alpha factors through time-series and cross-sectional operators.
Supports method chaining for complex factor calculations.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Union, Optional, Callable, List
from scipy.stats import rankdata, norm, uniform, cauchy


class Factor:
    """
    Factor matrix for quantitative analysis.
    
    Transforms raw OHLCV data into alpha factors via operators.
    Internal format: [timestamp, symbol, factor] DataFrame.
    """
    
    # ==================== Initialization ====================
    
    def __init__(self, data: Union[pd.DataFrame, str], name: Optional[str] = None):
        """
        Initialize factor matrix from DataFrame or CSV.
        
        Parameters
        ----------
        data : DataFrame or str
            Factor data or CSV path with [timestamp, symbol, factor] columns
        name : str, optional
            Factor name
        """
        if isinstance(data, str):
            df = pd.read_csv(data, parse_dates=['timestamp'])
        else:
            df = data.copy()
        
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

    # ==================== Validation & Helper Methods ====================

    def _validate_window(self, window: int) -> None:
        """Validate window parameter."""
        if window <= 0:
            raise ValueError("Window must be positive")
    
    def _validate_factor(self, other: 'Factor', op_name: str) -> None:
        """Validate that other is a Factor object."""
        if not isinstance(other, Factor):
            raise TypeError(f"{op_name}: other must be a Factor object.")
    
    @staticmethod
    def _replace_inf(series: pd.Series) -> pd.Series:
        """Replace infinite values with NaN."""
        return series.replace([np.inf, -np.inf], np.nan)
    
    def _apply_cs_operation(self, operation: Callable, name_suffix: str, 
                           require_no_nan: bool = False) -> 'Factor':
        """Apply cross-sectional operation per timestamp."""
        result = self.data.copy()
        result['factor'] = pd.to_numeric(result['factor'], errors='coerce')
        
        if require_no_nan and result['factor'].isna().all():
            raise ValueError("All factor values are NaN")
        
        def safe_op(group):
            if group.isna().any():
                return pd.Series(np.nan, index=group.index)
            output = operation(group)
            if isinstance(output, (int, float, np.number)):
                return pd.Series(output, index=group.index)
            return output
        
        result['factor'] = result.groupby('timestamp')['factor'].transform(safe_op)
        return Factor(result, f"{name_suffix}({self.name})")

    def _binary_op(self, other: Union['Factor', float], op_func: Callable, 
                   op_name: str, scalar_suffix: Optional[str] = None) -> 'Factor':
        """
        Execute binary operation with another Factor or scalar.
        
        Parameters
        ----------
        other : Factor or float
            Second operand
        op_func : Callable
            Function applying operation: op_func(x_vals, y_vals) for Factor,
            op_func(x_vals, scalar) for scalar
        op_name : str
            Symbol for naming result: '+', '-', '*', '/'
        scalar_suffix : str, optional
            Custom suffix for name construction
        """
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                            on=['timestamp', 'symbol'],
                            suffixes=('_x', '_y'), how='inner')
            merged['factor'] = op_func(merged['factor_x'], merged['factor_y'])
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"({self.name}{op_name}{other.name})")
        else:
            result = self.data.copy()
            result['factor'] = op_func(result['factor'], other)
            suffix = scalar_suffix if scalar_suffix is not None else str(other)
            return Factor(result, f"({self.name}{op_name}{suffix})")

    def _comparison_op(self, other: Union['Factor', float], comp_func: Callable, 
                       op_name: str) -> 'Factor':
        """
        Execute comparison operation.
        
        Parameters
        ----------
        other : Factor or float
            Second operand
        comp_func : Callable
            Comparison function: comp_func(x, y) -> bool array
        op_name : str
            Symbol for naming: '<', '<=', '>', '>=', '==', '!='
        """
        if isinstance(other, Factor):
            merged = pd.merge(self.data, other.data,
                            on=['timestamp', 'symbol'],
                            suffixes=('_x', '_y'))
            merged['factor'] = comp_func(merged['factor_x'], merged['factor_y']).astype(int)
            result = merged[['timestamp', 'symbol', 'factor']]
        else:
            result = self.data.copy()
            result['factor'] = comp_func(result['factor'], other).astype(int)
        return Factor(result, f"({self.name}{op_name}{getattr(other, 'name', other)})")

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

    # ==================== Cross-Sectional Operations (cs_op) ====================

    def rank(self) -> 'Factor':
        """Cross-sectional rank (0-1). NaN if any value is NaN in timestamp."""
        def rank_op(group):
            if group.nunique() == 1:
                return pd.Series(np.nan, index=group.index)
            return group.rank(method='min', pct=True)
        
        return self._apply_cs_operation(rank_op, 'rank', require_no_nan=True)

    def mean(self) -> 'Factor':
        """Cross-sectional mean within timestamp."""
        return self._apply_cs_operation(lambda g: g.mean(), 'mean', require_no_nan=True)

    def median(self) -> 'Factor':
        """Cross-sectional median within timestamp."""
        return self._apply_cs_operation(lambda g: g.median(), 'median', require_no_nan=True)

    def quantile(self, driver: str = "gaussian", sigma: float = 1.0) -> 'Factor':
        """Cross-sectional quantile transformation (Gaussian/Uniform/Cauchy)."""
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
        """Scale to target book size with optional long/short asymmetry."""
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

    def normalize(self, useStd: bool = False, limit: float = 0.0) -> 'Factor':
        """Cross-sectional normalization: subtract mean, optionally divide by std, optionally clip."""
        result = self.data.copy()

        def apply_normalize(group: pd.Series):
            if group.isna().any():
                return pd.Series(np.nan, index=group.index)
            
            normalized_group = group - group.mean()

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

    def zscore(self) -> 'Factor':
        """Cross-sectional Z-score."""
        return self.normalize(useStd=True)

    def group_neutralize(self, group_data: 'Factor') -> 'Factor':
        """Neutralize against group membership (industry, sector, etc.)."""
        self._validate_factor(group_data, "group_neutralize")

        merged = pd.merge(self.data, group_data.data, 
                         on=['timestamp', 'symbol'],
                         suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data between factor and group data.")

        def safe_neutralize(group):
            if group['factor'].isna().any():
                group['factor'] = np.nan
            else:
                group['factor'] = group.groupby('factor_group')['factor'].transform(
                    lambda x: x - x.mean()
                )
            return group

        merged = merged.groupby(['timestamp', 'factor_group'], group_keys=False).apply(
            safe_neutralize, include_groups=False
        )
        
        result_data = merged[['timestamp', 'symbol', 'factor']]
        return Factor(result_data, name=f"group_neutralize({self.name},{group_data.name})")

    def vector_neut(self, other: 'Factor') -> 'Factor':
        """Orthogonalize: remove linear component of other from self."""
        self._validate_factor(other, "vector_neut")

        merged = pd.merge(self.data, other.data,
                        on=['timestamp', 'symbol'],
                        suffixes=('_target', '_neut'))

        if merged.empty:
            raise ValueError("No common data for vector neutralization.")

        def neutralize_single_date(group):
            """Perform vector neutralization for a single timestamp."""
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
        """Orthogonalize against one or more factors via OLS residuals."""
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

    # ==================== Time-Series Operations (ts_op) ====================

    def ts_rank(self, window: int) -> 'Factor':
        """Rolling time-series rank within window (0-1)."""
        self._validate_window(window)
        
        def safe_ts_rank(x: pd.Series) -> float:
            if x.isna().any() or len(x) < window:
                return np.nan
            
            if x.nunique() == 1:
                return np.nan
            
            ranks = rankdata(x.to_numpy(), method='min')
            return ranks[-1] / len(ranks)
        
        result = self._apply_rolling(safe_ts_rank, window)
        return Factor(result, f"ts_rank({self.name},{window})")
    
    def ts_sum(self, window: int) -> 'Factor':
        """Rolling sum over window."""
        self._validate_window(window)
        
        def safe_sum(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.sum()
        
        result = self._apply_rolling(safe_sum, window)
        return Factor(result, f"ts_sum({self.name},{window})")

    def ts_product(self, window: int) -> 'Factor':
        """Rolling product over window."""
        self._validate_window(window)
        
        def safe_prod(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.prod()
        
        result = self._apply_rolling(safe_prod, window)
        return Factor(result, f"ts_product({self.name},{window})")

    def ts_mean(self, window: int) -> 'Factor':
        """Rolling mean over window."""
        self._validate_window(window)
        
        def safe_mean(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.mean()
        
        result = self._apply_rolling(safe_mean, window)
        return Factor(result, f"ts_mean({self.name},{window})")
        
    def ts_median(self, window: int) -> 'Factor':
        """Rolling median over window."""
        self._validate_window(window)
        result = self._apply_rolling(lambda x: x.median() if not x.isna().all() else np.nan, window)
        return Factor(result, f"ts_median({self.name},{window})")
    
    def ts_std_dev(self, window: int) -> 'Factor':
        """Rolling standard deviation over window."""
        self._validate_window(window)
        
        def safe_std(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.std()
        
        result = self._apply_rolling(safe_std, window)
        return Factor(result, f"ts_std_dev({self.name},{window})")
    
    def ts_min(self, window: int) -> 'Factor':
        """Rolling minimum over window."""
        self._validate_window(window)
        
        def safe_min(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.min()
        
        result = self._apply_rolling(safe_min, window)
        return Factor(result, f"ts_min({self.name},{window})")
    
    def ts_max(self, window: int) -> 'Factor':
        """Rolling maximum over window."""
        self._validate_window(window)
        
        def safe_max(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.max()
        
        result = self._apply_rolling(safe_max, window)
        return Factor(result, f"ts_max({self.name},{window})")
    
    def ts_arg_max(self, window: int) -> 'Factor':
        """Relative index of max value in rolling window."""
        self._validate_window(window)
        
        def safe_arg_max(s):
            return np.nan if s.isna().all() else (len(s) - 1) - s.argmax()
        
        result = self._apply_rolling(safe_arg_max, window)
        return Factor(result, f"ts_arg_max({self.name},{window})")
        
    def ts_arg_min(self, window: int) -> 'Factor':
        """Relative index of min value in rolling window."""
        self._validate_window(window)
        
        def safe_arg_min(s):
            return np.nan if s.isna().all() else (len(s) - 1) - s.argmin()
        
        result = self._apply_rolling(safe_arg_min, window)
        return Factor(result, f"ts_arg_min({self.name},{window})")

    def ts_count_nans(self, window: int) -> 'Factor':
        """Count NaN values in rolling window."""
        self._validate_window(window)
        result = self._apply_rolling(lambda x: x.isna().sum(), window)
        return Factor(result, f"ts_count_nans({self.name},{window})")
    
    def ts_av_diff(self, window: int) -> 'Factor':
        """Returns x - ts_mean(x, window)."""
        self._validate_window(window)
        mean_factor = self.ts_mean(window)
        result = self.subtract(mean_factor)
        return Factor(result.data, f"ts_av_diff({self.name},{window})")

    def ts_scale(self, window: int, constant: float = 0) -> 'Factor':
        """Normalize to [0,1] over window: (x - min) / (max - min) + constant."""
        self._validate_window(window)

        min_factor = self.ts_min(window)
        max_factor = self.ts_max(window)
        
        result = (self - min_factor) / (max_factor - min_factor)
        result.data['factor'] = self._replace_inf(result.data['factor'])
        result = result + constant

        return Factor(result.data, f"ts_scale({self.name},{window},{constant})")
    
    def ts_zscore(self, window: int) -> 'Factor':
        """Z-score: (x - ts_mean) / ts_std_dev."""
        self._validate_window(window)
        
        mean = self.ts_mean(window)
        std = self.ts_std_dev(window)
        
        result = (self - mean) / std
        result.data['factor'] = self._replace_inf(result.data['factor'])
        
        return Factor(result.data, f"ts_zscore({self.name},{window})")

    def ts_quantile(self, window: int, driver: str = "gaussian") -> 'Factor':
        """Inverse CDF of ts_rank with Gaussian/Uniform/Cauchy driver."""
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

    def ts_backfill(self, window: int, k: int = 1) -> 'Factor':
        """Backfill NaN with k-th most recent non-NaN value in window."""
        self._validate_window(window)
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
        """Exponential decay weighted average over rolling window."""
        self._validate_window(window)
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
            return (valid_s * weights).sum() / weight_sum if weight_sum > 0 else (np.nan if nan else 0.0)
        
        result = self._apply_rolling(decay_func, window)
        return Factor(result, f"ts_decay_exp_window({self.name},{window},{factor},{nan})")
    
    def ts_decay_linear(self, window: int, dense: bool = False) -> 'Factor':
        """Linear decay weighted average over rolling window."""
        self._validate_window(window)
            
        def linear_decay_func(s):
            current_s = s.copy() if dense else s.fillna(0)
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
        """Lag: x from window periods ago."""
        result = self.data.copy()
        result['factor'] = result.groupby('symbol')['factor'].shift(window)
        return Factor(result, f"ts_delay({self.name},{window})")
        
    def ts_delta(self, window: int) -> 'Factor':
        """Difference: x - ts_delay(x, window)."""
        result = self.data.copy()
        result['factor'] = result.groupby('symbol')['factor'].diff(window)
        return Factor(result, f"ts_delta({self.name},{window})")

    def ts_corr(self, other: 'Factor', window: int) -> 'Factor':
        """Rolling Pearson correlation over window."""
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
    
    def ts_covariance(self, other: 'Factor', window: int) -> 'Factor':
        """Rolling covariance over window."""
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

    def ts_regression(self, x_factor: 'Factor', window: int, lag: int = 0, rettype: int = 0) -> 'Factor':
        """
        Rolling linear regression Y = alpha + beta * X.
        
        rettype: 0=residual, 1=alpha, 2=beta, 3=prediction, 4=SSE, 5=SST, 6=R², 7=MSE, 8=stderr_beta, 9=stderr_alpha
        """
        self._validate_window(window)
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
                    if X.std() == 0:
                        results.append(np.nan)
                        continue
                    
                    X_with_const = np.column_stack([np.ones(len(X)), X])
                    
                    if np.linalg.cond(X_with_const) > 1e10:
                        results.append(np.nan)
                        continue
                    
                    XtX = X_with_const.T @ X_with_const
                    Xty = X_with_const.T @ Y
                    params = np.linalg.solve(XtX, Xty)
                    
                    alpha = params[0]
                    beta = params[1]
                    
                    y_pred = X_with_const @ params
                    residuals = Y - y_pred
                    
                    y_estimate = y_pred[-1]
                    error_term = residuals[-1]
                    
                    SSE = np.sum(residuals ** 2)
                    SST = np.sum((Y - Y.mean()) ** 2)
                    
                    R_squared = 1 - (SSE / SST) if SST > 0 else np.nan
                    df_resid = len(Y) - 2
                    MSE = SSE / df_resid if df_resid > 0 else np.nan
                    
                    if MSE > 0:
                        var_covar = MSE * np.linalg.inv(XtX)
                        std_err_alpha = np.sqrt(var_covar[0, 0])
                        std_err_beta = np.sqrt(var_covar[1, 1])
                    else:
                        std_err_alpha = np.nan
                        std_err_beta = np.nan
                    
                    result_values = [
                        error_term, alpha, beta, y_estimate, SSE, SST,
                        R_squared, MSE, std_err_beta, std_err_alpha
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

    # ==================== Unary Operations ====================

    def abs(self) -> 'Factor':
        """Absolute value."""
        result = self.data.copy()
        result['factor'] = np.abs(result['factor'])
        return Factor(result, f"abs({self.name})")

    def sign(self) -> 'Factor':
        """Sign: -1, 0, or 1."""
        result = self.data.copy()
        result['factor'] = np.sign(result['factor'])
        return Factor(result, f"sign({self.name})")

    def inverse(self) -> 'Factor':
        """Inverse: 1/x (safe: x=0 -> NaN)."""
        result = self.data.copy()
        result['factor'] = np.where(
            result['factor'] != 0,
            1 / result['factor'],
            np.nan
        )
        return Factor(result, f"inverse({self.name})")

    def log(self, base: Optional[float] = None) -> 'Factor':
        """Logarithm (natural log if base=None, safe: x<=0 -> NaN)."""
        result = self.data.copy()
        
        if base is None:
            result['factor'] = np.where(
                result['factor'] > 0,
                np.log(result['factor']),
                np.nan
            )
            return Factor(result, f"log({self.name})")
        else:
            if base <= 0 or base == 1:
                raise ValueError(f"Invalid log base: {base}. Base must be positive and not equal to 1.")
            
            result['factor'] = np.where(
                result['factor'] > 0,
                np.log(result['factor']) / np.log(base),
                np.nan
            )
            return Factor(result, f"log({self.name},base={base})")
    
    def ln(self) -> 'Factor':
        """Natural logarithm (safe: x<=0 -> NaN)."""
        return self.log()

    def sqrt(self) -> 'Factor':
        """Square root (safe: x<0 -> NaN)."""
        result = self.data.copy()
        result['factor'] = np.where(
            result['factor'] >= 0,
            np.sqrt(result['factor']),
            np.nan
        )
        return Factor(result, f"sqrt({self.name})")

    def s_log_1p(self) -> 'Factor':
        """Sign-preserving log: sign(x) * log(1 + |x|)."""
        result = self.data.copy()
        result['factor'] = np.sign(result['factor']) * np.log1p(np.abs(result['factor']))
        return Factor(result, f"s_log_1p({self.name})")

    def signed_power(self, exponent: Union['Factor', float]) -> 'Factor':
        """Sign-preserving power: sign(x) * |x|^exponent."""
        if isinstance(exponent, Factor):
            merged = pd.merge(self.data, exponent.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            
            sign = np.sign(merged['factor_x'])
            abs_val = np.abs(merged['factor_x'])
            
            with np.errstate(invalid='ignore', divide='ignore'):
                result_val = sign * (abs_val ** merged['factor_y'])
            
            merged['factor'] = self._replace_inf(result_val)
            
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"signed_power({self.name},{exponent.name})")
        else:
            result = self.data.copy()
            
            sign = np.sign(result['factor'])
            abs_val = np.abs(result['factor'])
            
            with np.errstate(invalid='ignore', divide='ignore'):
                result_val = sign * (abs_val ** exponent)
            
            result['factor'] = self._replace_inf(result_val)
            
            return Factor(result, f"signed_power({self.name},{exponent})")

    def power(self, exponent: Union['Factor', float]) -> 'Factor':
        """Power: x^exponent (safe: invalid cases -> NaN)."""
        if isinstance(exponent, Factor):
            merged = pd.merge(self.data, exponent.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            
            with np.errstate(invalid='ignore', divide='ignore'):
                merged['factor'] = merged['factor_x'] ** merged['factor_y']
            
            merged['factor'] = self._replace_inf(merged['factor'])
            
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"({self.name}**{exponent.name})")
        else:
            result = self.data.copy()
            
            with np.errstate(invalid='ignore', divide='ignore'):
                result['factor'] = result['factor'] ** exponent
            
            result['factor'] = self._replace_inf(result['factor'])
            
            return Factor(result, f"({self.name}**{exponent})")

    # ==================== Binary Operations ====================

    def add(self, other: Union['Factor', float]) -> 'Factor':
        """Addition: factor + other."""
        return self.__add__(other)
    
    def subtract(self, other: Union['Factor', float]) -> 'Factor':
        """Subtraction: factor - other."""
        return self.__sub__(other)
    
    def multiply(self, other: Union['Factor', float]) -> 'Factor':
        """Multiplication: factor * other."""
        return self.__mul__(other)
    
    def divide(self, other: Union['Factor', float]) -> 'Factor':
        """Division: factor / other (safe: div by 0 -> NaN)."""
        return self.__truediv__(other)

    def where(self, cond: 'Factor', other: Union['Factor', float] = np.nan) -> 'Factor':
        """Conditional: where(cond, self, other)."""
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

    def maximum(self, other: Union['Factor', float]) -> 'Factor':
        """Element-wise maximum."""
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
        """Element-wise minimum."""
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

    def reverse(self) -> 'Factor':
        """Reverse sign of factor: -factor"""
        return self.__neg__()

    # ==================== Operator Overloads ====================

    def __neg__(self) -> 'Factor':
        """Unary negation: -factor"""
        return self.multiply(-1)
    
    def __invert__(self) -> 'Factor':
        """Boolean NOT operator: ~factor. Non-zero becomes 0, zero/NaN becomes 1."""
        result = self.data.copy()
        bool_values = result['factor'].fillna(0).astype(bool)
        result['factor'] = (~bool_values).astype(int)
        return Factor(result, f"~{self.name}")
    
    def __add__(self, other: Union['Factor', float]) -> 'Factor':
        """Addition: factor + other"""
        return self._binary_op(other, lambda x, y: x + y, '+')
    
    def __radd__(self, other: Union['Factor', float]) -> 'Factor':
        """Right addition: other + factor"""
        return self.__add__(other)
    
    def __sub__(self, other: Union['Factor', float]) -> 'Factor':
        """Subtraction: factor - other"""
        return self._binary_op(other, lambda x, y: x - y, '-')
    
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
        return self._binary_op(other, lambda x, y: x * y, '*')
    
    def __rmul__(self, other: Union['Factor', float]) -> 'Factor':
        """Right multiplication: other * factor"""
        return self.__mul__(other)

    def __abs__(self) -> 'Factor':
        """Absolute value: abs(factor)"""
        return self.abs()

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
            
            result['factor'] = self._replace_inf(result['factor'])
            
            return Factor(result, f"({other}**{self.name})")

    def __truediv__(self, other: Union['Factor', float]) -> 'Factor':
        """
        Division: factor / other.
        
        Safely handles division by zero by returning NaN.
        """
        def safe_div(x, y):
            if isinstance(y, (int, float)):
                if y == 0:
                    return np.nan
                return x / y
            else:
                return np.where(np.abs(y) > 1e-10, x / y, np.nan)
        
        return self._binary_op(other, safe_div, '/')

    def __rtruediv__(self, other: Union['Factor', float]) -> 'Factor':
        """
        Right division: other / factor.
        
        Safely handles division by zero by returning NaN.
        """
        if isinstance(other, Factor):
            return other.__truediv__(self)
        else:
            result = self.data.copy()
            
            result['factor'] = np.where(
                result['factor'] != 0,
                other / result['factor'],
                np.nan
            )
            
            return Factor(result, f"({other}/{self.name})")

    def __lt__(self, other: Union['Factor', float]) -> 'Factor':
        """Less than: factor < other"""
        return self._comparison_op(other, lambda x, y: x < y, '<')

    def __le__(self, other: Union['Factor', float]) -> 'Factor':
        """Less than or equal to: factor <= other"""
        return self._comparison_op(other, lambda x, y: x <= y, '<=')

    def __gt__(self, other: Union['Factor', float]) -> 'Factor':
        """Greater than: factor > other"""
        return self._comparison_op(other, lambda x, y: x > y, '>')

    def __ge__(self, other: Union['Factor', float]) -> 'Factor':
        """Greater than or equal to: factor >= other"""
        return self._comparison_op(other, lambda x, y: x >= y, '>=')

    def __eq__(self, other: Union['Factor', float]) -> 'Factor':
        """Equal to: factor == other"""
        return self._comparison_op(other, lambda x, y: x == y, '==')

    def __ne__(self, other: Union['Factor', float]) -> 'Factor':
        """Not equal to: factor != other"""
        return self._comparison_op(other, lambda x, y: x != y, '!=')

    # ==================== Data Access & Information ====================

    def to_weights(self, date: Optional[Union[str, pd.Timestamp]] = None) -> dict:
        """Convert factor to dollar-neutral portfolio weights (demeaned & normalized)."""
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
        """Save to CSV."""
        self.data.to_csv(path, index=False)
        return path
    
    def info(self) -> None:
        """Print data quality info: metadata and NaN statistics."""
        n_obs = len(self.data)
        n_symbols = self.data['symbol'].nunique()
        n_nan = self.data['factor'].isna().sum()
        nan_ratio = n_nan / n_obs if n_obs > 0 else 0
        time_range = f"{self.data['timestamp'].min().strftime('%Y-%m-%d')} to {self.data['timestamp'].max().strftime('%Y-%m-%d')}"
        
        print(f"Factor: {self.name}")
        print(f"  obs={n_obs}, symbols={n_symbols}, period={time_range}")
        print(f"  NaN: {n_nan} ({nan_ratio:.1%})")
    
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