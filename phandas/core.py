"""Core factor computation engine for alpha factor transformations via operators."""

import warnings
import pandas as pd
import numpy as np
from typing import Union, Optional, Callable, List
from scipy.stats import norm, uniform, cauchy


class Factor:
    """Factor matrix for quantitative analysis.
    
    Parameters
    ----------
    data : pd.DataFrame or str
        DataFrame with columns (timestamp, symbol, factor) or path to CSV
    name : str, optional
        Factor name for tracking in transformations
    
    Attributes
    ----------
    data : pd.DataFrame
        Sorted factor data (timestamp, symbol, factor)
    name : str
        Factor identifier
    """
    
    def __init__(self, data: Union[pd.DataFrame, str], name: Optional[str] = None):
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
        self.data = self.data.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        self.name = name or 'factor'

    def _validate_window(self, window: int) -> None:
        if window <= 0:
            raise ValueError("Window must be positive")
    
    def _validate_factor(self, other: 'Factor', op_name: str) -> None:
        if not isinstance(other, Factor):
            raise TypeError(f"{op_name}: other must be a Factor object.")
    
    @staticmethod
    def _replace_inf(series: pd.Series) -> pd.Series:
        return series.replace([np.inf, -np.inf], np.nan)
    
    def _apply_cs_operation(self, operation: Callable, name_suffix: str, 
                           require_no_nan: bool = False) -> 'Factor':
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

    def rank(self) -> 'Factor':
        def rank_op(group):
            if group.nunique() == 1:
                return pd.Series(np.nan, index=group.index)
            return group.rank(method='min', pct=True)
        
        return self._apply_cs_operation(rank_op, 'rank', require_no_nan=True)

    def mean(self) -> 'Factor':
        return self._apply_cs_operation(lambda g: g.mean(), 'mean', require_no_nan=True)

    def median(self) -> 'Factor':
        return self._apply_cs_operation(lambda g: g.median(), 'median', require_no_nan=True)

    def quantile(self, driver: str = "gaussian", sigma: float = 1.0) -> 'Factor':
        valid_drivers = {
            "gaussian": norm.ppf,
            "uniform": uniform.ppf,
            "cauchy": cauchy.ppf,
        }

        if driver not in valid_drivers:
            raise ValueError(f"Invalid driver: {driver}. Must be one of {list(valid_drivers.keys())}")

        ppf_func = valid_drivers[driver]
        result = self.data.copy()

        def apply_quantile_vectorized(group: pd.Series):
            vals = group.values
            valid_mask = ~np.isnan(vals)
            N = valid_mask.sum()
            
            if N < 2:
                return pd.Series(np.nan, index=group.index)
            
            ranked = np.full(len(vals), np.nan)
            valid_vals = vals[valid_mask]
            
            sorted_idx = np.argsort(valid_vals)
            rank_array = np.empty_like(sorted_idx, dtype=float)
            rank_array[sorted_idx] = np.arange(1, N + 1)
            ranked[valid_mask] = rank_array / N
            
            epsilon = 1e-6
            shifted_rank = 1/N + np.clip(ranked, epsilon, 1-epsilon) * (1 - 2/N)
            shifted_rank = np.clip(shifted_rank, epsilon, 1-epsilon)
            
            transformed = ppf_func(shifted_rank)
            if sigma != 1.0:
                transformed *= sigma
            
            return pd.Series(transformed, index=group.index)

        result['factor'] = result.groupby('timestamp', group_keys=False)['factor'].apply(
            apply_quantile_vectorized
        )
        return Factor(result, f"quantile({self.name},driver={driver},sigma={sigma})")

    def scale(self, scale: float = 1.0, longscale: Optional[float] = None, 
              shortscale: Optional[float] = None) -> 'Factor':
        result = self.data.copy()

        def apply_scale(group: pd.Series):
            if longscale is not None or shortscale is not None:
                scaled_group = group.copy()
                
                long_mask = group > 0
                short_mask = group < 0
                
                ls = longscale if longscale is not None else scale
                ss = shortscale if shortscale is not None else scale
                
                if long_mask.any() and ls > 0:
                    long_abs_sum = group[long_mask].abs().sum()
                    if long_abs_sum > 0:
                        scaled_group[long_mask] = (group[long_mask] / long_abs_sum) * ls
                
                if short_mask.any() and ss > 0:
                    short_abs_sum = group[short_mask].abs().sum()
                    if short_abs_sum > 0:
                        scaled_group[short_mask] = (group[short_mask] / short_abs_sum) * (-ss)
                
                return scaled_group
            else:
                abs_sum = group.abs().sum()
                if abs_sum == 0:
                    return group
                return (group / abs_sum) * scale

        result['factor'] = result.groupby('timestamp')['factor'].transform(apply_scale)
        
        name_parts = [f"scale={scale}"]
        if longscale is not None:
            name_parts.append(f"longscale={longscale}")
        if shortscale is not None:
            name_parts.append(f"shortscale={shortscale}")
        return Factor(result, f"scale({self.name},{','.join(name_parts)})")

    def normalize(self, use_std: bool = False, limit: float = 0.0) -> 'Factor':
        result = self.data.copy()

        def apply_normalize(group: pd.Series):
            if group.isna().any():
                return pd.Series(np.nan, index=group.index)
            
            normalized_group = group - group.mean()

            if use_std:
                std_val = group.std()
                if std_val == 0 or pd.isna(std_val):
                    return pd.Series(np.nan, index=group.index)
                normalized_group /= std_val
            
            if limit != 0.0:
                normalized_group = normalized_group.clip(lower=-limit, upper=limit)
            
            return normalized_group

        result['factor'] = result.groupby('timestamp')['factor'].transform(apply_normalize)
        return Factor(result, f"normalize({self.name},use_std={use_std},limit={limit})")

    def zscore(self) -> 'Factor':
        return self.normalize(use_std=True)

    def spread(self, pct: float = 0.5) -> 'Factor':
        if not 0 < pct < 1:
            raise ValueError("pct must be between 0 and 1")
        
        result = self.data.copy()
        
        def create_spread(group: pd.Series) -> pd.Series:
            values = group.values
            n_assets = len(values)
            n_long = int(n_assets * pct) or 1
            
            spread_values = np.zeros(n_assets)
            valid_mask = ~np.isnan(values)
            
            if valid_mask.sum() < 2:
                return pd.Series(spread_values, index=group.index)
            
            sorted_indices = np.argsort(values)
            long_indices = sorted_indices[-n_long:]
            spread_values[long_indices] = 0.5
            
            short_indices = sorted_indices[:n_long]
            spread_values[short_indices] = -0.5
            
            return pd.Series(spread_values, index=group.index)
        
        result['factor'] = result.groupby('timestamp', group_keys=False)['factor'].apply(create_spread)
        return Factor(result, f"spread({self.name},{pct})")

    def signal(self) -> 'Factor':
        result = self.data.copy()
        
        def to_dn_signal(group):
            if group.isna().any():
                return pd.Series(np.nan, index=group.index)
            
            demeaned = group - group.mean()
            abs_sum = np.abs(demeaned).sum()
            if abs_sum < 1e-10:
                return pd.Series(np.nan, index=group.index)
            return demeaned / abs_sum
        
        result['factor'] = result.groupby('timestamp', group_keys=False)['factor'].transform(to_dn_signal)
        return Factor(result, f"signal({self.name})")

    def _is_signal(self, date=None) -> bool:
        if date is not None:
            factors = self.data[self.data['timestamp'] == date].set_index('symbol')['factor']
        else:
            latest_date = self.data['timestamp'].max()
            factors = self.data[self.data['timestamp'] == latest_date].set_index('symbol')['factor']
        
        if factors.empty or factors.isna().all():
            return False
        
        long_sum = factors[factors > 0].sum()
        short_sum = factors[factors < 0].sum()
        total_sum = long_sum + short_sum
        
        return (np.isclose(long_sum, 0.5, atol=1e-2) and 
                np.isclose(short_sum, -0.5, atol=1e-2) and
                np.isclose(total_sum, 0.0, atol=1e-2))

    def vector_neut(self, other: 'Factor') -> 'Factor':
        self._validate_factor(other, "vector_neut")

        merged = pd.merge(self.data, other.data,
                        on=['timestamp', 'symbol'],
                        suffixes=('_target', '_neut'))

        if merged.empty:
            raise ValueError("No common data for vector neutralization.")

        def neutralize_single_date(group):
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
        if isinstance(neut_factors, Factor):
            neut_factors = [neut_factors]
        if not all(isinstance(f, Factor) for f in neut_factors):
            raise TypeError("neut_factors must be a Factor or list of Factors.")

        merged = self.data.rename(columns={'factor': self.name or 'target'})
        neut_names = []
        
        for i, f in enumerate(neut_factors):
            neut_name = f.name or f'neut_{i}'
            neut_names.append(neut_name)
            merged = pd.merge(merged, f.data.rename(columns={'factor': neut_name}),
                              on=['timestamp', 'symbol'], how='inner')

        if merged.empty:
            raise ValueError("No common data for regression neutralization.")
        
        def get_residuals_vectorized(group):
            Y = group[self.name or 'target'].values
            X = group[neut_names].values
            
            valid_mask = ~(np.isnan(Y) | np.isnan(X).any(axis=1))
            n_valid = valid_mask.sum()
            
            if n_valid < 2:
                return pd.Series(np.nan, index=group.index, name='factor')
            
            Y_valid = Y[valid_mask]
            X_valid = X[valid_mask]
            
            X_const = np.column_stack([np.ones(n_valid), X_valid])
            
            if np.linalg.cond(X_const) > 1e10:
                return pd.Series(np.nan, index=group.index, name='factor')
            
            try:
                params = np.linalg.lstsq(X_const, Y_valid, rcond=None)[0]
                residuals = Y_valid - X_const @ params
                
                result = np.full(len(Y), np.nan)
                result[valid_mask] = residuals
                return pd.Series(result, index=group.index, name='factor')
            except (np.linalg.LinAlgError, ValueError):
                return pd.Series(np.nan, index=group.index, name='factor')

        result = merged.groupby('timestamp', group_keys=False).apply(
            get_residuals_vectorized, include_groups=False
        ).reset_index(level=0, drop=True)
        
        merged['factor'] = result
        result_data = merged[['timestamp', 'symbol', 'factor']]

        neut_factors_str = ",".join([f.name for f in neut_factors])
        return Factor(result_data, name=f"regression_neut({self.name},[{neut_factors_str}])")

    def group(self, mapping: Union[str, dict], name: Optional[str] = None) -> 'Factor':
        from .constants import GROUP_DEFINITIONS
        
        if isinstance(mapping, str):
            if mapping not in GROUP_DEFINITIONS:
                raise ValueError(f"Unknown mapping name '{mapping}'. Available: {list(GROUP_DEFINITIONS.keys())}")
            mapping_dict = GROUP_DEFINITIONS[mapping]
            map_name = mapping
        elif isinstance(mapping, dict):
            mapping_dict = mapping
            map_name = "custom_map"
        else:
            raise TypeError("Mapping must be a string (key in GROUP_DEFINITIONS) or a dictionary.")
            
        result = self.data[['timestamp', 'symbol']].copy()
        result['factor'] = result['symbol'].map(mapping_dict)
        
        return Factor(result, name or f"group({map_name})")

    def group_neutralize(self, group: 'Factor') -> 'Factor':
        self._validate_factor(group, "group_neutralize")
        
        merged = pd.merge(self.data, group.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data for group neutralization.")
            
        group_means = merged.groupby(['timestamp', 'factor_group'])['factor'].transform('mean')
        
        merged['factor'] = merged['factor'] - group_means
        
        return Factor(merged[['timestamp', 'symbol', 'factor']], f"group_neutralize({self.name},{group.name})")

    def group_mean(self, group: 'Factor') -> 'Factor':
        self._validate_factor(group, "group_mean")
        
        merged = pd.merge(self.data, group.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data for group mean stats.")
            
        group_means = merged.groupby(['timestamp', 'factor_group'])['factor'].transform('mean')
        
        merged['factor'] = group_means
        
        return Factor(merged[['timestamp', 'symbol', 'factor']], f"group_mean({self.name},{group.name})")

    def group_median(self, group: 'Factor') -> 'Factor':
        self._validate_factor(group, "group_median")
        
        merged = pd.merge(self.data, group.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data for group median stats.")
            
        group_medians = merged.groupby(['timestamp', 'factor_group'])['factor'].transform('median')
        
        merged['factor'] = group_medians
        
        return Factor(merged[['timestamp', 'symbol', 'factor']], f"group_median({self.name},{group.name})")

    def group_rank(self, group: 'Factor') -> 'Factor':
        self._validate_factor(group, "group_rank")
        
        merged = pd.merge(self.data, group.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data for group rank.")
            
        merged['factor'] = merged.groupby(['timestamp', 'factor_group'])['factor'].rank(pct=True)
        
        return Factor(merged[['timestamp', 'symbol', 'factor']], f"group_rank({self.name},{group.name})")

    def group_scale(self, group: 'Factor') -> 'Factor':
        self._validate_factor(group, "group_scale")
        
        merged = pd.merge(self.data, group.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data for group scale.")
            
        grouped = merged.groupby(['timestamp', 'factor_group'])['factor']
        g_min = grouped.transform('min')
        g_max = grouped.transform('max')
        
        denom = g_max - g_min
        numerator = merged['factor'] - g_min
        
        merged['factor'] = np.where(denom > 1e-10, numerator / denom, 0.5)
        
        return Factor(merged[['timestamp', 'symbol', 'factor']], f"group_scale({self.name},{group.name})")

    def group_zscore(self, group: 'Factor') -> 'Factor':
        self._validate_factor(group, "group_zscore")
        
        merged = pd.merge(self.data, group.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data for group zscore.")
            
        grouped = merged.groupby(['timestamp', 'factor_group'])['factor']
        g_mean = grouped.transform('mean')
        g_std = grouped.transform('std')
        
        merged['factor'] = np.where(g_std > 1e-10, (merged['factor'] - g_mean) / g_std, np.nan)
        
        return Factor(merged[['timestamp', 'symbol', 'factor']], f"group_zscore({self.name},{group.name})")

    def group_normalize(self, group: 'Factor', scale: float = 1.0) -> 'Factor':
        self._validate_factor(group, "group_normalize")
        
        merged = pd.merge(self.data, group.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data for group normalize.")
            
        merged['abs_val'] = merged['factor'].abs()
        g_abs_sum = merged.groupby(['timestamp', 'factor_group'])['abs_val'].transform('sum')
        
        merged['factor'] = np.where(g_abs_sum > 1e-10, (merged['factor'] / g_abs_sum) * scale, 0.0)
        
        return Factor(merged[['timestamp', 'symbol', 'factor']], f"group_normalize({self.name},{group.name},{scale})")

    def ts_rank(self, window: int) -> 'Factor':
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
    
    def ts_sum(self, window: int) -> 'Factor':
        self._validate_window(window)
        
        def safe_sum(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.sum()
        
        result = self._apply_rolling(safe_sum, window)
        return Factor(result, f"ts_sum({self.name},{window})")

    def ts_product(self, window: int) -> 'Factor':
        self._validate_window(window)
        
        def safe_prod(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.prod()
        
        result = self._apply_rolling(safe_prod, window)
        return Factor(result, f"ts_product({self.name},{window})")

    def ts_mean(self, window: int) -> 'Factor':
        self._validate_window(window)
        
        def safe_mean(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.mean()
        
        result = self._apply_rolling(safe_mean, window)
        return Factor(result, f"ts_mean({self.name},{window})")
        
    def ts_median(self, window: int) -> 'Factor':
        self._validate_window(window)
        result = self._apply_rolling(lambda x: x.median() if not x.isna().all() else np.nan, window)
        return Factor(result, f"ts_median({self.name},{window})")
    
    def ts_std_dev(self, window: int) -> 'Factor':
        self._validate_window(window)
        
        def safe_std(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.std()
        
        result = self._apply_rolling(safe_std, window)
        return Factor(result, f"ts_std_dev({self.name},{window})")
    
    def ts_min(self, window: int) -> 'Factor':
        self._validate_window(window)
        
        def safe_min(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.min()
        
        result = self._apply_rolling(safe_min, window)
        return Factor(result, f"ts_min({self.name},{window})")
    
    def ts_max(self, window: int) -> 'Factor':
        self._validate_window(window)
        
        def safe_max(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.max()
        
        result = self._apply_rolling(safe_max, window)
        return Factor(result, f"ts_max({self.name},{window})")
    
    def ts_arg_max(self, window: int) -> 'Factor':
        self._validate_window(window)
        
        def safe_arg_max(s):
            return np.nan if s.isna().all() else (len(s) - 1) - s.argmax()
        
        result = self._apply_rolling(safe_arg_max, window)
        return Factor(result, f"ts_arg_max({self.name},{window})")
        
    def ts_arg_min(self, window: int) -> 'Factor':
        self._validate_window(window)
        
        def safe_arg_min(s):
            return np.nan if s.isna().all() else (len(s) - 1) - s.argmin()
        
        result = self._apply_rolling(safe_arg_min, window)
        return Factor(result, f"ts_arg_min({self.name},{window})")

    def ts_count_nans(self, window: int) -> 'Factor':
        self._validate_window(window)
        
        result = self.data.copy()
        result['factor'] = (result.groupby('symbol')['factor']
                           .rolling(window, min_periods=1)
                           .apply(lambda x: x.isna().sum(), raw=False)
                           .reset_index(level=0, drop=True))
        
        return Factor(result, f"ts_count_nans({self.name},{window})")
    
    def ts_av_diff(self, window: int) -> 'Factor':
        self._validate_window(window)
        mean_factor = self.ts_mean(window)
        result = self.subtract(mean_factor)
        return Factor(result.data, f"ts_av_diff({self.name},{window})")

    def ts_scale(self, window: int, constant: float = 0) -> 'Factor':
        self._validate_window(window)

        min_factor = self.ts_min(window)
        max_factor = self.ts_max(window)
        
        result = (self - min_factor) / (max_factor - min_factor)
        result.data['factor'] = self._replace_inf(result.data['factor'])
        result = result + constant

        return Factor(result.data, f"ts_scale({self.name},{window},{constant})")
    
    def ts_zscore(self, window: int) -> 'Factor':
        self._validate_window(window)
        
        mean = self.ts_mean(window)
        std = self.ts_std_dev(window)
        
        result = (self - mean) / std
        result.data['factor'] = self._replace_inf(result.data['factor'])
        
        return Factor(result.data, f"ts_zscore({self.name},{window})")

    def ts_quantile(self, window: int, driver: str = "gaussian") -> 'Factor':
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

    def ts_kurtosis(self, window: int) -> 'Factor':
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

    def ts_skewness(self, window: int) -> 'Factor':
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

    def ts_backfill(self, window: int, k: int = 1) -> 'Factor':
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
    
    def ts_decay_exp_window(self, window: int, factor: float = 1.0, nan: bool = True) -> 'Factor':
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
    
    def ts_decay_linear(self, window: int, dense: bool = False) -> 'Factor':
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
    
    def ts_step(self, start: int = 1) -> 'Factor':
        result = self.data.copy()
        result['factor'] = result.groupby('symbol').cumcount() + start
        return Factor(result, f"ts_step({start})")
    
    def ts_delay(self, window: int) -> 'Factor':
        result = self.data.copy()
        result['factor'] = result.groupby('symbol')['factor'].shift(window)
        return Factor(result, f"ts_delay({self.name},{window})")
        
    def ts_delta(self, window: int) -> 'Factor':
        result = self.data.copy()
        result['factor'] = result.groupby('symbol')['factor'].diff(window)
        return Factor(result, f"ts_delta({self.name},{window})")

    def ts_corr(self, other: 'Factor', window: int) -> 'Factor':
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

    def ts_regression(self, x_factor, window: int, lag: int = 0, rettype: int = 0) -> 'Factor':
        self._validate_window(window)
        if lag < 0:
            raise ValueError("Lag must be non-negative")
        
        is_multi = isinstance(x_factor, list)
        if is_multi:
            if not all(isinstance(f, Factor) for f in x_factor):
                raise TypeError("x_factor list must contain only Factor objects")
            x_factors = x_factor
        else:
            if not isinstance(x_factor, Factor):
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

    def ts_cv(self, window: int) -> 'Factor':
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

    def ts_jumpiness(self, window: int) -> 'Factor':
        self._validate_window(window)
        diff = self.ts_delta(1).abs()
        total_jump = diff.ts_sum(window)
        range_val = self.ts_max(window) - self.ts_min(window)
        result = total_jump / (range_val + 1e-10)
        result.data['factor'] = self._replace_inf(result.data['factor'])
        return Factor(result.data, f"ts_jumpiness({self.name},{window})")

    def ts_trend_strength(self, window: int) -> 'Factor':
        self._validate_window(window)
        time_step = self.ts_step()
        result = self.ts_regression(time_step, window, rettype=6)
        return Factor(result.data, f"ts_trend_strength({self.name},{window})")

    def ts_vr(self, window: int, k: int = 2) -> 'Factor':
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

    def ts_autocorr(self, window: int, lag: int = 1) -> 'Factor':
        self._validate_window(window)
        if lag <= 0:
            raise ValueError("lag must be positive")
        lagged = self.ts_delay(lag)
        result = self.ts_corr(lagged, window)
        return Factor(result.data, f"ts_autocorr({self.name},{window},{lag})")

    def ts_reversal_count(self, window: int) -> 'Factor':
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

    def abs(self) -> 'Factor':
        result = self.data.copy()
        result['factor'] = np.abs(result['factor'])
        return Factor(result, f"abs({self.name})")

    def sign(self) -> 'Factor':
        result = self.data.copy()
        result['factor'] = np.sign(result['factor'])
        return Factor(result, f"sign({self.name})")

    def inverse(self) -> 'Factor':
        result = self.data.copy()
        result['factor'] = np.where(
            result['factor'] != 0,
            1 / result['factor'],
            np.nan
        )
        return Factor(result, f"inverse({self.name})")

    def log(self, base: Optional[float] = None) -> 'Factor':
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
        return self.log()

    def sqrt(self) -> 'Factor':
        result = self.data.copy()
        result['factor'] = np.where(
            result['factor'] >= 0,
            np.sqrt(result['factor']),
            np.nan
        )
        return Factor(result, f"sqrt({self.name})")

    def s_log_1p(self) -> 'Factor':
        result = self.data.copy()
        result['factor'] = np.sign(result['factor']) * np.log1p(np.abs(result['factor']))
        return Factor(result, f"s_log_1p({self.name})")

    def signed_power(self, exponent: Union['Factor', float]) -> 'Factor':
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

    def add(self, other: Union['Factor', float]) -> 'Factor':
        return self.__add__(other)
    
    def subtract(self, other: Union['Factor', float]) -> 'Factor':
        return self.__sub__(other)
    
    def multiply(self, other: Union['Factor', float]) -> 'Factor':
        return self.__mul__(other)
    
    def divide(self, other: Union['Factor', float]) -> 'Factor':
        return self.__truediv__(other)

    def where(self, cond: 'Factor', other: Union['Factor', float] = np.nan) -> 'Factor':
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
        return self.__neg__()

    def __neg__(self) -> 'Factor':
        return self.multiply(-1)
    
    def __add__(self, other: Union['Factor', float]) -> 'Factor':
        return self._binary_op(other, lambda x, y: x + y, '+')
    
    def __radd__(self, other: Union['Factor', float]) -> 'Factor':
        return self.__add__(other)
    
    def __sub__(self, other: Union['Factor', float]) -> 'Factor':
        return self._binary_op(other, lambda x, y: x - y, '-')
    
    def __rsub__(self, other: Union['Factor', float]) -> 'Factor':
        if isinstance(other, Factor):
            return other.subtract(self)
        else:
            result = self.data.copy()
            result['factor'] = other - result['factor']
            return Factor(result, f"({other}-{self.name})")
    
    def __mul__(self, other: Union['Factor', float]) -> 'Factor':
        return self._binary_op(other, lambda x, y: x * y, '*')
    
    def __rmul__(self, other: Union['Factor', float]) -> 'Factor':
        return self.__mul__(other)

    def __abs__(self) -> 'Factor':
        return self.abs()

    def __pow__(self, other: Union['Factor', float]) -> 'Factor':
        return self.power(other)
    
    def __rpow__(self, other: Union['Factor', float]) -> 'Factor':
        if isinstance(other, Factor):
            return other.power(self)
        else:
            result = self.data.copy()
            
            with np.errstate(invalid='ignore', divide='ignore'):
                result['factor'] = other ** result['factor']
            
            result['factor'] = self._replace_inf(result['factor'])
            
            return Factor(result, f"({other}**{self.name})")

    def __truediv__(self, other: Union['Factor', float]) -> 'Factor':
        def safe_div(x, y):
            if isinstance(y, (int, float)):
                if y == 0:
                    return np.nan
                return x / y
            else:
                return np.where(np.abs(y) > 1e-10, x / y, np.nan)
        
        return self._binary_op(other, safe_div, '/')

    def __rtruediv__(self, other: Union['Factor', float]) -> 'Factor':
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
        return self._comparison_op(other, lambda x, y: x < y, '<')

    def __le__(self, other: Union['Factor', float]) -> 'Factor':
        return self._comparison_op(other, lambda x, y: x <= y, '<=')

    def __gt__(self, other: Union['Factor', float]) -> 'Factor':
        return self._comparison_op(other, lambda x, y: x > y, '>')

    def __ge__(self, other: Union['Factor', float]) -> 'Factor':
        return self._comparison_op(other, lambda x, y: x >= y, '>=')

    def __eq__(self, other: Union['Factor', float]) -> 'Factor':
        return self._comparison_op(other, lambda x, y: x == y, '==')

    def __ne__(self, other: Union['Factor', float]) -> 'Factor':
        return self._comparison_op(other, lambda x, y: x != y, '!=')

    def to_weights(self, date: Optional[Union[str, pd.Timestamp]] = None) -> dict:
        if date is None:
            target_date = self.data['timestamp'].max()
        else:
            target_date = pd.to_datetime(date)
        
        pivot = self.data.pivot(index='timestamp', columns='symbol', values='factor')
        
        first_valid_date = pivot.dropna(how='any').index.min()
        
        if first_valid_date is not None and target_date >= first_valid_date:
            target_row = pivot.loc[target_date] if target_date in pivot.index else None
            if target_row is not None:
                nan_symbols = target_row[target_row.isna()].index.tolist()
                if nan_symbols:
                    warnings.warn(
                        f"Factor has NaN values on {target_date.strftime('%Y-%m-%d')}: {nan_symbols}. "
                        f"Consider using ts_backfill() to handle missing values."
                    )
        
        date_data = self.data[self.data['timestamp'] == target_date]
        if date_data.empty:
            return {}
        
        factors = date_data.set_index('symbol')['factor'].dropna()
        if factors.empty:
            return {}
        
        long_sum = factors[factors > 0].sum()
        short_sum = factors[factors < 0].sum()
        total_sum = long_sum + short_sum
        is_already_signal = (np.isclose(long_sum, 0.5, atol=1e-2) and 
                            np.isclose(short_sum, -0.5, atol=1e-2) and
                            np.isclose(total_sum, 0.0, atol=1e-2))
        
        if is_already_signal:
            weights = factors
        else:
            demeaned = factors - factors.mean()
            abs_sum = np.abs(demeaned).sum()
            
            if abs_sum < 1e-10:
                return {}
            
            weights = demeaned / abs_sum
        
        return weights.to_dict()

    def to_df(self) -> pd.DataFrame:
        return self.data.copy()
    
    def to_csv(self, path: str) -> str:
        self.data.to_csv(path, index=False)
        return path
    
    @property
    def symbols(self) -> List[str]:
        return self.data['symbol'].unique().tolist()
    
    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.data['timestamp'].unique())
    
    def __len__(self) -> int:
        return len(self.data)
    
    def info(self) -> None:
        from .console import print
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
        n_symbols = self.data['symbol'].nunique()
        return f"Factor({self.name}): {len(self.data)} obs, {n_symbols} symbols"

    def show(self, symbol: Optional[str] = None, figsize: tuple = (12, 6), 
             title: Optional[str] = None) -> None:
        from .plot import FactorPlotter
        plotter = FactorPlotter(self)
        plotter.plot(symbol, figsize, title)