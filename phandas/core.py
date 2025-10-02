"""
Core factor computation engine for quantitative analysis.

Provides efficient factor matrix operations with pandas-like API.

The Factor class supports:
- Time-series operations (rolling windows, delays, correlations)
- Cross-sectional operations (ranking, normalization, scaling)
- Neutralization (group, vector, regression)
- Mathematical operations (arithmetic, power, logarithms)
- Method chaining for complex factor expressions
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Union, Optional, Callable, List, TYPE_CHECKING
from scipy.stats import rankdata, norm, uniform, cauchy

if TYPE_CHECKING:
    from .panel import Panel


class Factor:
    """
    Professional factor matrix for quantitative analysis.
    
    Internal format: MultiIndex DataFrame with (timestamp, symbol) index.
    Supports method chaining and vectorized operations.
    """
    
    def __init__(self, data: Union[pd.DataFrame, str], name: Optional[str] = None):
        """
        Initialize factor matrix.
        
        Parameters
        ----------
        data : DataFrame or str
            Factor data or CSV path with [timestamp, symbol, factor] columns
        name : str, optional
            Factor name for identification
        """
        if isinstance(data, str):
            df = pd.read_csv(data, parse_dates=['timestamp'])
        else:
            df = data.copy()
        
        # Convert to MultiIndex format if needed
        if isinstance(df.index, pd.MultiIndex):
            # Already MultiIndex
            if 'factor' not in df.columns:
                # Take first column as factor
                df = df.iloc[:, [0]].copy()
                df.columns = ['factor']
            self.data = df[['factor']].copy()
        else:
            # Convert from flat format to MultiIndex
            # Standardize column names
            if len(df.columns) == 3 and 'factor' not in df.columns:
                df.columns = ['timestamp', 'symbol', 'factor']
            elif 'factor' not in df.columns:
                factor_cols = [col for col in df.columns 
                              if col not in ['timestamp', 'symbol']]
                if not factor_cols:
                    raise ValueError("No factor column found")
                df = df[['timestamp', 'symbol', factor_cols[0]]]
                df.columns = ['timestamp', 'symbol', 'factor']
            
            # Set MultiIndex
            df = df[['timestamp', 'symbol', 'factor']].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index(['timestamp', 'symbol']).sort_index()
            self.data = df
            
        self.name = name or 'factor'
    
    # ==================== Cross-Sectional Operations ====================
    # These operations work within each timestamp across all symbols
    
    def rank(self) -> 'Factor':
        """Cross-sectional rank within each timestamp."""
        result = self.data.copy()
        result['factor'] = pd.to_numeric(result['factor'], errors='coerce')
        
        if result['factor'].isna().all():
            raise ValueError("All factor values are NaN")
        
        def safe_rank(group):
            try:
                valid_mask = group.notna()
                if valid_mask.sum() == 0:
                    return group
                
                ranked = group.copy()
                ranked[valid_mask] = group[valid_mask].rank(method='min', pct=True)
                return ranked
            except Exception:
                return pd.Series(np.nan, index=group.index)
        
        result['factor'] = (result.groupby(level='timestamp')['factor']
                           .transform(safe_rank))
        
        return Factor(result, f"rank({self.name})")
    
    # ==================== Time-Series Operations ====================
    # These operations work along the time axis for each symbol
    
    def ts_rank(self, window: int) -> 'Factor':
        """Rolling time-series rank within window."""
        if window <= 0:
            raise ValueError("Window must be positive")
        
        def safe_ts_rank(x: pd.Series) -> float:
            try:
                if x.notna().sum() < window:
                    return np.nan
                
                ranks = rankdata(x.to_numpy(), method='min')
                current_rank = ranks[-1]
                return current_rank / len(ranks)
            except Exception:
                return np.nan
        
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
        result = self._apply_rolling('product', window)
        return Factor(result, f"ts_product({self.name},{window})")

    def ts_quantile(self, window: int, driver: str = "gaussian") -> 'Factor':
        """Calculates ts_rank and applies an inverse cumulative density function from a driver distribution.

        Parameters
        ----------
        window : int
            Number of periods for rolling calculation.
        driver : str, optional
            Distribution driver for the inverse cumulative density function. 
            Possible values: "gaussian", "uniform", "cauchy". Default is "gaussian".

        Returns
        -------
        Factor
            Factor object with quantile-transformed values.
        """
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
            
        def count_nans_func(s):
            return s.isna().sum()
            
        result = self._apply_rolling(count_nans_func, window)
        return Factor(result, f"ts_count_nans({self.name},{window})")
        
    def ts_mean(self, window: int) -> 'Factor':
        """Returns average value of x for the past d days."""
        result = self._apply_rolling('mean', window)
        return Factor(result, f"ts_mean({self.name},{window})")
        
    def ts_median(self, window: int) -> 'Factor':
        """Returns median value of x for the past d days."""
        if window <= 0:
            raise ValueError("Window must be positive")
            
        def safe_median(s):
            if s.isna().all():
                return np.nan
            return s.median()
            
        result = self._apply_rolling(safe_median, window)
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
            max_idx = s.argmax()
            return (len(s) - 1) - max_idx
            
        result = self._apply_rolling(safe_arg_max, window)
        return Factor(result, f"ts_arg_max({self.name},{window})")
        
    def ts_arg_min(self, window: int) -> 'Factor':
        """Returns relative index of min value in time series for past d days."""
        if window <= 0:
            raise ValueError("Window must be positive")
            
        def safe_arg_min(s):
            if s.isna().all():
                return np.nan
            min_idx = s.argmin()
            return (len(s) - 1) - min_idx
            
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
        """Returns (x - ts_min(x, d)) / (ts_max(x, d) - ts_min(x, d)) + constant.

        This operator is similar to scale down operator but acts in time series space.
        """
        if window <= 0:
            raise ValueError("Window must be positive")

        min_factor = self.ts_min(window)
        max_factor = self.ts_max(window)
        denominator = max_factor - min_factor
        result = self.subtract(min_factor)
        
        scaled_result_data = pd.merge(result.data, denominator.data, 
                                      on=['timestamp', 'symbol'], 
                                      suffixes=('_num', '_den'))
        
        scaled_result_data['factor'] = scaled_result_data.apply(
            lambda row: np.nan if row['factor_den'] == 0 else row['factor_num'] / row['factor_den'], axis=1
        )
        
        final_result_data = scaled_result_data[['timestamp', 'symbol', 'factor']].copy()
        final_result_data['factor'] += constant

        return Factor(final_result_data, f"ts_scale({self.name},{window},{constant})")
    
    def ts_zscore(self, window: int) -> 'Factor':
        """Returns Z-score: (x - ts_mean(x,d)) / ts_std_dev(x,d)."""
        if window <= 0:
            raise ValueError("Window must be positive")
        
        mean_factor = self.ts_mean(window)
        std_dev_factor = self.ts_std_dev(window)
        numerator = self.subtract(mean_factor)
        
        merged = pd.merge(numerator.data, std_dev_factor.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('_num', '_den'))
        
        merged['factor'] = merged.apply(
            lambda row: np.nan if row['factor_den'] == 0 else row['factor_num'] / row['factor_den'], axis=1
        )
        
        result_data = merged[['timestamp', 'symbol', 'factor']].copy()
        return Factor(result_data, f"ts_zscore({self.name},{window})")

    # ==================== Cross-Sectional Statistical Transforms ====================
    # These operations transform values within each timestamp
    
    def quantile(self, driver: str = "gaussian", sigma: float = 1.0) -> 'Factor':
        """Cross-sectional quantile transformation with specified driver and scale."
        """
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

        result['factor'] = result.groupby(level='timestamp')['factor'].transform(apply_quantile)

        return Factor(result, f"quantile({self.name},driver={driver},sigma={sigma})")

    def scale(self, scale: float = 1.0, longscale: float = -1.0, shortscale: float = -1.0) -> 'Factor':
        """Scales input to booksize, with optional separate scaling for long and short positions."""
        result = self.data.copy()

        def apply_scale(group: pd.Series):
            if longscale != -1.0 or shortscale != -1.0:
                long_positions = group[group > 0]
                short_positions = group[group < 0]
                
                scaled_group = group.copy()
                
                if not long_positions.empty and longscale > 0:
                    long_abs_sum = long_positions.abs().sum()
                    if long_abs_sum > 0:
                        scaled_group[group > 0] = (long_positions / long_abs_sum) * longscale
                
                if not short_positions.empty and shortscale > 0:
                    short_abs_sum = short_positions.abs().sum()
                    if short_abs_sum > 0:
                        scaled_group[group < 0] = (short_positions / short_abs_sum) * shortscale
                
                return scaled_group
            else:
                abs_sum = group.abs().sum()
                if abs_sum == 0:
                    return group
                return (group / abs_sum) * scale

        result['factor'] = result.groupby(level='timestamp')['factor'].transform(apply_scale)

        return Factor(result, f"scale({self.name},scale={scale},longscale={longscale},shortscale={shortscale})")

    def zscore(self) -> 'Factor':
        """Computes the cross-sectional Z-score of the factor."""
        return self.normalize(useStd=True)

    # ==================== Neutralization Operations ====================
    # Remove unwanted exposures to other factors or groups
    
    def group_neutralize(self, group_data: 'Factor') -> 'Factor':
        """
        Neutralizes the factor against specified groups (e.g., industry).

        Parameters
        ----------
        group_data : Factor
            A Factor object where the 'factor' column contains group labels
            (e.g., industry names, sectors).

        Returns
        -------
        Factor
            A new Factor object with group-neutralized values.
        """
        if not isinstance(group_data, Factor):
            raise TypeError("group_data must be a Factor object.")

        merged = pd.merge(self.data, group_data.data, 
                         left_index=True, right_index=True,
                         suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data between factor and group data.")

        # Create temporary columns for groupby
        temp_df = merged.reset_index()
        temp_df['factor'] = temp_df.groupby(['timestamp', 'factor_group'])['factor'].transform(
            lambda x: x - x.mean()
        )
        
        result_data = temp_df.set_index(['timestamp', 'symbol'])[['factor']]
        return Factor(result_data, name=f"group_neutralize({self.name},{group_data.name})")

    def vector_neut(self, other: 'Factor') -> 'Factor':
        """
        Neutralizes the factor against another factor using vector projection.

        Parameters
        ----------
        other : Factor
            The factor to neutralize against.

        Returns
        -------
        Factor
            A new Factor object with values orthogonal to the other factor.
        """
        if not isinstance(other, Factor):
            raise TypeError("other must be a Factor object.")

        merged = pd.merge(self.data, other.data,
                         left_index=True, right_index=True,
                         suffixes=('_target', '_neut'))

        if merged.empty:
            raise ValueError("No common data for vector neutralization.")

        def neutralize_single_date(group):
            x = group['factor_target'].values
            y = group['factor_neut'].values
            
            if np.all(y == 0) or np.dot(y, y) == 0:
                return pd.Series(x, index=group.index)

            projection = (np.dot(x, y) / np.dot(y, y)) * y
            neutralized_x = x - projection
            
            return pd.Series(neutralized_x, index=group.index)

        neutralized_series = merged.groupby(level='timestamp').apply(neutralize_single_date).reset_index(level=0, drop=True)
        
        result_data = merged[['factor_target']].copy()
        result_data.columns = ['factor']
        result_data['factor'] = neutralized_series
        
        return Factor(result_data, name=f"vector_neut({self.name},{other.name})")

    def regression_neut(self, neut_factors: Union['Factor', List['Factor']]) -> 'Factor':
        """
        Neutralizes the factor against one or more factors using OLS regression.

        Parameters
        ----------
        neut_factors : Factor or list of Factor
            Factor(s) to use as independent variables in the regression.

        Returns
        -------
        Factor
            A new Factor object containing the residuals of the regression.
        """
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
                              left_index=True, right_index=True, how='inner')

        if merged.empty:
            raise ValueError("No common data for regression neutralization.")
        
        def get_residuals(group):
            Y = group[self.name or 'target']
            X = group[neut_names]
            
            valid_idx = Y.notna() & X.notna().all(axis=1)
            if valid_idx.sum() < 2:
                return pd.Series(np.nan, index=group.index)
                
            Y = Y[valid_idx]
            X = X[valid_idx]

            if X.empty:
                return pd.Series(np.nan, index=group.index)

            X_const = sm.add_constant(X)
            if np.linalg.cond(X_const) > 1e10:
                return pd.Series(np.nan, index=group.index)

            model = sm.OLS(Y, X_const).fit()
            residuals = pd.Series(np.nan, index=group.index)
            residuals[valid_idx] = model.resid
            return residuals

        residuals_series = merged.groupby(level='timestamp').apply(get_residuals, include_groups=False).reset_index(level=0, drop=True)
        
        result_data = merged[[self.name or 'target']].copy()
        result_data.columns = ['factor']
        result_data['factor'] = residuals_series

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
                    normalized_group = pd.Series(np.nan, index=group.index)
                else:
                    normalized_group /= std_val
            
            if limit != 0.0:
                normalized_group = normalized_group.clip(lower=-limit, upper=limit)
            
            return normalized_group

        result['factor'] = result.groupby(level='timestamp')['factor'].transform(apply_normalize)

        return Factor(result, f"normalize({self.name},useStd={useStd},limit={limit})")

    # ==================== Time-Series Advanced Operations ====================
    # Backfilling, decay, and other advanced time-series transformations
    
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
        """
        Returns exponential decay weighted average over rolling window.
        
        Formula: (x[t] + x[t-1]*f + x[t-2]*f^2 + ...) / (1 + f + f^2 + ...)
        where newer values receive higher weights.
        
        Parameters
        ----------
        window : int
            Rolling window size
        factor : float, default 1.0
            Decay factor (0 < factor < 1). Smaller values decay faster.
        nan : bool, default True
            If True, ignore NaNs in calculation. If False, treat NaNs as 0.
        
        Returns
        -------
        Factor
            Exponentially weighted factor values
        """
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
            
            current_weights = np.array([factor**i for i in range(len(valid_s))])
            current_weights = current_weights[::-1]
            
            weight_sum = current_weights.sum()
            if weight_sum == 0:
                return np.nan if nan else 0.0
            
            decayed_sum = (valid_s * current_weights).sum()
            return decayed_sum / weight_sum
        
        result = self._apply_rolling(decay_func, window)
        
        return Factor(result, f"ts_decay_exp_window({self.name},{window},{factor},{nan})")
    
    def ts_decay_linear(self, window: int, dense: bool = False) -> 'Factor':
        """
        Returns linear decay weighted average over rolling window.
        
        Applies linearly decreasing weights to past values.
        Weights: d, d-1, d-2, ..., 1 (older values get less weight).
        
        Parameters
        ----------
        window : int
            Rolling window size (d)
        dense : bool, default False
            If False (sparse mode), treat NaNs as 0.
            If True (dense mode), only average non-NaN values.
        
        Returns
        -------
        Factor
            Linearly weighted factor values
        """
        if window <= 0:
            raise ValueError("Window must be positive")
            
        def linear_decay_func(s):
            current_s = s.copy()
            if not dense:
                current_s = current_s.fillna(0) # sparse mode: treat NaN as 0
            
            if current_s.isna().all():
                return np.nan # In dense mode, if all are NaN, result is NaN
                
            # Weights are d, d-1, ..., 1
            weights = np.arange(1, len(current_s) + 1)[::-1] 
            
            # Only apply weights to non-NaN values if dense mode (otherwise NaNs are 0)
            if dense:
                # For dense mode, we still ignore NaNs for sum, but don't fill with 0.
                # Only include valid values in weighted sum and weight sum.
                valid_mask = current_s.notna()
                if not valid_mask.any():
                    return np.nan
                
                weighted_sum = (current_s[valid_mask] * weights[valid_mask]).sum()
                weight_sum = weights[valid_mask].sum()
            else:
                weighted_sum = (current_s * weights).sum()
                weight_sum = weights.sum()
                
            if weight_sum == 0:
                return np.nan
                
            return weighted_sum / weight_sum
            
        result = self._apply_rolling(linear_decay_func, window)
        return Factor(result, f"ts_decay_linear({self.name},{window},dense={dense})")
    
    # ==================== Time-Series Basic Transformations ====================
    # Simple time shifts and differences
    
    def ts_delay(self, window: int) -> 'Factor':
        """Returns x value d days ago."""
        result = self._apply_groupby('shift', window)
        return Factor(result, f"ts_delay({self.name},{window})")
        
    def ts_delta(self, window: int) -> 'Factor':
        """Returns x - ts_delay(x, d)."""
        result = self._apply_groupby('diff', window)
        return Factor(result, f"ts_delta({self.name},{window})")
        
    def returns(self, periods: int = 1) -> 'Factor':
        """Percentage returns over periods."""
        result = self._apply_groupby('pct_change', periods)
        return Factor(result, f"returns({self.name},{periods})")
    
    # ==================== Mathematical Operations ====================
    # Basic arithmetic operations with factors or scalars
    
    def add(self, other: Union['Factor', float], filter_nan: bool = False) -> 'Factor':
        """Addition with factor or scalar, with optional NaN filtering."""
        if isinstance(other, Factor):
            if filter_nan:
                merged = pd.merge(self.data, other.data,
                                  left_index=True, right_index=True,
                                  suffixes=('_x', '_y'), how='outer')
                merged['factor_x'] = merged['factor_x'].fillna(0)
                merged['factor_y'] = merged['factor_y'].fillna(0)
                merged['factor'] = merged['factor_x'] + merged['factor_y']
                result = merged[['factor']]
                return Factor(result, f"add({self.name},{other.name},filter_nan={filter_nan})")
            else:
                result = self._binary_op(other, lambda x, y: x + y)
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
            if filter_nan:
                merged = pd.merge(self.data, other.data,
                                  left_index=True, right_index=True,
                                  suffixes=('_x', '_y'), how='outer')
                merged['factor_x'] = merged['factor_x'].fillna(0)
                merged['factor_y'] = merged['factor_y'].fillna(0)
                merged['factor'] = merged['factor_x'] - merged['factor_y']
                result = merged[['factor']]
                return Factor(result, f"subtract({self.name},{other.name},filter_nan={filter_nan})")
            else:
                result = self._binary_op(other, lambda x, y: x - y)
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
            if filter_nan:
                merged = pd.merge(self.data, other.data,
                                  left_index=True, right_index=True,
                                  suffixes=('_x', '_y'), how='outer')
                merged['factor_x'] = merged['factor_x'].fillna(1)
                merged['factor_y'] = merged['factor_y'].fillna(1)
                merged['factor'] = merged['factor_x'] * merged['factor_y']
                result = merged[['factor']]
                return Factor(result, f"multiply({self.name},{other.name},filter_nan={filter_nan})")
            else:
                result = self._binary_op(other, lambda x, y: x * y)
                return Factor(result, f"({self.name}*{other.name})")
        else:
            result = self.data.copy()
            if filter_nan:
                result['factor'] = result['factor'].fillna(1) * other
            else:
                result['factor'] *= other
            return Factor(result, f"({self.name}*{other})")
    
    # ==================== Non-Linear Mathematical Operations ====================
    # Logarithms, powers, roots, and sign operations
    
    def log(self, base: Optional[float] = None) -> 'Factor':
        """Logarithm of factor with optional base."""
        result = self.data.copy()
        if base is None:
            # Natural logarithm (default)
            result['factor'] = np.log(result['factor'])
            return Factor(result, f"log({self.name})")
        else:
            # Arbitrary base using change of base formula: log_b(x) = ln(x) / ln(b)
            result['factor'] = np.log(result['factor']) / np.log(base)
            return Factor(result, f"log({self.name},base={base})")
    
    def ln(self) -> 'Factor':
        """Natural logarithm of factor (explicit ln function)."""
        result = self.data.copy()
        result['factor'] = np.log(result['factor'])
        return Factor(result, f"ln({self.name})")
    
    def sqrt(self) -> 'Factor':
        """Square root of factor values."""
        result = self.data.copy()
        result['factor'] = np.sqrt(result['factor'])
        return Factor(result, f"sqrt({self.name})")
    
    def s_log_1p(self) -> 'Factor':
        """Confine factor values to a shorter range using sign(x) * log(1 + abs(x))."""
        result = self.data.copy()
        # Use np.sign for sign handling, np.log1p(x) equivalent to log(1 + x)
        result['factor'] = np.sign(result['factor']) * np.log1p(np.abs(result['factor']))
        return Factor(result, f"s_log_1p({self.name})")
    
    def sign(self) -> 'Factor':
        """Returns the sign of factor values. NaN input returns NaN."""
        result = self.data.copy()
        result['factor'] = np.sign(result['factor'])
        return Factor(result, f"sign({self.name})")
    
    def signed_power(self, exponent: Union['Factor', float]) -> 'Factor':
        """x raised to the power of y such that final result preserves sign of x."""
        result = self.data.copy()
        
        if isinstance(exponent, Factor):
            # Merge data and calculate sign(x) * (abs(x) ** y)
            merged = pd.merge(self.data, exponent.data,
                             left_index=True, right_index=True,
                             suffixes=('_x', '_y'), how='outer')
            
            # Align and ensure correct data types
            x_values = merged['factor_x']
            y_values = merged['factor_y']
            
            # Calculate signed power
            final_values = np.sign(x_values) * (np.abs(x_values) ** y_values)
            
            result = merged[['factor_x']].copy()
            result.columns = ['factor']
            result['factor'] = final_values
            
            return Factor(result, f"signed_power({self.name},{exponent.name})")
        else:
            # Calculate signed_power for scalar exponent
            result['factor'] = np.sign(result['factor']) * (np.abs(result['factor']) ** exponent)
            return Factor(result, f"signed_power({self.name},{exponent})")
    
    def power(self, exponent: Union['Factor', float]) -> 'Factor':
        """Factor to the power of exponent."""
        if isinstance(exponent, Factor):
            result = self._binary_op(exponent, lambda x, y: x ** y)
            return Factor(result, f"({self.name}**{exponent.name})")
        else:
            result = self.data.copy()
            result['factor'] = result['factor'] ** exponent
            return Factor(result, f"({self.name}**{exponent})")
    
    def abs(self) -> 'Factor':
        """Absolute value of factor."""
        result = self.data.copy()
        result['factor'] = np.abs(result['factor'])
        return Factor(result, f"abs({self.name})")
    
    def inverse(self) -> 'Factor':
        """Inverse: 1 / factor."""
        result = self.data.copy()
        result['factor'] = 1 / result['factor']
        return Factor(result, f"inverse({self.name})")

    def where(self, cond: 'Factor', other: Union['Factor', float] = np.nan) -> 'Factor':
        """
        Equivalent to pandas.Series.where.
        Replace values where the condition is False.
        """
        if not isinstance(cond, Factor):
            raise TypeError("cond must be a Factor object.")

        # Prepare the data, setting index for alignment
        self_s = self.data.set_index(['timestamp', 'symbol'])['factor']
        cond_s = cond.data.set_index(['timestamp', 'symbol'])['factor']
        
        # Ensure condition is boolean, treating NaNs as False
        cond_s = cond_s.fillna(False).astype(bool)

        other_val = other
        if isinstance(other, Factor):
            other_val = other.data.set_index(['timestamp', 'symbol'])['factor']

        # Perform the where operation. Pandas handles alignment based on index.
        result_s = self_s.where(cond_s, other_val)

        # Convert back to standard Factor format
        result_df = result_s.reset_index()
        
        return Factor(result_df, name=f"where({self.name})")
    
    # ==================== Python Operator Overloading ====================
    # Enable intuitive factor expressions (e.g., factor1 + factor2)
    
    def __neg__(self) -> 'Factor':
        """Unary negation: -factor"""
        return self.multiply(-1)
    
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
        """Power: factor ** other"""
        return self.power(other)
    
    def __rpow__(self, other: Union['Factor', float]) -> 'Factor':
        """Right power: other ** factor"""
        if isinstance(other, Factor):
            return other.power(self)
        else:
            result = self.data.copy()
            result['factor'] = other ** result['factor']
            return Factor(result, f"({other}**{self.name})")
    
    def __truediv__(self, other: Union['Factor', float]) -> 'Factor':
        """Division: factor / other"""
        if isinstance(other, Factor):
            result = self._binary_op(other, lambda x, y: x / y)
            return Factor(result, f"({self.name}/{other.name})")
        else:
            result = self.data.copy()
            result['factor'] /= other
            return Factor(result, f"({self.name}/{other})")
    
    def __rtruediv__(self, other: Union['Factor', float]) -> 'Factor':
        """Right division: other / factor"""
        if isinstance(other, Factor):
            return other.__truediv__(self)
        else:
            result = self.data.copy()
            result['factor'] = other / result['factor']
            return Factor(result, f"({other}/{self.name})")

    # ==================== Comparison Operators ====================
    # Boolean comparison operations returning Factor objects
    
    def __lt__(self, other: Union['Factor', float]) -> 'Factor':
        """Less than: factor < other"""
        if isinstance(other, Factor):
            result_df = self._binary_op(other, lambda x, y: x < y)
        else:
            result_df = self.data.copy()
            result_df['factor'] = result_df['factor'] < other
        return Factor(result_df, f"({self.name}<{getattr(other, 'name', other)})")

    def __le__(self, other: Union['Factor', float]) -> 'Factor':
        """Less than or equal to: factor <= other"""
        if isinstance(other, Factor):
            result_df = self._binary_op(other, lambda x, y: x <= y)
        else:
            result_df = self.data.copy()
            result_df['factor'] = result_df['factor'] <= other
        return Factor(result_df, f"({self.name}<={getattr(other, 'name', other)})")

    def __gt__(self, other: Union['Factor', float]) -> 'Factor':
        """Greater than: factor > other"""
        if isinstance(other, Factor):
            result_df = self._binary_op(other, lambda x, y: x > y)
        else:
            result_df = self.data.copy()
            result_df['factor'] = result_df['factor'] > other
        return Factor(result_df, f"({self.name}>{getattr(other, 'name', other)})")

    def __ge__(self, other: Union['Factor', float]) -> 'Factor':
        """Greater than or equal to: factor >= other"""
        if isinstance(other, Factor):
            result_df = self._binary_op(other, lambda x, y: x >= y)
        else:
            result_df = self.data.copy()
            result_df['factor'] = result_df['factor'] >= other
        return Factor(result_df, f"({self.name}>={getattr(other, 'name', other)})")

    def __eq__(self, other: Union['Factor', float]) -> 'Factor':
        """Equal to: factor == other"""
        if isinstance(other, Factor):
            result_df = self._binary_op(other, lambda x, y: x == y)
        else:
            result_df = self.data.copy()
            result_df['factor'] = result_df['factor'] == other
        return Factor(result_df, f"({self.name}=={getattr(other, 'name', other)})")

    def __ne__(self, other: Union['Factor', float]) -> 'Factor':
        """Not equal to: factor != other"""
        if isinstance(other, Factor):
            result_df = self._binary_op(other, lambda x, y: x != y)
        else:
            result_df = self.data.copy()
            result_df['factor'] = result_df['factor'] != other
        return Factor(result_df, f"({self.name}!={getattr(other, 'name', other)})")
    
    def maximum(self, other: Union['Factor', float]) -> 'Factor':
        """Element-wise maximum between this factor and another factor or scalar."""
        if isinstance(other, Factor):
            result_df = self._binary_op(other, lambda x, y: x if x > y else y)
        else:
            result_df = self.data.copy()
            result_df['factor'] = result_df['factor'].apply(lambda x: x if x > other else other)
        return Factor(result_df, f"max({self.name},{getattr(other, 'name', other)})")
    
    def minimum(self, other: Union['Factor', float]) -> 'Factor':
        """Element-wise minimum between this factor and another factor or scalar."""
        if isinstance(other, Factor):
            result_df = self._binary_op(other, lambda x, y: x if x < y else y)
        else:
            result_df = self.data.copy()
            result_df['factor'] = result_df['factor'].apply(lambda x: x if x < other else other)
        return Factor(result_df, f"min({self.name},{getattr(other, 'name', other)})")
    
    # ==================== Correlation and Regression ====================
    # Time-series correlation, covariance, and regression operations
    
    def ts_corr(self, other: 'Factor', window: int) -> 'Factor':
        """
        Returns Pearson correlation of two factors for the past d days.
        
        Measures linear relationship between variables. Most effective when
        variables are normally distributed and relationship is linear.
        
        Parameters
        ----------
        other : Factor
            The other factor to correlate with
        window : int
            Number of periods for rolling correlation
            
        Returns
        -------
        Factor
            Rolling correlation values between -1 and 1
        """
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if not isinstance(other, Factor):
            raise TypeError("Other must be a Factor object")
        
        # Merge factor data
        try:
            merged = pd.merge(self.data, other.data,
                             left_index=True, right_index=True,
                             suffixes=('_x', '_y'), how='inner')
        except Exception as e:
            raise ValueError(f"Failed to merge factor data: {e}")
        
        if merged.empty:
            raise ValueError("No common data between factors")
        
        # Calculate rolling correlation by symbol
        result_series = []
        
        for symbol in merged.index.get_level_values('symbol').unique():
            try:
                symbol_data = merged.xs(symbol, level='symbol')
                
                # Ensure correct data types
                x_values = pd.to_numeric(symbol_data['factor_x'], errors='coerce')
                y_values = pd.to_numeric(symbol_data['factor_y'], errors='coerce')
                
                # Calculate rolling correlation
                corr_values = x_values.rolling(window, min_periods=window).corr(y_values)
                
                # Handle invalid correlation values
                corr_values = corr_values.where(
                    (corr_values >= -1) & (corr_values <= 1), np.nan
                )
                
                result_series.append(corr_values)
                
            except Exception:
                # Add NaN values on error
                symbol_data = merged.xs(symbol, level='symbol')
                result_series.append(pd.Series(np.nan, index=symbol_data.index))
        
        if result_series:
            result = pd.concat(result_series)
            result = result.to_frame('factor')
        else:
            result = merged[['factor_x']].copy()
            result.columns = ['factor']
            result['factor'] = np.nan
        
        return Factor(result, f"ts_corr({self.name},{other.name},{window})")
    
    def ts_covariance(self, other: 'Factor', window: int) -> 'Factor':
        """Returns covariance of self and other for the past d days."""
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if not isinstance(other, Factor):
            raise TypeError("Other must be a Factor object")
            
        # Merge factor data
        merged = pd.merge(self.data, other.data,
                         left_index=True, right_index=True,
                         suffixes=('_x', '_y'), how='inner')
        
        if merged.empty:
            raise ValueError("No common data between factors")
        
        # Calculate rolling covariance by symbol
        result_series = []
        
        for symbol in merged.index.get_level_values('symbol').unique():
            symbol_data = merged.xs(symbol, level='symbol')
            
            x_values = pd.to_numeric(symbol_data['factor_x'], errors='coerce')
            y_values = pd.to_numeric(symbol_data['factor_y'], errors='coerce')
            
            # Calculate rolling covariance
            cov_values = x_values.rolling(window, min_periods=window).cov(y_values)
            
            result_series.append(cov_values)
            
        if result_series:
            result = pd.concat(result_series)
            result = result.to_frame('factor')
        else:
            result = merged[['factor_x']].copy()
            result.columns = ['factor']
            result['factor'] = np.nan
            
        return Factor(result, f"ts_covariance({self.name},{other.name},{window})")

    def ts_regression(self, x_factor: 'Factor', window: int, lag: int = 0, rettype: int = 0) -> 'Factor':
        """
        Performs rolling OLS linear regression and returns various parameters based on rettype.

        Parameters
        ----------
        x_factor : Factor
            The independent variable (X) Factor.
        window : int
            Number of periods for rolling calculation.
        lag : int, optional
            Lag for the independent variable (X). Default is 0.
        rettype : int, optional
            Determines the regression parameter to return:
            0: Error Term (y_i - y_estimate)
            1: y-intercept (alpha)
            2: slope (beta)
            3: y-estimate
            4: Sum of Squares of Error (SSE)
            5: Sum of Squares of Total (SST)
            6: R-Square
            7: Mean Square Error (MSE)
            8: Standard Error of Beta
            9: Standard Error of Alpha
            Default is 0.

        Returns
        -------
        Factor
            A new Factor object with the requested regression parameter values.
        """
        if window <= 0:
            raise ValueError("Window must be positive")
        if not isinstance(x_factor, Factor):
            raise TypeError("x_factor must be a Factor object.")
        if not 0 <= rettype <= 9:
            raise ValueError("rettype must be between 0 and 9.")

        # Prepare data: dependent variable (self) and independent variable (x_factor)
        y_data = self.data.rename(columns={'factor': 'y'})
        x_data = x_factor.data.rename(columns={'factor': 'x'})

        if lag > 0:
            x_data = Factor(x_data, name=x_factor.name).ts_delay(lag).data.rename(columns={'factor': 'x'})

        merged = pd.merge(y_data, x_data, left_index=True, right_index=True, how='inner')

        results_list = []

        for symbol in merged.index.get_level_values('symbol').unique():
            symbol_df = merged.xs(symbol, level='symbol').copy()
            symbol_df['y'] = pd.to_numeric(symbol_df['y'], errors='coerce')
            symbol_df['x'] = pd.to_numeric(symbol_df['x'], errors='coerce')
            original_index = symbol_df.index

            def _rolling_regression(series_window):
                if len(series_window) < window or series_window['y'].isna().all() or series_window['x'].isna().all():
                    return pd.Series([np.nan] * 10)

                clean_window = series_window.dropna(subset=['y', 'x'])

                if len(clean_window) < 2:
                    return pd.Series([np.nan] * 10)

                Y = clean_window['y']
                X = sm.add_constant(clean_window['x'])

                try:
                    model = sm.OLS(Y, X).fit()

                    alpha = model.params.get('const', np.nan)
                    beta = model.params.get('x', np.nan)
                    y_estimate = model.predict(X).iloc[-1] # Get the last predicted value

                    # Error term for the last observation
                    error_term = Y.iloc[-1] - y_estimate if pd.notna(Y.iloc[-1]) and pd.notna(y_estimate) else np.nan

                    # Sum of Squares calculations
                    SSE = np.sum(model.resid ** 2) # Sum of squared errors
                    SST = np.sum((Y - Y.mean()) ** 2) # Total sum of squares

                    R_squared = model.rsquared if SST > 0 else np.nan
                    MSE = SSE / model.df_resid if model.df_resid > 0 else np.nan

                    # Standard Errors
                    std_err_beta = model.bse.get('x', np.nan)
                    std_err_alpha = model.bse.get('const', np.nan)

                    return pd.Series([
                        error_term,
                        alpha,
                        beta,
                        y_estimate,
                        SSE,
                        SST,
                        R_squared,
                        MSE,
                        std_err_beta,
                        std_err_alpha
                    ])
                except Exception:
                    return pd.Series([np.nan] * 10)

            # The rolling apply will return a Series for each window. Extract the last value for each row.
            # We need to apply this rolling window to each symbol's data.
            # Create an empty DataFrame to store the results for this symbol
            symbol_results_df = pd.DataFrame(index=symbol_df.index, columns=range(10), dtype=float)
            
            for i in range(window - 1, len(symbol_df)):
                window_data = symbol_df.iloc[i - window + 1 : i + 1]
                result_series = _rolling_regression(window_data)
                symbol_results_df.loc[symbol_df.index[i]] = result_series.values

            # Assign the requested rettype column
            symbol_df['factor'] = symbol_results_df.iloc[:, rettype]
            results_list.append(symbol_df[['factor']])

        if results_list:
            final_result_df = pd.concat(results_list)
        else:
            final_result_df = merged[['y']].copy()
            final_result_df.columns = ['factor']
            final_result_df['factor'] = np.nan

        return Factor(final_result_df, name=f"ts_regression({self.name},{x_factor.name},{window},lag={lag},rettype={rettype})")

    # ==================== Internal Utility Methods ====================
    # Helper methods for efficient operations (not part of public API)
    
    def _apply_rolling(self, func: Union[str, Callable], window: int) -> pd.DataFrame:
        """Apply rolling function by symbol."""
        result = self.data.copy()
        if isinstance(func, str):
            if func == 'product':
                rolled = (result.groupby(level='symbol')['factor']
                         .rolling(window, min_periods=window)
                         .apply(np.prod, raw=True))
            else:
                rolled = (result.groupby(level='symbol')['factor']
                         .rolling(window, min_periods=window)
                         .agg(func))
        else:
            rolled = (result.groupby(level='symbol')['factor']
                     .rolling(window, min_periods=window)
                     .apply(func, raw=False))
        
        # Remove extra level from rolling
        rolled = rolled.reset_index(level=0, drop=True)
        result['factor'] = rolled
        return result
    
    def _apply_groupby(self, func: str, *args) -> pd.DataFrame:
        """Apply groupby function by symbol."""
        result = self.data.copy()
        result['factor'] = (result.groupby(level='symbol')['factor']
                           .transform(func, *args))
        return result
    
    def _binary_op(self, other: 'Factor', op: Callable) -> pd.DataFrame:
        """Apply binary operation with another factor (fast index alignment)."""
        result = op(self.data['factor'], other.data['factor'])
        return result.to_frame('factor')
    
    # ==================== Data Access and Information ====================
    # Methods for exporting data and retrieving factor information
    
    def to_csv(self, path: str) -> str:
        """Save to CSV file."""
        self.data.reset_index().to_csv(path, index=False)
        return path
    
    def to_multiindex(self) -> pd.Series:
        """Convert to MultiIndex Series (already in this format)."""
        return self.data['factor']
    
    def to_flat(self) -> pd.DataFrame:
        """Convert to flat DataFrame with timestamp, symbol, factor columns."""
        return self.data.reset_index()
    
    def info(self) -> dict:
        """Get factor information."""
        timestamps = self.data.index.get_level_values('timestamp')
        symbols = self.data.index.get_level_values('symbol')
        return {
            'name': self.name,
            'shape': self.data.shape,
            'time_range': (timestamps.min(), timestamps.max()),
            'symbols': sorted(symbols.unique()),
            'valid_ratio': self.data['factor'].notna().mean()
        }
    
    def __repr__(self):
        n_obs = self.data.shape[0]
        timestamps = self.data.index.get_level_values('timestamp')
        symbols = self.data.index.get_level_values('symbol')
        n_symbols = len(symbols.unique())
        valid_ratio = self.data['factor'].notna().mean()
        time_range = f"{timestamps.min().strftime('%Y-%m-%d')} to {timestamps.max().strftime('%Y-%m-%d')}"
        return (f"Factor(name={self.name}, obs={n_obs}, symbols={n_symbols}, "
               f"valid={valid_ratio:.1%}, period={time_range})")
    
    def __str__(self):
        """User-friendly string representation."""
        n_symbols = len(self.data.index.get_level_values('symbol').unique())
        return f"Factor({self.name}): {self.data.shape[0]} obs, {n_symbols} symbols"


# ==================== Factory Functions ====================
# Convenience functions for creating Factor objects from various sources

def load_factor(data: Union[str, pd.DataFrame, 'Panel'], column: str, name: Optional[str] = None) -> Factor:
    """
    Load factor from data source.
    
    Deprecated: Use panel['column'] or panel.get_factor() instead.
    
    Parameters
    ----------
    data : str, DataFrame, or Panel
        Data source
    column : str
        Column name to extract as factor
    name : str, optional
        Factor name
        
    Returns
    -------
    Factor
        Factor object with specified column
    """
    from .panel import Panel
    
    if isinstance(data, Panel):
        return data.get_factor(column, name)
    elif isinstance(data, str):
        df = pd.read_csv(data, parse_dates=['timestamp'])
    else:
        df = data.copy()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    factor_data = df[['timestamp', 'symbol', column]].copy()
    factor_data.columns = ['timestamp', 'symbol', 'factor']
    
    return Factor(factor_data, name or column)
