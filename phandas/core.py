"""
Core factor computation engine for quantitative analysis.

Provides efficient factor matrix operations with pandas-like API.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Union, Optional, Callable, List
from scipy.stats import rankdata, norm, uniform, cauchy


class Factor:
    """
    Professional factor matrix for quantitative analysis.
    
    Standardized format: timestamp, symbol, factor columns.
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
            
        self.data = df[['timestamp', 'symbol', 'factor']].copy()
        self.data = self.data.sort_values(['timestamp', 'symbol'])
        self.name = name or 'factor'
    
    # Core operations
    def rank(self) -> 'Factor':
        """Cross-sectional rank within each timestamp."""
        result = self.data.copy()
        
        # 確保數據類型正確
        result['factor'] = pd.to_numeric(result['factor'], errors='coerce')
        
        # 檢查是否有有效數據
        if result['factor'].isna().all():
            raise ValueError("All factor values are NaN")
        
        # 分組計算排名，處理 NaN 值
        def safe_rank(group):
            try:
                # 只對非 NaN 值計算排名
                valid_mask = group.notna()
                if valid_mask.sum() == 0:
                    return group  # 如果全是 NaN，保持原樣
                
                ranked = group.copy()
                ranked[valid_mask] = group[valid_mask].rank(method='min', pct=True)
                return ranked
            except Exception as e:
                # 如果出錯，返回 NaN
                return pd.Series(np.nan, index=group.index)
        
        result['factor'] = (result.groupby('timestamp')['factor']
                           .transform(safe_rank))
        
        return Factor(result, f"rank({self.name})")
    
    def ts_rank(self, window: int) -> 'Factor':
        """Rolling time-series rank within window."""
        if window <= 0:
            raise ValueError("Window must be positive")
        
        def safe_ts_rank(x: pd.Series) -> float:
            """
            Calculates rank only if there are enough valid data points.
            """
            try:
                # 只有當窗口內的有效數據點數量達到窗口大小時才計算
                if x.notna().sum() < window:
                    return np.nan
                
                # 使用 scipy.stats.rankdata 計算排名
                ranks = rankdata(x.to_numpy(), method='min')
                
                # 獲取窗口中最後一個值的排名
                current_rank = ranks[-1]
                
                # 轉換為百分位數 (0-1]
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

        # First, calculate ts_rank
        ranked_factor = self.ts_rank(window)

        result_data = ranked_factor.data.copy()
        # Apply PPF, ensuring values are within (0, 1) exclusive for some distributions
        # Rank data is already in (0, 1] range, so clamp to (epsilon, 1-epsilon) if necessary
        epsilon = 1e-6 # A small value to avoid issues with 0 or 1
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
            # argmax returns integer position of the first occurrence of the max
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
            
        # ts_mean already ignores NaNs, so this is straightforward
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

        # Perform the calculation: (x - min) / (max - min) + constant
        # Need to handle division by zero if max_factor == min_factor
        denominator = max_factor - min_factor
        
        # If denominator is 0, the result of (x - min) / (max - min) should be NaN
        # Then add constant to it.
        result = self.subtract(min_factor) # (x - min)
        
        # Use a custom binary operation to handle division by zero
        scaled_result_data = pd.merge(result.data, denominator.data, 
                                      on=['timestamp', 'symbol'], 
                                      suffixes=('_num', '_den'))
        
        scaled_result_data['factor'] = scaled_result_data.apply(
            lambda row: np.nan if row['factor_den'] == 0 else row['factor_num'] / row['factor_den'], axis=1
        )
        
        # Add constant
        final_result_data = scaled_result_data[['timestamp', 'symbol', 'factor']].copy()
        final_result_data['factor'] += constant

        return Factor(final_result_data, f"ts_scale({self.name},{window},{constant})")
    
    def ts_zscore(self, window: int) -> 'Factor':
        """Returns Z-score: (x - ts_mean(x,d)) / ts_std_dev(x,d)."""
        if window <= 0:
            raise ValueError("Window must be positive")
        
        mean_factor = self.ts_mean(window)
        std_dev_factor = self.ts_std_dev(window)
        
        # numerator: (x - ts_mean(x,d))
        numerator = self.subtract(mean_factor)
        
        # denominator: ts_std_dev(x,d)
        # Handle division by zero: if std_dev is 0, result should be NaN
        
        # Merge numerator and denominator data
        merged = pd.merge(numerator.data, std_dev_factor.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('_num', '_den'))
        
        # Perform division, handling zero in denominator
        merged['factor'] = merged.apply(
            lambda row: np.nan if row['factor_den'] == 0 else row['factor_num'] / row['factor_den'], axis=1
        )
        
        result_data = merged[['timestamp', 'symbol', 'factor']].copy()
        return Factor(result_data, f"ts_zscore({self.name},{window})")

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

        result['factor'] = result.groupby('timestamp')['factor'].transform(apply_quantile)

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

        result['factor'] = result.groupby('timestamp')['factor'].transform(apply_scale)

        return Factor(result, f"scale({self.name},scale={scale},longscale={longscale},shortscale={shortscale})")

    def zscore(self) -> 'Factor':
        """Computes the cross-sectional Z-score of the factor."""
        # Z-score is equivalent to normalize(x, useStd=True)
        return self.normalize(useStd=True)

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

        # Merge factor data with group data
        merged = pd.merge(self.data, group_data.data, on=['timestamp', 'symbol'], 
                          suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data between factor and group data.")

        # Group by timestamp and the group label, then de-mean
        merged['factor'] = merged.groupby(['timestamp', 'factor_group'])['factor'].transform(
            lambda x: x - x.mean()
        )
        
        result_data = merged[['timestamp', 'symbol', 'factor']]
        return Factor(result_data, name=f"group_neutralize({self.name},{group_data.name})")

    def vector_neutralize(self, other: 'Factor') -> 'Factor':
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

        merged = pd.merge(self.data, other.data, on=['timestamp', 'symbol'], 
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

        neutralized_series = merged.groupby('timestamp').apply(neutralize_single_date).reset_index(level=0, drop=True)
        
        result_data = merged[['timestamp', 'symbol']].copy()
        result_data['factor'] = neutralized_series
        
        return Factor(result_data, name=f"vector_neutralize({self.name},{other.name})")

    def regression_neutralize(self, neut_factors: Union['Factor', List['Factor']]) -> 'Factor':
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

        # Start with the target factor's data
        merged = self.data.rename(columns={'factor': self.name or 'target'})
        
        # Sequentially merge all neutralization factors
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
            
            # Drop rows with NaNs in this slice
            valid_idx = Y.notna() & X.notna().all(axis=1)
            if valid_idx.sum() < 2: # Not enough data to run regression
                return pd.Series(np.nan, index=group.index)
                
            Y = Y[valid_idx]
            X = X[valid_idx]

            if X.empty:
                return pd.Series(np.nan, index=group.index)

            # Check for multicollinearity by inspecting the condition number of the design matrix
            X_const = sm.add_constant(X)
            if np.linalg.cond(X_const) > 1e10: # A common threshold for high multicollinearity
                return pd.Series(np.nan, index=group.index)

            model = sm.OLS(Y, X_const).fit()
            
            # Align residuals back to the original group index
            residuals = pd.Series(np.nan, index=group.index)
            residuals[valid_idx] = model.resid
            return residuals

        residuals_series = merged.groupby('timestamp').apply(get_residuals, include_groups=False).reset_index(level=0, drop=True)
        
        result_data = merged[['timestamp', 'symbol']].copy()
        result_data['factor'] = residuals_series

        neut_factors_str = ",".join([f.name for f in neut_factors])
        return Factor(result_data, name=f"regression_neutralize({self.name},[{neut_factors_str}])")

    def normalize(self, useStd: bool = False, limit: float = 0.0) -> 'Factor':
        """Cross-sectional normalization: (x - mean) / std (optional), then limit (optional)."""
        result = self.data.copy()

        def apply_normalize(group: pd.Series):
            mean_val = group.mean()
            normalized_group = group - mean_val

            if useStd:
                std_val = group.std() # Calculate std of original group for normalization
                if std_val == 0 or pd.isna(std_val):
                    normalized_group = pd.Series(np.nan, index=group.index)
                else:
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
            # Only backfill if the last value is NaN
            if pd.isna(s.iloc[-1]):
                non_nan = s.dropna()
                if len(non_nan) >= k:
                    return non_nan.iloc[-k]
            return s.iloc[-1]

        result = self._apply_rolling(backfill_func, window)
        return Factor(result, f"ts_backfill({self.name},{window},{k})")
    
    def ts_decay_exp_window(self, window: int, factor: float = 1.0, nan: bool = True) -> 'Factor':
        """Returns exponential decay of x with smoothing factor for the past d days."""
        if window <= 0:
            raise ValueError("Window must be positive")
        if not (0 < factor < 1):
            raise ValueError("Factor must be between 0 and 1 (exclusive)")
        
        def exp_decay_weights(length, alpha):
            weights = np.array([alpha**i for i in range(length)])
            return weights[::-1] / weights[::-1].sum() # newer values get more weight
            
        def decay_func(s):
            if s.isna().all():
                return np.nan if nan else 0.0
            
            # If nan=False, fill NaNs with 0 for calculation, then restore if needed.
            # This is complex because we need to handle original NaNs correctly for the output.
            # Let's use pandas ewm which handles this more gracefully.
            # Pandas ewm.mean calculates exp moving average. We need a specific sum as per formula.
            
            # The formula is (x[date] + x[date - 1] * f + …) / (1 + f + …) 
            # This is equivalent to an EWMA with alpha = 1 - factor.
            # Span = 2 / (1 - alpha) - 1 => alpha = 1 - 2 / (span + 1)
            # So, alpha in EWMA is 1 - factor here.
            
            # If nan=False, need to replace NaNs with 0 before calculating weights.
            # Pandas ewm.mean with adjust=True and ignore_na=True (default) should handle.
            
            # Re-implementing the exact formula:
            valid_s = s.dropna() if nan else s.fillna(0)
            
            if len(valid_s) == 0:
                 return np.nan if nan else 0.0
                 
            # Generate weights for the current window
            current_weights = np.array([factor**i for i in range(len(valid_s))])
            current_weights = current_weights[::-1] # Newest values get factor^0 = 1
            
            # Avoid division by zero if all weights are 0 (e.g., factor=0 and only old value)
            weight_sum = current_weights.sum()
            if weight_sum == 0:
                return np.nan if nan else 0.0
                
            # Apply weights and sum
            decayed_sum = (valid_s * current_weights).sum()
            
            return decayed_sum / weight_sum
            
        # The rolling window needs to call this function for each window.
        # _apply_rolling already handles the rolling part.
        result = self._apply_rolling(decay_func, window)
        
        return Factor(result, f"ts_decay_exp_window({self.name},{window},{factor},{nan})")
    
    def ts_decay_linear(self, window: int, dense: bool = False) -> 'Factor':
        """Returns the linear decay on x for the past d days."""
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
    
    # Math operations
    def add(self, other: Union['Factor', float], filter_nan: bool = False) -> 'Factor':
        """Addition with factor or scalar, with optional NaN filtering."""
        if isinstance(other, Factor):
            if filter_nan:
                merged = pd.merge(self.data, other.data, on=['timestamp', 'symbol'], 
                                  suffixes=('_x', '_y'), how='outer')
                merged['factor_x'] = merged['factor_x'].fillna(0)
                merged['factor_y'] = merged['factor_y'].fillna(0)
                merged['factor'] = merged['factor_x'] + merged['factor_y']
                result = merged[['timestamp', 'symbol', 'factor']]
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
                merged = pd.merge(self.data, other.data, on=['timestamp', 'symbol'], 
                                  suffixes=('_x', '_y'), how='outer')
                merged['factor_x'] = merged['factor_x'].fillna(0)
                merged['factor_y'] = merged['factor_y'].fillna(0)
                merged['factor'] = merged['factor_x'] - merged['factor_y']
                result = merged[['timestamp', 'symbol', 'factor']]
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
                merged = pd.merge(self.data, other.data, on=['timestamp', 'symbol'], 
                                  suffixes=('_x', '_y'), how='outer')
                merged['factor_x'] = merged['factor_x'].fillna(1)
                merged['factor_y'] = merged['factor_y'].fillna(1)
                merged['factor'] = merged['factor_x'] * merged['factor_y']
                result = merged[['timestamp', 'symbol', 'factor']]
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
    
    def scale(self, k: float = 1.0) -> 'Factor':
        """Scale to sum of absolute values equals k."""
        result = self.data.copy()
        abs_sum = np.abs(result['factor']).sum()
        if abs_sum != 0:
            result['factor'] *= k / abs_sum
        return Factor(result, f"scale({self.name},{k})")
    
    def log(self) -> 'Factor':
        """Natural logarithm of factor."""
        result = self.data.copy()
        result['factor'] = np.log(result['factor'])
        return Factor(result, f"log({self.name})")
    
    def sqrt(self) -> 'Factor':
        """Square root of factor values."""
        result = self.data.copy()
        result['factor'] = np.sqrt(result['factor'])
        return Factor(result, f"sqrt({self.name})")
    
    def s_log_1p(self) -> 'Factor':
        """Confine factor values to a shorter range using sign(x) * log(1 + abs(x))."""
        result = self.data.copy()
        # 使用 np.sign 處理符號，np.log1p(x) 等同於 log(1 + x)
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
            # 合併數據並計算 sign(x) * (abs(x) ** y)
            merged = pd.merge(self.data, exponent.data, on=['timestamp', 'symbol'], 
                             suffixes=('_x', '_y'), how='outer')
            
            # 對齊並確保數據類型正確
            x_values = merged['factor_x']
            y_values = merged['factor_y']
            
            # 計算簽名次方
            final_values = np.sign(x_values) * (np.abs(x_values) ** y_values)
            
            result = merged[['timestamp', 'symbol']].copy()
            result['factor'] = final_values
            
            return Factor(result, f"signed_power({self.name},{exponent.name})")
        else:
            # 對純量指數計算 signed_power
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
    
    # Python 運算符重載 - 讓因子表達式更直觀
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
    
    # Correlation
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
        
        # 合併兩個因子的數據
        try:
            merged = pd.merge(self.data, other.data, on=['timestamp', 'symbol'], 
                             suffixes=('_x', '_y'), how='inner')
        except Exception as e:
            raise ValueError(f"Failed to merge factor data: {e}")
        
        if merged.empty:
            raise ValueError("No common data between factors")
        
        merged = merged.sort_values(['symbol', 'timestamp'])
        
        # 分組計算滾動相關性
        result_data = []
        
        for symbol in merged['symbol'].unique():
            try:
                symbol_data = merged[merged['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('timestamp')
                
                # 確保數據類型正確
                x_values = pd.to_numeric(symbol_data['factor_x'], errors='coerce')
                y_values = pd.to_numeric(symbol_data['factor_y'], errors='coerce')
                
                # 計算滾動相關性
                corr_values = x_values.rolling(window, min_periods=window).corr(y_values)
                
                # 處理無效相關性值
                corr_values = corr_values.where(
                    (corr_values >= -1) & (corr_values <= 1), np.nan
                )
                
                # 添加到結果中
                symbol_data['factor'] = corr_values
                result_data.append(symbol_data[['timestamp', 'symbol', 'factor']])
                
            except Exception as e:
                # 如果出錯，添加 NaN 值
                symbol_data = merged[merged['symbol'] == symbol][['timestamp', 'symbol']].copy()
                symbol_data['factor'] = np.nan
                result_data.append(symbol_data)
        
        if result_data:
            merged = pd.concat(result_data, ignore_index=True)
        else:
            merged['factor'] = np.nan
        
        # 處理 NaN 值 - 保持 NaN 而不是填充為 0
        result = merged[['timestamp', 'symbol', 'factor']].copy()
        
        return Factor(result, f"ts_corr({self.name},{other.name},{window})")
    
    def ts_covariance(self, other: 'Factor', window: int) -> 'Factor':
        """Returns covariance of self and other for the past d days."""
        if window <= 0:
            raise ValueError("Window must be positive")
        
        if not isinstance(other, Factor):
            raise TypeError("Other must be a Factor object")
            
        # 合併兩個因子的數據
        merged = pd.merge(self.data, other.data, on=['timestamp', 'symbol'], 
                         suffixes=('_x', '_y'), how='inner')
        
        if merged.empty:
            raise ValueError("No common data between factors")
        
        merged = merged.sort_values(['symbol', 'timestamp'])
        
        # 分組計算滾動協方差
        result_data = []
        
        for symbol in merged['symbol'].unique():
            symbol_data = merged[merged['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('timestamp')
            
            x_values = pd.to_numeric(symbol_data['factor_x'], errors='coerce')
            y_values = pd.to_numeric(symbol_data['factor_y'], errors='coerce')
            
            # 計算滾動協方差
            cov_values = x_values.rolling(window, min_periods=window).cov(y_values)
            
            symbol_data['factor'] = cov_values
            result_data.append(symbol_data[['timestamp', 'symbol', 'factor']])
            
        if result_data:
            merged = pd.concat(result_data, ignore_index=True)
        else:
            merged = merged[['timestamp', 'symbol']].copy()
            merged['factor'] = np.nan
            
        result = merged[['timestamp', 'symbol', 'factor']].copy()
        return Factor(result, f"ts_covariance({self.name},{other.name},{window})")
    
    # Utility methods
    def _apply_rolling(self, func: Union[str, Callable], window: int) -> pd.DataFrame:
        """Apply rolling function by symbol."""
        result = self.data.copy().sort_values(['symbol', 'timestamp'])
        if isinstance(func, str):
            if func == 'product':
                result['factor'] = (result.groupby('symbol')['factor']
                                   .rolling(window, min_periods=window)
                                   .apply(np.prod, raw=True).values)
            else:
                result['factor'] = (result.groupby('symbol')['factor']
                                   .rolling(window, min_periods=window)
                                   .agg(func).values)
        else:
            result['factor'] = (result.groupby('symbol')['factor']
                               .rolling(window, min_periods=window)
                               .apply(func, raw=False).values)
        return result.sort_values(['timestamp', 'symbol'])
    
    def _apply_groupby(self, func: str, *args) -> pd.DataFrame:
        """Apply groupby function by symbol."""
        result = self.data.copy().sort_values(['symbol', 'timestamp'])
        result['factor'] = (result.groupby('symbol')['factor']
                           .transform(func, *args))
        return result.sort_values(['timestamp', 'symbol'])
    
    def _binary_op(self, other: 'Factor', op: Callable) -> pd.DataFrame:
        """Apply binary operation with another factor."""
        merged = pd.merge(self.data, other.data, on=['timestamp', 'symbol'], 
                         suffixes=('_x', '_y'))
        merged['factor'] = op(merged['factor_x'], merged['factor_y'])
        return merged[['timestamp', 'symbol', 'factor']]
    
    # Data access
    def to_csv(self, path: str) -> str:
        """Save to CSV file."""
        self.data.to_csv(path, index=False)
        return path
    
    def to_multiindex(self) -> pd.Series:
        """Convert to MultiIndex Series."""
        return self.data.set_index(['timestamp', 'symbol'])['factor']
    
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
        n_obs = self.data.shape[0]
        n_symbols = len(self.data['symbol'].unique())
        valid_ratio = self.data['factor'].notna().mean()
        time_range = f"{self.data['timestamp'].min().strftime('%Y-%m-%d')} to {self.data['timestamp'].max().strftime('%Y-%m-%d')}"
        return (f"Factor(name={self.name}, obs={n_obs}, symbols={n_symbols}, "
               f"valid={valid_ratio:.1%}, period={time_range})")
    
    def __str__(self):
        """User-friendly string representation."""
        return f"Factor({self.name}): {self.data.shape[0]} obs, {len(self.data['symbol'].unique())} symbols"


# Factory functions
def load_factor(data: Union[str, pd.DataFrame], column: str, name: Optional[str] = None) -> Factor:
    """
    Load factor from data source.
    
    Parameters
    ----------
    data : str or DataFrame
        Data source (CSV path or DataFrame)
    column : str
        Column name to extract as factor
    name : str, optional
        Factor name
        
    Returns
    -------
    Factor
        Factor object with specified column
    """
    if isinstance(data, str):
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


# End of core Factor implementation
