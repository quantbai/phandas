"""Cross-sectional operators for Factor."""

import numpy as np
import pandas as pd
from typing import Optional, TYPE_CHECKING
from scipy.stats import norm, uniform, cauchy

if TYPE_CHECKING:
    from .base import FactorBase


class CrossSectionalMixin:
    """Mixin providing cross-sectional operators.
    
    Cross-sectional operators compute statistics across all symbols
    at each timestamp (e.g., rank, mean, zscore).
    """
    
    def rank(self: 'FactorBase') -> 'FactorBase':
        """Rank values from 0 to 1 across all assets at each time.
        
        Highest value gets 1.0, lowest gets close to 0.
        
        Returns
        -------
        Factor
            Percentile rank (0-1) at each timestamp
        """
        def rank_op(group):
            if group.nunique() == 1:
                return pd.Series(np.nan, index=group.index)
            return group.rank(method='min', pct=True)
        
        return self._apply_cs_operation(rank_op, 'rank', require_no_nan=True)

    def mean(self: 'FactorBase') -> 'FactorBase':
        """Average across all assets at each time.
        
        Returns
        -------
        Factor
            Average value at each timestamp
        """
        return self._apply_cs_operation(lambda g: g.mean(), 'mean', require_no_nan=True)

    def median(self: 'FactorBase') -> 'FactorBase':
        """Middle value across all assets at each time.
        
        Returns
        -------
        Factor
            Median value at each timestamp
        """
        return self._apply_cs_operation(lambda g: g.median(), 'median', require_no_nan=True)

    def quantile(self: 'FactorBase', driver: str = "gaussian", sigma: float = 1.0) -> 'FactorBase':
        """Cross-sectional quantile transform (normal/uniform/Cauchy).
        
        Transforms factor values using quantile mapping to target distribution.
        
        Parameters
        ----------
        driver : {'gaussian', 'uniform', 'cauchy'}, default 'gaussian'
            Target distribution for quantile transform
        sigma : float, default 1.0
            Scale parameter for the distribution
        
        Returns
        -------
        Factor
            Quantile-transformed factor
        """
        from . import Factor
        
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

    def scale(self: 'FactorBase', scale: float = 1.0, longscale: Optional[float] = None, 
              shortscale: Optional[float] = None) -> 'FactorBase':
        """Scale to sum(|factor|)=scale with optional separate long/short sizing.
        
        Parameters
        ----------
        scale : float, default 1.0
            Target sum of absolute values
        longscale : float, optional
            Long-only scale. If None, uses `scale`.
        shortscale : float, optional
            Short-only scale. If None, uses `scale`.
        
        Returns
        -------
        Factor
            Scaled factor
        """
        from . import Factor
        
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

    def normalize(self: 'FactorBase', use_std: bool = False, limit: float = 0.0) -> 'FactorBase':
        """Cross-sectional demean with optional std normalization.
        
        Parameters
        ----------
        use_std : bool, default False
            Whether to divide by standard deviation (z-score)
        limit : float, default 0.0
            Clipping limit. If > 0, clips to [-limit, limit]
        
        Returns
        -------
        Factor
            Demeaned and optionally normalized factor
        """
        from . import Factor
        
        result = self.data.copy()

        def apply_normalize(group: pd.Series):
            if group.isna().any():
                return pd.Series(np.nan, index=group.index)
            
            normalized_group = group - group.mean()

            if use_std:
                std_val = group.std()
                if std_val == 0 or pd.isna(std_val):
                    return pd.Series(0.0, index=group.index)
                normalized_group /= std_val
            
            if limit != 0.0:
                normalized_group = normalized_group.clip(lower=-limit, upper=limit)
            
            return normalized_group

        result['factor'] = result.groupby('timestamp')['factor'].transform(apply_normalize)
        return Factor(result, f"normalize({self.name},use_std={use_std},limit={limit})")

    def zscore(self: 'FactorBase') -> 'FactorBase':
        """Standardize to mean=0 and std=1 at each time.
        
        Formula: (x - mean) / std
        
        Returns
        -------
        Factor
            Standardized factor
        """
        return self.normalize(use_std=True)

    def spread(self: 'FactorBase', pct: float = 0.5) -> 'FactorBase':
        """Binary long-short signal based on top and bottom percentiles.
        
        Parameters
        ----------
        pct : float, default 0.5
            Percentile threshold. Top pct% get +0.5, bottom pct% get -0.5.
        
        Returns
        -------
        Factor
            Binary long-short signal per timestamp
        """
        from . import Factor
        
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

    def signal(self: 'FactorBase') -> 'FactorBase':
        """Convert to trading weights (long +0.5, short -0.5, net zero).
        
        Subtracts mean then scales so positive values sum to 0.5
        and negative values sum to -0.5.
        
        Returns
        -------
        Factor
            Trading weights that sum to zero
        """
        from . import Factor
        
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

    def _is_signal(self: 'FactorBase', date=None) -> bool:
        """Check if factor is a valid dollar-neutral signal.
        
        Parameters
        ----------
        date : str or Timestamp, optional
            Date to check. If None, uses latest date.
        
        Returns
        -------
        bool
            True if long sum is ~0.5, short sum is ~-0.5, total ~0.0
        """
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
