"""Group operators for Factor."""

import numpy as np
import pandas as pd
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import FactorBase


class GroupMixin:
    """Mixin providing group-based operators.
    
    Group operators allow calculations within subsets of symbols
    defined by a grouping factor (e.g., sector, industry).
    """
    
    def group(self: 'FactorBase', mapping: dict, name: Optional[str] = None) -> 'FactorBase':
        """Map symbol to group ID using a custom dict.
        
        Parameters
        ----------
        mapping : dict
            Symbol to group ID mapping, e.g. {'ETH': 1, 'SOL': 1, 'ARB': 2}
        name : str, optional
            Name for the resulting group factor
        
        Returns
        -------
        Factor
            Group ID factor with same index as self
        
        Examples
        --------
        sector_map = {'ETH': 1, 'SOL': 1, 'ARB': 2, 'OP': 2}
        group_factor = close.group(sector_map)
        """
        from . import Factor
        
        if not isinstance(mapping, dict):
            raise TypeError("Mapping must be a dictionary of symbol -> group_id.")
            
        result = self.data[['timestamp', 'symbol']].copy()
        result['factor'] = result['symbol'].map(mapping)
        
        return Factor(result, name or "group(custom)")

    def group_neutralize(self: 'FactorBase', group: 'FactorBase') -> 'FactorBase':
        """Neutralize factor against groups (demean within group).
        
        Parameters
        ----------
        group : Factor
            Factor defining the group for each asset
        
        Returns
        -------
        Factor
            Neutralized factor (x - group_mean)
        """
        from . import Factor
        
        self._validate_factor(group, "group_neutralize")
        
        merged = pd.merge(self.data, group.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data for group neutralization.")
            
        group_means = merged.groupby(['timestamp', 'factor_group'])['factor'].transform('mean')
        
        merged['factor'] = merged['factor'] - group_means
        
        return Factor(merged[['timestamp', 'symbol', 'factor']], f"group_neutralize({self.name},{group.name})")

    def group_mean(self: 'FactorBase', group: 'FactorBase') -> 'FactorBase':
        """Calculate group mean for each asset.
        
        Parameters
        ----------
        group : Factor
            Factor defining the group for each asset
        
        Returns
        -------
        Factor
            Mean value of the group the asset belongs to
        """
        from . import Factor
        
        self._validate_factor(group, "group_mean")
        
        merged = pd.merge(self.data, group.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data for group mean stats.")
            
        group_means = merged.groupby(['timestamp', 'factor_group'])['factor'].transform('mean')
        
        merged['factor'] = group_means
        
        return Factor(merged[['timestamp', 'symbol', 'factor']], f"group_mean({self.name},{group.name})")

    def group_median(self: 'FactorBase', group: 'FactorBase') -> 'FactorBase':
        """Calculate group median for each asset.
        
        Parameters
        ----------
        group : Factor
            Factor defining the group for each asset
        
        Returns
        -------
        Factor
            Median value of the group the asset belongs to
        """
        from . import Factor
        
        self._validate_factor(group, "group_median")
        
        merged = pd.merge(self.data, group.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data for group median stats.")
            
        group_medians = merged.groupby(['timestamp', 'factor_group'])['factor'].transform('median')
        
        merged['factor'] = group_medians
        
        return Factor(merged[['timestamp', 'symbol', 'factor']], f"group_median({self.name},{group.name})")

    def group_rank(self: 'FactorBase', group: 'FactorBase') -> 'FactorBase':
        """Calculate percentile rank within each group.
        
        Parameters
        ----------
        group : Factor
            Factor defining the group for each asset
        
        Returns
        -------
        Factor
            Percentile rank (0-1) within the group
        """
        from . import Factor
        
        self._validate_factor(group, "group_rank")
        
        merged = pd.merge(self.data, group.data, 
                          on=['timestamp', 'symbol'], 
                          suffixes=('', '_group'))
        
        if merged.empty:
            raise ValueError("No common data for group rank.")
            
        merged['factor'] = merged.groupby(['timestamp', 'factor_group'])['factor'].rank(pct=True)
        
        return Factor(merged[['timestamp', 'symbol', 'factor']], f"group_rank({self.name},{group.name})")

    def group_scale(self: 'FactorBase', group: 'FactorBase') -> 'FactorBase':
        """Scale values within each group to 0-1 range.
        
        Parameters
        ----------
        group : Factor
            Factor defining the group for each asset
        
        Returns
        -------
        Factor
            Scaled values: (x - min) / (max - min)
        """
        from . import Factor
        
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

    def group_zscore(self: 'FactorBase', group: 'FactorBase') -> 'FactorBase':
        """Calculate Z-score within each group.
        
        Parameters
        ----------
        group : Factor
            Factor defining the group for each asset
        
        Returns
        -------
        Factor
            Z-score: (x - mean) / std
        """
        from . import Factor
        
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

    def group_normalize(self: 'FactorBase', group: 'FactorBase', scale: float = 1.0) -> 'FactorBase':
        """Normalize such that each group's absolute sum equals scale.
        
        Parameters
        ----------
        group : Factor
            Factor defining the group for each asset
        scale : float, default 1.0
            Target sum of absolute values for each group
        
        Returns
        -------
        Factor
            Normalized values: x / sum(|x|_group) * scale
        """
        from . import Factor
        
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
