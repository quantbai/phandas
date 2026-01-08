"""Neutralization operators for Factor."""

import numpy as np
import pandas as pd
from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import FactorBase


class NeutralizationMixin:
    """Mixin providing neutralization operators.
    
    Neutralization removes exposure to specific factors or directions
    via vector projection or OLS regression.
    """
    
    def vector_neut(self: 'FactorBase', other: 'FactorBase') -> 'FactorBase':
        """Remove other's influence from self, keeping only the independent part.
        
        Formula: x - (x dot y / y dot y) * y
        
        Parameters
        ----------
        other : Factor
            Factor whose influence to remove
        
        Returns
        -------
        Factor
            Self with other's influence removed
        """
        from . import Factor
        
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

    def regression_neut(
        self: 'FactorBase',
        neut_factors: Union['FactorBase', List['FactorBase']],
        ridge_alpha: float = 1e-8
    ) -> 'FactorBase':
        """Remove other factors' influence using linear regression.
        
        Performs cross-sectional regression y = alpha + beta @ X + epsilon
        and returns the residuals (epsilon).
        
        Parameters
        ----------
        neut_factors : Factor or List[Factor]
            Risk factor(s) whose influence to remove.
            Tip: apply rank() or zscore() before passing if factors have
            very different scales.
        ridge_alpha : float, default 1e-8
            Small L2 regularization for numerical stability.
            Set to 0 for pure OLS (may fail on collinear factors).
        
        Returns
        -------
        Factor
            Residuals after removing linear effects of neut_factors.
            Assets with NaN in y or any x will have NaN in output.
        """
        from . import Factor
        from .base import FactorBase
        
        if isinstance(neut_factors, FactorBase):
            neut_factors = [neut_factors]
        if not all(isinstance(f, FactorBase) for f in neut_factors):
            raise TypeError("neut_factors must be a Factor or list of Factors.")

        merged = self.data.rename(columns={'factor': self.name or 'target'})
        neut_names = []
        
        for i, f in enumerate(neut_factors):
            neut_name = f.name or f'neut_{i}'
            neut_names.append(neut_name)
            merged = pd.merge(
                merged,
                f.data.rename(columns={'factor': neut_name}),
                on=['timestamp', 'symbol'],
                how='inner'
            )

        if merged.empty:
            raise ValueError("No common data for regression neutralization.")
        
        target_name = self.name or 'target'
        n_features = len(neut_names)
        
        def get_residuals(group):
            Y = group[target_name].values.astype(np.float64)
            X = group[neut_names].values.astype(np.float64)
            
            valid_mask = ~(np.isnan(Y) | np.isnan(X).any(axis=1))
            n_valid = valid_mask.sum()
            
            result = np.full(len(Y), np.nan)
            
            if n_valid < n_features + 2:
                return pd.Series(result, index=group.index, name='factor')
            
            Y_valid = Y[valid_mask]
            X_valid = X[valid_mask]
            
            X_const = np.column_stack([np.ones(n_valid), X_valid])
            n_params = X_const.shape[1]
            
            XtX = X_const.T @ X_const
            if ridge_alpha > 0:
                reg = ridge_alpha * np.eye(n_params)
                reg[0, 0] = 0
                XtX = XtX + reg
            
            XtY = X_const.T @ Y_valid
            
            try:
                params = np.linalg.solve(XtX, XtY)
                result[valid_mask] = Y_valid - X_const @ params
            except np.linalg.LinAlgError:
                pass
            
            return pd.Series(result, index=group.index, name='factor')

        result = merged.groupby('timestamp', group_keys=False).apply(
            get_residuals, include_groups=False
        ).reset_index(level=0, drop=True)
        
        merged['factor'] = result
        result_data = merged[['timestamp', 'symbol', 'factor']]

        neut_factors_str = ",".join([f.name for f in neut_factors])
        return Factor(result_data, name=f"regression_neut({self.name},[{neut_factors_str}])")
