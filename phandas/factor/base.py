"""Factor base class with core data structure and utilities."""

import pandas as pd
import numpy as np
from typing import Union, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Factor


class FactorBase:
    """Base class for Factor with data container and helper methods.
    
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
        """Validate window parameter is positive."""
        if window <= 0:
            raise ValueError("Window must be positive")
    
    def _validate_factor(self, other: 'FactorBase', op_name: str) -> None:
        """Validate other is a Factor instance."""
        if not isinstance(other, FactorBase):
            raise TypeError(f"{op_name}: other must be a Factor object.")
    
    @staticmethod
    def _replace_inf(series: pd.Series) -> pd.Series:
        """Replace inf values with NaN."""
        return series.replace([np.inf, -np.inf], np.nan)
    
    def _apply_cs_operation(self, operation: Callable, name_suffix: str, 
                           require_no_nan: bool = False) -> 'Factor':
        """Apply cross-sectional operation grouped by timestamp.
        
        Parameters
        ----------
        operation : Callable
            Function to apply to each timestamp group
        name_suffix : str
            Suffix to add to the factor name
        require_no_nan : bool
            If True, raise error if all values are NaN
        
        Returns
        -------
        Factor
            New factor with operation applied
        """
        from . import Factor
        
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

    def _binary_op(self, other: Union['FactorBase', float], op_func: Callable, 
                   op_name: str, scalar_suffix: Optional[str] = None) -> 'Factor':
        """Apply binary operation between two factors or factor and scalar.
        
        Parameters
        ----------
        other : Factor or float
            Other operand
        op_func : Callable
            Binary operation function
        op_name : str
            Operation name for naming result
        scalar_suffix : str, optional
            Suffix for scalar operations
        
        Returns
        -------
        Factor
            Result of binary operation
        """
        from . import Factor
        
        if isinstance(other, FactorBase):
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

    def _comparison_op(self, other: Union['FactorBase', float], comp_func: Callable, 
                       op_name: str) -> 'Factor':
        """Apply comparison operation between factors or factor and scalar.
        
        Parameters
        ----------
        other : Factor or float
            Other operand
        comp_func : Callable
            Comparison function
        op_name : str
            Operation name for naming result
        
        Returns
        -------
        Factor
            Boolean result as 0/1 factor
        """
        from . import Factor
        
        if isinstance(other, FactorBase):
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
        """Apply rolling window operation grouped by symbol.
        
        Parameters
        ----------
        func : str or Callable
            Aggregation function name or callable
        window : int
            Window size
        
        Returns
        -------
        pd.DataFrame
            DataFrame with rolling result in 'factor' column
        """
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
