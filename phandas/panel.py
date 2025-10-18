"""
Multi-column market data container (OHLCV, Factors)
with unified MultiIndex structure.
"""

import pandas as pd
from typing import Union, Optional, List
from .core import Factor


class Panel:
    """
    Multi-column market data container.
    
    Internal format: MultiIndex DataFrame with (timestamp, symbol) index.
    Supports data extraction to Factor objects.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Panel from DataFrame.
        
        Parameters
        ----------
        data : DataFrame
            MultiIndex DataFrame with (timestamp, symbol) index
        """
        df = data.copy()
        if not isinstance(df.index, pd.MultiIndex):
            if 'timestamp' in df.columns and 'symbol' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index(['timestamp', 'symbol']).sort_index()
            else:
                raise ValueError("Data must have timestamp and symbol columns or MultiIndex")
        
        self.data = df
    
    @classmethod
    def from_csv(cls, path: str) -> 'Panel':
        """
        Load Panel from CSV file.
        
        Parameters
        ----------
        path : str
            CSV file path
            
        """
        df = pd.read_csv(path, parse_dates=['timestamp'])
        if 'timestamp' in df.columns and 'symbol' in df.columns:
            df = df.set_index(['timestamp', 'symbol']).sort_index()
        return cls(df)
    
    def get_factor(self, column: str, name: Optional[str] = None) -> Factor:
        """
        Extract column as Factor object.
        
        Parameters
        ----------
        column : str
            Column name to extract
        name : str, optional
            Factor name (defaults to column name)
            
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")
        
        factor_data = self.data[[column]].copy()
        factor_data.columns = ['factor']
        return Factor(factor_data, name or column)
    
    def add_factor(self, factor: Factor, name: str) -> 'Panel':
        """
        Add Factor as new column.
        
        Parameters
        ----------
        factor : Factor
            Factor object to add
        name : str
            Column name for the factor
            
        """
        result = self.data.copy()
        
        factor_series = factor.data['factor']
        result[name] = factor_series
        
        return Panel(result)
    
    def add_column(self, data: pd.Series, name: str) -> 'Panel':
        """
        Add arbitrary Series as new column.
        
        Parameters
        ----------
        data : Series
            Series with MultiIndex (timestamp, symbol)
        name : str
            Column name
            
        """
        result = self.data.copy()
        result[name] = data
        return Panel(result)
    
    
    def __getitem__(self, key):
        """
        Access columns using bracket notation.
        
        Returns Factor for single column, Panel for multiple columns.
        """
        if isinstance(key, str):
            return self.get_factor(key)
        elif isinstance(key, list):
            subset = self.data[key].copy()
            return Panel(subset)
        else:
            raise TypeError("Key must be str or list of str")
    
    def slice_time(self, start: Optional[str] = None, end: Optional[str] = None) -> 'Panel':
        """
        Slice panel by time range.
        
        Parameters
        ----------
        start : str, optional
            Start date (inclusive)
        end : str, optional
            End date (inclusive)
            
        """
        idx = self.data.index.get_level_values('timestamp')
        
        if start is not None:
            start = pd.to_datetime(start)
            mask_start = idx >= start
        else:
            mask_start = pd.Series(True, index=self.data.index)
        
        if end is not None:
            end = pd.to_datetime(end)
            mask_end = idx <= end
        else:
            mask_end = pd.Series(True, index=self.data.index)
        
        result = self.data[mask_start & mask_end].copy()
        return Panel(result)
    
    def slice_symbols(self, symbols: Union[str, List[str]]) -> 'Panel':
        """
        Slice panel by symbols.
        
        Parameters
        ----------
        symbols : str or list of str
            Symbol(s) to extract
            
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        result = self.data.loc[(slice(None), symbols), :].copy()
        return Panel(result)
    
    def to_csv(self, path: str) -> str:
        """
        Save to CSV file.
        
        Parameters
        ----------
        path : str
            Output file path
            
        """
        self.data.reset_index().to_csv(path, index=False)
        return path
    
    def info(self) -> None:
        """
        Print panel data quality: metadata and NaN statistics per column.
        """
        timestamps = self.data.index.get_level_values('timestamp')
        symbols = self.data.index.get_level_values('symbol')
        n_symbols = len(symbols.unique())
        n_periods = len(timestamps.unique())
        
        print(f"Panel: {self.data.shape[0]} obs, {len(self.data.columns)} columns")
        print(f"  symbols={n_symbols}, periods={n_periods}")
        print(f"  time: {timestamps.min().strftime('%Y-%m-%d')} to {timestamps.max().strftime('%Y-%m-%d')}")
        print("  NaN per column:")
        for col in self.data.columns:
            n_nan = self.data[col].isna().sum()
            nan_ratio = n_nan / len(self.data)
            print(f"    {col}: {n_nan} ({nan_ratio:.1%})")
    
    def __repr__(self):
        timestamps = self.data.index.get_level_values('timestamp')
        symbols = self.data.index.get_level_values('symbol')
        n_symbols = len(symbols.unique())
        n_periods = len(timestamps.unique())
        n_cols = len(self.data.columns)
        valid_ratio = self.data.notna().values.mean()
        
        time_range = f"{timestamps.min().strftime('%Y-%m-%d')} to {timestamps.max().strftime('%Y-%m-%d')}"
        
        return (f"Panel(obs={self.data.shape[0]}, cols={n_cols}, "
                f"symbols={n_symbols}, periods={n_periods}, "
                f"valid={valid_ratio:.1%}, range={time_range})")
    
    def __str__(self):
        n_symbols = len(self.data.index.get_level_values('symbol').unique())
        n_cols = len(self.data.columns)
        return f"Panel: {self.data.shape[0]} obs, {n_cols} columns, {n_symbols} symbols"

