"""Multi-column market data container with unified MultiIndex structure."""

import pandas as pd
from typing import Union, Optional, List
from .core import Factor


class Panel:
    """Multi-column market data container with (timestamp, symbol) MultiIndex."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize Panel from DataFrame with MultiIndex (timestamp, symbol)."""
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
        """Load Panel from CSV file."""
        df = pd.read_csv(path, parse_dates=['timestamp'])
        if 'timestamp' in df.columns and 'symbol' in df.columns:
            df = df.set_index(['timestamp', 'symbol']).sort_index()
        return cls(df)
    
    def get_factor(self, column: str, name: Optional[str] = None) -> Factor:
        """Extract column as Factor object."""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")
        
        factor_data = self.data[[column]].copy()
        factor_data.columns = ['factor']
        return Factor(factor_data, name or column)
    
    def add_factor(self, factor: Factor, name: str) -> 'Panel':
        """Add Factor as new column."""
        result = self.data.copy()
        result[name] = factor.data['factor']
        return Panel(result)
    
    def add_column(self, data: pd.Series, name: str) -> 'Panel':
        """Add Series with MultiIndex (timestamp, symbol) as new column."""
        result = self.data.copy()
        result[name] = data
        return Panel(result)
    
    def __getitem__(self, key):
        """Access columns: returns Factor for str, Panel for list."""
        if isinstance(key, str):
            return self.get_factor(key)
        elif isinstance(key, list):
            return Panel(self.data[key].copy())
        else:
            raise TypeError("Key must be str or list of str")
    
    def slice_time(self, start: Optional[str] = None, end: Optional[str] = None) -> 'Panel':
        """Slice panel by time range [start, end]."""
        idx = self.data.index.get_level_values('timestamp')
        
        mask_start = idx >= pd.to_datetime(start) if start else pd.Series(True, index=self.data.index)
        mask_end = idx <= pd.to_datetime(end) if end else pd.Series(True, index=self.data.index)
        
        return Panel(self.data[mask_start & mask_end].copy())
    
    def slice_symbols(self, symbols: Union[str, List[str]]) -> 'Panel':
        """Slice panel by symbol(s)."""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        return Panel(self.data.loc[(slice(None), symbols), :].copy())
    
    def to_csv(self, path: str) -> str:
        """Save Panel to CSV file."""
        self.data.reset_index().to_csv(path, index=False)
        return path
    
    def info(self) -> None:
        """Print data quality: shape, symbols, periods, time range, NaN per column."""
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
            print(f"    {col}: {n_nan} ({n_nan/len(self.data):.1%})")
    
    def __repr__(self):
        timestamps = self.data.index.get_level_values('timestamp')
        symbols = self.data.index.get_level_values('symbol')
        n_symbols = len(symbols.unique())
        n_periods = len(timestamps.unique())
        valid_ratio = self.data.notna().values.mean()
        time_range = f"{timestamps.min().strftime('%Y-%m-%d')} to {timestamps.max().strftime('%Y-%m-%d')}"
        return (f"Panel(obs={self.data.shape[0]}, cols={len(self.data.columns)}, "
                f"symbols={n_symbols}, periods={n_periods}, valid={valid_ratio:.1%}, range={time_range})")
    
    def __str__(self):
        n_symbols = len(self.data.index.get_level_values('symbol').unique())
        return f"Panel: {self.data.shape[0]} obs, {len(self.data.columns)} columns, {n_symbols} symbols"

