"""Multi-column market data container with (timestamp, symbol) MultiIndex."""

import pandas as pd
import logging
from typing import Union, Optional, List
from .core import Factor

logger = logging.getLogger(__name__)


class Panel:
    """Multi-column data with (timestamp, symbol) MultiIndex."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize Panel from DataFrame with (timestamp, symbol) MultiIndex."""
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
        """Load from CSV."""
        df = pd.read_csv(path, parse_dates=['timestamp'])
        if 'timestamp' in df.columns and 'symbol' in df.columns:
            df = df.set_index(['timestamp', 'symbol']).sort_index()
        return cls(df)
    
    def get_factor(self, column: str, name: Optional[str] = None) -> Factor:
        """Extract column as Factor."""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")
        factor_data = self.data[[column]].copy()
        factor_data.columns = ['factor']
        return Factor(factor_data, name or column)
    
    def add_factor(self, factor: Factor, name: str) -> 'Panel':
        """Add Factor as column."""
        result = self.data.copy()
        result[name] = factor.data['factor']
        return Panel(result)
    
    def add_column(self, data: pd.Series, name: str) -> 'Panel':
        """Add Series with MultiIndex as column."""
        result = self.data.copy()
        result[name] = data
        return Panel(result)
    
    def __add__(self, other: 'Panel') -> 'Panel':
        """Merge on (timestamp, symbol) index."""
        if not isinstance(other, Panel):
            raise TypeError(f"unsupported operand type(s) for +: 'Panel' and '{type(other).__name__}'")
        
        common_idx = self.data.index.intersection(other.data.index)
        if len(common_idx) == 0:
            raise ValueError("No common (timestamp, symbol) pairs")
        
        left = self.data.loc[common_idx]
        right = other.data.loc[common_idx]
        
        overlap_cols = left.columns.intersection(right.columns)
        if len(overlap_cols) > 0:
            raise ValueError(f"Duplicate columns: {list(overlap_cols)}")
        
        result = pd.concat([left, right], axis=1)
        return Panel(result)
    
    def __radd__(self, other):
        """Support sum() with initial value 0."""
        if other == 0:
            return self
        return self.__add__(other)
    
    def __getitem__(self, key):
        """Access columns: str -> Factor, list -> Panel."""
        if isinstance(key, str):
            return self.get_factor(key)
        elif isinstance(key, list):
            return Panel(self.data[key].copy())
        else:
            raise TypeError("Key must be str or list")
    
    def slice_time(self, start: Optional[str] = None, end: Optional[str] = None) -> 'Panel':
        """Slice by time range."""
        idx = self.data.index.get_level_values('timestamp')
        mask_start = idx >= pd.to_datetime(start) if start else pd.Series(True, index=self.data.index)
        mask_end = idx <= pd.to_datetime(end) if end else pd.Series(True, index=self.data.index)
        return Panel(self.data[mask_start & mask_end].copy())
    
    def slice_symbols(self, symbols: Union[str, List[str]]) -> 'Panel':
        """Slice by symbol(s)."""
        if isinstance(symbols, str):
            symbols = [symbols]
        return Panel(self.data.loc[(slice(None), symbols), :].copy())
    
    def to_csv(self, path: str) -> str:
        """Save to CSV."""
        self.data.reset_index().to_csv(path, index=False)
        return path
    
    def info(self) -> None:
        """Print data quality summary."""
        timestamps = self.data.index.get_level_values('timestamp')
        symbols = self.data.index.get_level_values('symbol')
        n_symbols = len(symbols.unique())
        n_periods = len(timestamps.unique())
        
        logger.info(f"Panel: {self.data.shape[0]} obs, {len(self.data.columns)} columns")
        logger.info(f"  symbols={n_symbols}, periods={n_periods}")
        logger.info(f"  time: {timestamps.min().strftime('%Y-%m-%d')} to {timestamps.max().strftime('%Y-%m-%d')}")
        logger.info("  NaN per column:")
        for col in self.data.columns:
            n_nan = self.data[col].isna().sum()
            logger.info(f"    {col}: {n_nan} ({n_nan/len(self.data):.1%})")
    
    def __repr__(self):
        """String representation with data shape and coverage info."""
        timestamps = self.data.index.get_level_values('timestamp')
        symbols = self.data.index.get_level_values('symbol')
        n_symbols = len(symbols.unique())
        n_periods = len(timestamps.unique())
        valid_ratio = self.data.notna().values.mean()
        time_range = f"{timestamps.min().strftime('%Y-%m-%d')} to {timestamps.max().strftime('%Y-%m-%d')}"
        return (f"Panel(obs={self.data.shape[0]}, cols={len(self.data.columns)}, "
                f"symbols={n_symbols}, periods={n_periods}, valid={valid_ratio:.1%}, range={time_range})")
    
    def __str__(self):
        """Human-readable string summary."""
        n_symbols = len(self.data.index.get_level_values('symbol').unique())
        return f"Panel: {self.data.shape[0]} obs, {len(self.data.columns)} columns, {n_symbols} symbols"

