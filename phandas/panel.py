"""Multi-column market data container with flat (timestamp, symbol) structure."""

import pandas as pd
from typing import Union, Optional, List
from .core import Factor


class Panel:
    """Multi-column market data container.
    
    Stores OHLCV and derived data in a flat DataFrame with
    columns ['timestamp', 'symbol', ...].
    """
    
    def __init__(self, data: pd.DataFrame):
        df = data.copy()
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        
        if 'timestamp' not in df.columns or 'symbol' not in df.columns:
            raise ValueError("Data must have 'timestamp' and 'symbol' columns")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        self.data = df
    
    @classmethod
    def from_csv(cls, path: str) -> 'Panel':
        df = pd.read_csv(path, parse_dates=['timestamp'])
        return cls(df)
    
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> 'Panel':
        return cls(df)
    
    def to_df(self) -> pd.DataFrame:
        return self.data.copy()
    
    def __getitem__(self, key) -> Union[Factor, 'Panel']:
        if isinstance(key, str):
            if key not in self.data.columns:
                raise ValueError(f"Column '{key}' not found")
            factor_data = self.data[['timestamp', 'symbol', key]].copy()
            factor_data.columns = ['timestamp', 'symbol', 'factor']
            return Factor(factor_data, key)
        elif isinstance(key, list):
            cols = ['timestamp', 'symbol'] + [c for c in key if c not in ['timestamp', 'symbol']]
            return Panel(self.data[cols].copy())
        else:
            raise TypeError("Key must be str or list")
    
    def slice_time(self, start: Optional[str] = None, end: Optional[str] = None) -> 'Panel':
        mask = pd.Series(True, index=self.data.index)
        if start:
            mask &= self.data['timestamp'] >= pd.to_datetime(start)
        if end:
            mask &= self.data['timestamp'] <= pd.to_datetime(end)
        return Panel(self.data[mask].copy())
    
    def slice_symbols(self, symbols: Union[str, List[str]]) -> 'Panel':
        if isinstance(symbols, str):
            symbols = [symbols]
        mask = self.data['symbol'].isin(symbols)
        return Panel(self.data[mask].copy())
    
    def to_csv(self, path: str) -> str:
        self.data.to_csv(path, index=False)
        return path
    
    @property
    def columns(self) -> List[str]:
        return [c for c in self.data.columns if c not in ['timestamp', 'symbol']]
    
    @property
    def symbols(self) -> List[str]:
        return self.data['symbol'].unique().tolist()
    
    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.data['timestamp'].unique())
    
    def info(self) -> None:
        from .console import print
        n_symbols = len(self.symbols)
        n_periods = len(self.timestamps)
        time_range = f"{self.timestamps.min().strftime('%Y-%m-%d')} to {self.timestamps.max().strftime('%Y-%m-%d')}"
        
        print(f"Panel: {len(self)} rows, {len(self.columns)} columns")
        print(f"  symbols={n_symbols}, periods={n_periods}, range={time_range}")
        
        if self.columns:
            nan_counts = {col: self.data[col].isna().sum() for col in self.columns}
            print(f"  NaN: {nan_counts}")
    
    def __repr__(self):
        n_symbols = len(self.symbols)
        n_periods = len(self.timestamps)
        time_range = f"{self.timestamps.min().strftime('%Y-%m-%d')} to {self.timestamps.max().strftime('%Y-%m-%d')}"
        return f"Panel({len(self)} rows, {len(self.columns)} cols, {n_symbols} symbols, {n_periods} periods, {time_range})"
    
    def __len__(self):
        return len(self.data)
