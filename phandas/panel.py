"""Multi-column market data container with flat (timestamp, symbol) structure.

Panel is the primary data container in phandas, storing multi-asset time series
data (OHLCV, alternative data, derived metrics) in a normalized flat structure.

Examples
--------
>>> from phandas import fetch_data
>>> panel = fetch_data(['ETH', 'SOL'], start_date='2024-01-01')
>>> close = panel['close']  # Extract Factor
>>> panel.info()
"""

import pandas as pd
from typing import Union, Optional, List
from .core import Factor


class Panel:
    """Multi-column market data container for quantitative research.
    
    Panel stores OHLCV and alternative data in a flat DataFrame with
    columns ['timestamp', 'symbol', ...]. It serves as the bridge between
    raw market data and Factor objects used for alpha research.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least 'timestamp' and 'symbol' columns.
        Additional columns become accessible via indexing.
    
    Attributes
    ----------
    data : pd.DataFrame
        Underlying flat DataFrame
    columns : List[str]
        Available data columns (excluding timestamp and symbol)
    symbols : List[str]
        Unique asset symbols in the panel
    timestamps : pd.DatetimeIndex
        Unique timestamps in the panel
    
    Examples
    --------
    >>> panel = fetch_data(['ETH', 'SOL'], start_date='2024-01-01')
    >>> close = panel['close']          # Returns Factor
    >>> sub = panel[['close', 'volume']] # Returns Panel subset
    >>> panel.slice_time('2024-06-01', '2024-12-31')
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
        """Load Panel from CSV file.
        
        Parameters
        ----------
        path : str
            Path to CSV file with timestamp and symbol columns
        
        Returns
        -------
        Panel
            New Panel instance
        """
        df = pd.read_csv(path, parse_dates=['timestamp'])
        return cls(df)
    
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> 'Panel':
        """Create Panel from existing DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with timestamp and symbol columns
        
        Returns
        -------
        Panel
            New Panel instance
        """
        return cls(df)
    
    def to_df(self) -> pd.DataFrame:
        """Export Panel data as DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Copy of underlying data
        """
        return self.data.copy()
    
    def __getitem__(self, key) -> Union[Factor, 'Panel']:
        """Extract column as Factor or subset as Panel.
        
        Parameters
        ----------
        key : str or List[str]
            Single column name returns Factor, list returns Panel subset
        
        Returns
        -------
        Factor or Panel
            Factor if key is str, Panel if key is list
        
        Raises
        ------
        ValueError
            If column not found
        TypeError
            If key is neither str nor list
        
        Examples
        --------
        >>> close = panel['close']           # Factor
        >>> subset = panel[['close', 'volume']]  # Panel
        """
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
        """Filter Panel to specific time range.
        
        Parameters
        ----------
        start : str, optional
            Start date (inclusive) in YYYY-MM-DD format
        end : str, optional
            End date (inclusive) in YYYY-MM-DD format
        
        Returns
        -------
        Panel
            Filtered Panel containing only data within time range
        """
        mask = pd.Series(True, index=self.data.index)
        if start:
            mask &= self.data['timestamp'] >= pd.to_datetime(start)
        if end:
            mask &= self.data['timestamp'] <= pd.to_datetime(end)
        return Panel(self.data[mask].copy())
    
    def slice_symbols(self, symbols: Union[str, List[str]]) -> 'Panel':
        """Filter Panel to specific symbols.
        
        Parameters
        ----------
        symbols : str or List[str]
            Symbol or list of symbols to include
        
        Returns
        -------
        Panel
            Filtered Panel containing only specified symbols
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        mask = self.data['symbol'].isin(symbols)
        return Panel(self.data[mask].copy())
    
    def to_csv(self, path: str) -> str:
        """Export Panel data to CSV file.
        
        Parameters
        ----------
        path : str
            Output file path
        
        Returns
        -------
        str
            Path to saved file
        """
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
        """Print summary information about the Panel.
        
        Displays row count, column count, symbol count, time range,
        and NaN counts per column.
        """
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
