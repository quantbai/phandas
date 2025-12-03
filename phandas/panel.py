"""Multi-column market data container with flat (timestamp, symbol) structure."""

import pandas as pd
import logging
from typing import Union, Optional, List
from .core import Factor

logger = logging.getLogger(__name__)


class Panel:
    """Multi-column market data container.
    
    Stores OHLCV and derived data in a simple flat DataFrame with 
    columns ['timestamp', 'symbol', ...]. Easy to construct from any data source.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns ['timestamp', 'symbol', ...]
    
    Attributes
    ----------
    data : pd.DataFrame
        Sorted data with columns ['timestamp', 'symbol', ...]
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'timestamp': ['2024-01-01', '2024-01-01'],
    ...     'symbol': ['BTC', 'ETH'],
    ...     'close': [40000, 2000]
    ... })
    >>> panel = Panel(df)
    >>> close = panel['close']  # Returns Factor
    """
    
    def __init__(self, data: pd.DataFrame):
        """Initialize Panel from DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns ['timestamp', 'symbol', ...] or
            MultiIndex (timestamp, symbol) - will be flattened automatically
        
        Raises
        ------
        ValueError
            If data lacks timestamp and symbol columns/index
        """
        df = data.copy()
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        
        if 'timestamp' not in df.columns or 'symbol' not in df.columns:
            raise ValueError("Data must have 'timestamp' and 'symbol' columns")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
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
            Panel instance
        """
        df = pd.read_csv(path, parse_dates=['timestamp'])
        return cls(df)
    
    def get_factor(self, column: str, name: Optional[str] = None) -> Factor:
        """Extract column as Factor.
        
        Parameters
        ----------
        column : str
            Column name to extract
        name : str, optional
            Factor name (defaults to column name)
        
        Returns
        -------
        Factor
            Factor object with extracted column
        
        Raises
        ------
        ValueError
            If column not found in Panel
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")
        
        factor_data = self.data[['timestamp', 'symbol', column]].copy()
        factor_data.columns = ['timestamp', 'symbol', 'factor']
        return Factor(factor_data, name or column)
    
    def add_factor(self, factor: Factor, name: str) -> 'Panel':
        """Add Factor as new column.
        
        Parameters
        ----------
        factor : Factor
            Factor to add
        name : str
            Column name
        
        Returns
        -------
        Panel
            New Panel with added column
        """
        result = self.data.copy()
        factor_df = factor.data[['timestamp', 'symbol', 'factor']].rename(
            columns={'factor': name}
        )
        result = pd.merge(result, factor_df, on=['timestamp', 'symbol'], how='left')
        return Panel(result)
    
    def add_column(self, data: pd.Series, name: str) -> 'Panel':
        """Add Series as column.
        
        Parameters
        ----------
        data : pd.Series
            Series aligned with Panel rows
        name : str
            Column name
        
        Returns
        -------
        Panel
            New Panel with added column
        """
        result = self.data.copy()
        result[name] = data.values if hasattr(data, 'values') else data
        return Panel(result)
    
    def __add__(self, other: 'Panel') -> 'Panel':
        """Merge two Panels on (timestamp, symbol).
        
        Parameters
        ----------
        other : Panel
            Panel to merge
        
        Returns
        -------
        Panel
            Merged Panel with non-overlapping columns
        
        Raises
        ------
        TypeError
            If other is not Panel
        ValueError
            If no common rows or duplicate columns
        """
        if not isinstance(other, Panel):
            raise TypeError(f"unsupported operand type(s) for +: 'Panel' and '{type(other).__name__}'")
        
        self_cols = set(self.data.columns) - {'timestamp', 'symbol'}
        other_cols = set(other.data.columns) - {'timestamp', 'symbol'}
        overlap_cols = self_cols & other_cols
        
        if overlap_cols:
            raise ValueError(f"Duplicate columns: {list(overlap_cols)}")
        
        result = pd.merge(
            self.data, 
            other.data, 
            on=['timestamp', 'symbol'], 
            how='inner'
        )
        
        if result.empty:
            raise ValueError("No common (timestamp, symbol) pairs")
        
        return Panel(result)
    
    def __radd__(self, other):
        """Support sum() with initial value 0.
        
        Parameters
        ----------
        other
            Initial value (must be 0)
        
        Returns
        -------
        Panel
            Self if other is 0
        """
        if other == 0:
            return self
        return self.__add__(other)
    
    def __getitem__(self, key):
        """Access columns or column subset.
        
        Parameters
        ----------
        key : str or list
            Column name(s) to access
        
        Returns
        -------
        Factor or Panel
            Single column as Factor, multiple columns as Panel
        
        Raises
        ------
        TypeError
            If key is not str or list
        """
        if isinstance(key, str):
            return self.get_factor(key)
        elif isinstance(key, list):
            cols = ['timestamp', 'symbol'] + [c for c in key if c not in ['timestamp', 'symbol']]
            return Panel(self.data[cols].copy())
        else:
            raise TypeError("Key must be str or list")
    
    def slice_time(self, start: Optional[str] = None, end: Optional[str] = None) -> 'Panel':
        """Slice Panel by time range.
        
        Parameters
        ----------
        start : str, optional
            Start date in YYYY-MM-DD format
        end : str, optional
            End date in YYYY-MM-DD format
        
        Returns
        -------
        Panel
            Panel sliced to time range
        """
        mask = pd.Series(True, index=self.data.index)
        
        if start:
            mask &= self.data['timestamp'] >= pd.to_datetime(start)
        if end:
            mask &= self.data['timestamp'] <= pd.to_datetime(end)
        
        return Panel(self.data[mask].copy())
    
    def slice_symbols(self, symbols: Union[str, List[str]]) -> 'Panel':
        """Slice Panel by symbol(s).
        
        Parameters
        ----------
        symbols : str or list of str
            Symbol(s) to keep
        
        Returns
        -------
        Panel
            Panel filtered to specified symbols
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        mask = self.data['symbol'].isin(symbols)
        return Panel(self.data[mask].copy())
    
    def to_csv(self, path: str) -> str:
        """Save Panel to CSV file.
        
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
        """List of data columns (excluding timestamp, symbol)."""
        return [c for c in self.data.columns if c not in ['timestamp', 'symbol']]
    
    @property
    def symbols(self) -> List[str]:
        """List of unique symbols."""
        return self.data['symbol'].unique().tolist()
    
    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Unique timestamps."""
        return pd.DatetimeIndex(self.data['timestamp'].unique())
    
    def info(self) -> None:
        """Print data quality summary and NaN statistics."""
        n_symbols = self.data['symbol'].nunique()
        n_periods = self.data['timestamp'].nunique()
        time_min = self.data['timestamp'].min()
        time_max = self.data['timestamp'].max()
        
        logger.info(f"Panel: {len(self.data)} obs, {len(self.columns)} columns")
        logger.info(f"  symbols={n_symbols}, periods={n_periods}")
        logger.info(f"  time: {time_min.strftime('%Y-%m-%d')} to {time_max.strftime('%Y-%m-%d')}")
        logger.info("  NaN per column:")
        for col in self.columns:
            n_nan = self.data[col].isna().sum()
            logger.info(f"    {col}: {n_nan} ({n_nan/len(self.data):.1%})")
    
    def __repr__(self):
        """String representation with shape, coverage, and time range."""
        n_symbols = self.data['symbol'].nunique()
        n_periods = self.data['timestamp'].nunique()
        valid_ratio = self.data[self.columns].notna().values.mean() if self.columns else 1.0
        time_min = self.data['timestamp'].min().strftime('%Y-%m-%d')
        time_max = self.data['timestamp'].max().strftime('%Y-%m-%d')
        return (f"Panel(obs={len(self.data)}, cols={len(self.columns)}, "
                f"symbols={n_symbols}, periods={n_periods}, valid={valid_ratio:.1%}, range={time_min} to {time_max})")
    
    def __str__(self):
        """Human-readable string summary."""
        n_symbols = self.data['symbol'].nunique()
        return f"Panel: {len(self.data)} obs, {len(self.columns)} columns, {n_symbols} symbols"

    def __len__(self):
        """Number of rows."""
        return len(self.data)
