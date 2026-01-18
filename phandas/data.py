"""Data acquisition and panel construction for cryptocurrency markets.

This module provides the unified data fetching interface for phandas.
It orchestrates multiple data providers, merges results, and constructs
time-aligned Panel objects ready for factor research.

Public API
----------
fetch_data : Fetch and merge multi-source cryptocurrency data into a Panel.

Example
-------
>>> from phandas import fetch_data
>>> panel = fetch_data(
...     symbols=['ETH', 'SOL', 'ARB', 'OP'],
...     start_date='2024-01-01',
...     sources=['binance', 'okx', 'defillama']
... )
"""

import os
import warnings
import pandas as pd
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .panel import Panel

from .providers import SOURCE_REGISTRY
from .providers.base import TIMEFRAME_MAP


def fetch_data(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> 'Panel':
    """Fetch and merge multi-source cryptocurrency data.
    
    Parameters
    ----------
    symbols : List[str]
        Cryptocurrency symbols (e.g., ['BTC', 'ETH', 'SOL'])
    timeframe : str, default '1d'
        Data timeframe. Recommended '1d' for alternative data compatibility.
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    sources : List[str], optional
        Data sources to fetch. Default: ['binance']
        Available: binance, okx, defillama
    output_path : str, optional
        Path to save resulting Panel as CSV
    
    Returns
    -------
    Panel
        Merged and time-aligned market data container
    
    Notes
    -----
    - Alternative data sources (okx, defillama) are aligned to Binance's end date
    - Missing values are preserved as NaN; use ts_backfill() or ts_ffill() if needed
    - Data is sorted by (symbol, timestamp) with full date range preserved
    
    Examples
    --------
    Basic OHLCV data:
    
    >>> panel = fetch_data(['ETH', 'SOL'], start_date='2024-01-01')
    >>> close = panel['close']
    
    With alternative data:
    
    >>> panel = fetch_data(
    ...     ['ETH', 'SOL'],
    ...     start_date='2024-01-01',
    ...     sources=['binance', 'okx', 'defillama']
    ... )
    """
    if sources is None:
        sources = ['binance']
    
    raw_dfs = []
    binance_end_date = None
    
    for source in sources:
        if source not in SOURCE_REGISTRY:
            warnings.warn(f"Unknown source: {source}. Available: {list(SOURCE_REGISTRY.keys())}")
            continue
        
        fetcher = SOURCE_REGISTRY[source]
        
        try:
            if source == 'binance':
                df = fetcher(symbols, start_date, end_date)
                if df is not None and 'timestamp' in df.columns:
                    binance_end_date = df['timestamp'].max().strftime('%Y-%m-%d')
            else:
                source_end = binance_end_date or end_date
                df = fetcher(symbols, start_date, source_end)
            
            if df is not None:
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()
                raw_dfs.append(df)
            else:
                warnings.warn(f"No data returned from {source}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to fetch from {source}: {e}")
    
    if not raw_dfs:
        raise ValueError("No data fetched from any source")
    
    combined = _merge_dataframes(raw_dfs)
    aligned = _align_and_fill(combined, timeframe, symbols)
    
    from .panel import Panel
    result = Panel(aligned)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path)
    
    return result


def _merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple DataFrames on (timestamp, symbol).
    
    Parameters
    ----------
    dfs : List[pd.DataFrame]
        DataFrames to merge, each with timestamp and symbol columns
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all columns
    """
    result = dfs[0]
    
    for df in dfs[1:]:
        result = pd.merge(result, df, on=['timestamp', 'symbol'], how='outer')
    
    if result.columns.duplicated().any():
        result = result.loc[:, ~result.columns.duplicated(keep='first')]
    
    if 'index' in result.columns:
        result = result.drop(columns=['index'])
    
    return result


def _align_and_fill(
    df: pd.DataFrame,
    timeframe: str,
    symbols: List[str],
) -> pd.DataFrame:
    """Align data to full date range without forward-filling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame with all data
    timeframe : str
        Data timeframe for reindexing
    symbols : List[str]
        Target symbols to include
    
    Returns
    -------
    pd.DataFrame
        Aligned DataFrame with NaN preserved for missing values
    
    Notes
    -----
    Missing values are preserved as NaN. Use ts_backfill() or ts_ffill()
    operators if forward/backward filling is needed before backtesting.
    """
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    
    freq = TIMEFRAME_MAP.get(timeframe, 'D')
    full_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    result_dfs = []
    value_columns = [c for c in df.columns if c not in ['timestamp', 'symbol']]
    
    for col in value_columns:
        pivot = df.pivot_table(index='timestamp', columns='symbol', values=col)
        pivot = pivot.reindex(full_range)
        pivot.index.name = 'timestamp'
        stacked = pivot.reset_index().melt(id_vars='timestamp', var_name='symbol', value_name=col)
        result_dfs.append(stacked)
    
    result = result_dfs[0]
    for df_part in result_dfs[1:]:
        result = pd.merge(result, df_part, on=['timestamp', 'symbol'], how='outer')
    
    result = result[result['symbol'].isin(symbols)]
    
    int_cols = ['year', 'month', 'day']
    for col in int_cols:
        if col in result.columns:
            result[col] = result[col].astype('Int64')
    
    return result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
