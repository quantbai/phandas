"""Binance OHLCV data provider.

Fetches spot market OHLCV data via CCXT library.
Handles symbol renames (e.g., MATIC -> POL) with seamless historical data.
"""

import time
import warnings
import pandas as pd
import ccxt
from typing import List, Optional, Callable

from .base import SYMBOL_RENAMES

FETCH_BATCH_SIZE = 1000


def fetch_binance(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch Binance spot OHLCV data.
    
    Parameters
    ----------
    symbols : List[str]
        Cryptocurrency symbols (e.g., ['BTC', 'ETH'])
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    
    Returns
    -------
    pd.DataFrame or None
        Columns: timestamp, symbol, open, high, low, close, volume
    
    Notes
    -----
    Automatically handles MATIC -> POL rename with cutoff at 2024-09-01.
    Uses USDT perpetual pairs for all symbols.
    """
    try:
        exchange = ccxt.binance()
        if not exchange.has['fetchOHLCV']:
            raise RuntimeError("Binance OHLCV not supported")
        
        since = exchange.parse8601(f'{start_date}T00:00:00Z') if start_date else None
        until = exchange.parse8601(f'{end_date}T00:00:00Z') if end_date else None
        
        symbols_to_fetch = list(set(symbols))
        
        for new_sym, rename_info in SYMBOL_RENAMES.items():
            if new_sym not in symbols_to_fetch:
                continue
            
            result = _fetch_with_rename(
                exchange, symbols_to_fetch, new_sym, rename_info, since, until
            )
            if result is not None:
                return result
        
        return _fetch_ohlcv(exchange, symbols_to_fetch, since, until)
        
    except Exception as e:
        raise RuntimeError(f"Binance fetch failed: {e}")


def _fetch_with_rename(
    exchange,
    symbols: List[str],
    new_symbol: str,
    rename_info: dict,
    since: Optional[int],
    until: Optional[int],
) -> Optional[pd.DataFrame]:
    """Fetch data handling symbol rename transition."""
    old_symbol = rename_info['old_symbol']
    cutoff_date = rename_info['cutoff_date']
    cutoff_ts = exchange.parse8601(f'{cutoff_date}T00:00:00Z')
    
    if since is not None and since >= cutoff_ts:
        return None
    
    old_until = cutoff_ts - 1
    other_symbols = [s for s in symbols if s != new_symbol]
    
    old_data = _fetch_ohlcv(
        exchange,
        [old_symbol] + other_symbols,
        since,
        old_until,
    )
    
    new_data = _fetch_ohlcv(
        exchange,
        symbols,
        cutoff_ts,
        until,
    )
    
    if old_data is None and new_data is None:
        return None
    
    if old_data is not None:
        old_data.loc[old_data['symbol'] == old_symbol, 'symbol'] = new_symbol
    
    if old_data is not None and new_data is not None:
        result = pd.concat([old_data, new_data], ignore_index=True)
        result = _fill_rename_gap(result, new_symbol)
    elif old_data is not None:
        result = old_data
    else:
        result = new_data
    
    return result.sort_values('timestamp').reset_index(drop=True)


def _fill_rename_gap(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Forward-fill gaps around symbol rename transition."""
    renamed_rows = df[df['symbol'] == symbol].copy()
    
    if len(renamed_rows) == 0:
        return df
    
    renamed_rows = renamed_rows.set_index('timestamp').sort_index()
    full_range = pd.date_range(renamed_rows.index.min(), renamed_rows.index.max(), freq='D')
    renamed_rows = renamed_rows.reindex(full_range).ffill()
    renamed_rows = renamed_rows.reset_index().rename(columns={'index': 'timestamp'})
    renamed_rows['volume'] = renamed_rows['volume'].fillna(0)
    
    result = pd.concat([
        df[df['symbol'] != symbol],
        renamed_rows
    ], ignore_index=True)
    
    return result.sort_values('timestamp').reset_index(drop=True)


def _fetch_ohlcv(
    exchange,
    symbols: List[str],
    since: Optional[int],
    until: Optional[int],
    timeframe: str = '1d',
    post_process: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for multiple symbols."""
    dfs = []
    
    for symbol in symbols:
        df = _fetch_symbol_ohlcv(exchange, symbol, timeframe, since, until)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        return None
    
    result = pd.concat(dfs, ignore_index=True)
    
    if post_process:
        result = post_process(result)
    
    return result


def _fetch_symbol_ohlcv(
    exchange,
    symbol: str,
    timeframe: str,
    since: Optional[int],
    until: Optional[int],
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for single symbol with pagination."""
    try:
        market = f'{symbol}/USDT'
        exchange.load_markets()
        
        if market not in exchange.symbols:
            warnings.warn(f"Market {market} not available on Binance")
            return None
        
        all_candles = []
        cursor = since
        
        while True:
            batch = exchange.fetch_ohlcv(
                market, timeframe, since=cursor, limit=FETCH_BATCH_SIZE
            )
            
            if not batch:
                break
            
            original_len = len(batch)
            
            if until:
                batch = [c for c in batch if c[0] <= until]
            
            all_candles.extend(batch)
            
            if original_len < FETCH_BATCH_SIZE:
                break
            
            if batch:
                cursor = batch[-1][0] + 1
            
            time.sleep(exchange.rateLimit / 1000)
        
        if not all_candles:
            return None
        
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        
        return df
        
    except Exception as e:
        warnings.warn(f"Failed to fetch {symbol}: {e}")
        return None

