"""Data acquisition and management for cryptocurrency markets via CCXT."""

import pandas as pd
import ccxt
import time
import os
import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .panel import Panel

logger = logging.getLogger(__name__)

_FREQ_MAP = {
    '1m': 'min', '5m': '5min', '15m': '15min', '30m': '30min',
    '1h': 'h', '4h': '4h', '1d': 'D', '1w': 'W', '1M': 'MS',
}
_OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']
_SYMBOL_MAP = {'MATIC': ['MATIC', 'POL'], 'POL': ['MATIC', 'POL']}


def fetch_data(
    symbols: List[str], 
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    exchange: str = 'binance',
    output_path: Optional[str] = None
) -> 'Panel':
    """
    Fetch and prepare cryptocurrency OHLCV data.
    
    Parameters
    ----------
    symbols : list of str
        E.g., ['BTC', 'ETH']
    timeframe : str, default '1d'
        Timeframe for data
    start_date : str, optional
        YYYY-MM-DD format
    exchange : str, default 'binance'
        Exchange name
    output_path : str, optional
        CSV path to save data
    """
    try:
        exchange_obj = getattr(ccxt, exchange)()
    except AttributeError:
        raise ValueError(f"Exchange '{exchange}' not supported")
    
    if not exchange_obj.has['fetchOHLCV']:
        raise ValueError(f"Exchange '{exchange}' does not support OHLCV")
    
    since = exchange_obj.parse8601(f'{start_date}T00:00:00Z') if start_date else None
    
    all_data = []
    
    for symbol in symbols:
        if symbol in _SYMBOL_MAP:
            data = _fetch_renamed_symbol(exchange_obj, _SYMBOL_MAP[symbol], timeframe, since)
            if data is not None:
                data['symbol'] = 'MATIC'
                all_data.append(data)
        else:
            data = _fetch_single_symbol(exchange_obj, symbol, timeframe, since)
            if data is not None:
                all_data.append(data)
    
    if not all_data:
        raise ValueError("No data fetched")
    
    combined = pd.concat(all_data, ignore_index=True)
    result = _process_data(combined, timeframe)
    
    from .panel import Panel
    panel = Panel(result)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        panel.to_csv(output_path)
        logger.info(f"Data saved to {output_path}")
    
    return panel


def _fetch_single_symbol(exchange, symbol: str, timeframe: str, since) -> Optional[pd.DataFrame]:
    """Fetch data for single symbol with pagination."""
    try:
        market_symbol = f'{symbol}/USDT'
        
        exchange.load_markets()
        if market_symbol not in exchange.symbols:
            logger.warning(f"{market_symbol} not available on exchange")
            return None
        
        all_ohlcv = []
        limit = 1000
        
        while True:
            ohlcv = exchange.fetch_ohlcv(market_symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        
        if not all_ohlcv:
            return None
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        
        logger.info(f"Fetched {len(df)} records for {symbol}")
        return df
        
    except Exception as e:
        logger.warning(f"Failed to fetch {symbol}: {e}")
        return None


def _fetch_renamed_symbol(exchange, symbols: List[str], timeframe: str, since) -> Optional[pd.DataFrame]:
    """Fetch and merge data for renamed symbols."""
    dfs = []
    for symbol in symbols:
        df = _fetch_single_symbol(exchange, symbol, timeframe, since)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    return combined


def _process_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Process and align data to common time range."""
    pivoted = df.pivot_table(index='timestamp', columns='symbol', values='close')
    common_start = pivoted.apply(lambda s: s.first_valid_index()).max()
    end_date = df['timestamp'].max()
    
    freq = _FREQ_MAP.get(timeframe, 'D')
    full_range = pd.date_range(start=common_start, end=end_date, freq=freq)
    
    aligned_data = {}
    for col in _OHLCV_COLS:
        pivot = df.pivot_table(index='timestamp', columns='symbol', values=col)
        pivot = pivot[pivot.index >= common_start].reindex(full_range).ffill().bfill()
        aligned_data[col] = pivot
    
    result_dfs = []
    for col, data in aligned_data.items():
        stacked = data.stack(future_stack=True).reset_index()
        stacked.columns = ['timestamp', 'symbol', col]
        result_dfs.append(stacked.set_index(['timestamp', 'symbol']))
    
    return pd.concat(result_dfs, axis=1).sort_index()

