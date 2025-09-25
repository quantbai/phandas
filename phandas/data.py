"""
Data acquisition and management for cryptocurrency markets.

Simplified API for fetching and preparing OHLCV data.
"""

import pandas as pd
import ccxt
import time
import os
import logging
from typing import List, Optional


logger = logging.getLogger(__name__)


def fetch_data(
    symbols: List[str], 
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    exchange: str = 'binance',
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch and prepare cryptocurrency OHLCV data.
    
    Parameters
    ----------
    symbols : list of str
        Cryptocurrency symbols (e.g., ['BTC', 'ETH'])
    timeframe : str, default '1d'
        Timeframe for data
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    exchange : str, default 'binance'
        Exchange name
    output_path : str, optional
        CSV file path to save data
        
    Returns
    -------
    DataFrame
        MultiIndex DataFrame with (timestamp, symbol) index
    """
    try:
        exchange_obj = getattr(ccxt, exchange)()
    except AttributeError:
        raise ValueError(f"Exchange '{exchange}' not supported")
    
    if not exchange_obj.has['fetchOHLCV']:
        raise ValueError(f"Exchange '{exchange}' does not support OHLCV")
    
    since = exchange_obj.parse8601(f'{start_date}T00:00:00Z') if start_date else None
    
    # Handle special cases
    symbol_map = {'MATIC': ['MATIC', 'POL'], 'POL': ['MATIC', 'POL']}
    
    all_data = []
    
    for symbol in symbols:
        if symbol in symbol_map:
            # Handle renamed tokens
            data = _fetch_renamed_symbol(exchange_obj, symbol_map[symbol], timeframe, since)
            if data is not None:
                data['symbol'] = 'MATIC'  # Unified name
                all_data.append(data)
        else:
            data = _fetch_single_symbol(exchange_obj, symbol, timeframe, since)
            if data is not None:
                all_data.append(data)
    
    if not all_data:
        raise ValueError("No data fetched")
    
    # Combine and process
    combined = pd.concat(all_data, ignore_index=True)
    result = _process_data(combined, timeframe)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path)
        logger.info(f"Data saved to {output_path}")
    
    return result


def _fetch_single_symbol(exchange, symbol: str, timeframe: str, since) -> Optional[pd.DataFrame]:
    """Fetch data for single symbol with pagination."""
    try:
        market_symbol = f'{symbol}/USDT'
        
        # Load markets to check symbol availability
        exchange.load_markets()
        if market_symbol not in exchange.symbols:
            logger.warning(f"{market_symbol} not available on exchange")
            return None
        
        all_ohlcv = []
        limit = 1000  # Fetch limit per request
        
        # Use pagination to fetch all historical data
        while True:
            ohlcv = exchange.fetch_ohlcv(market_symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            # Update since to continue from last timestamp + 1ms
            since = ohlcv[-1][0] + 1
            # Respect rate limits
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
    
    # Merge chronologically
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    return combined


def _process_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Process and align data."""
    # Pivot to find common start date
    pivoted = df.pivot_table(index='timestamp', columns='symbol', values='close')
    common_start = pivoted.apply(lambda s: s.first_valid_index()).max()
    
    # Create full date range
    end_date = df['timestamp'].max()
    
    freq_map = {
        '1m': 'min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': 'h',
        '4h': '4h',
        '1d': 'D',
        '1w': 'W',
        '1M': 'MS',
    }
    
    freq = freq_map.get(timeframe, 'D') # Default to daily if not found
    full_range = pd.date_range(start=common_start, end=end_date, freq=freq)
    
    # Align and fill for each column
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    aligned_data = {}
    
    for col in ohlcv_cols:
        pivot = df.pivot_table(index='timestamp', columns='symbol', values=col)
        pivot = pivot[pivot.index >= common_start]
        pivot = pivot.reindex(full_range).ffill().bfill()
        aligned_data[col] = pivot
    
    # Convert back to MultiIndex format
    result_dfs = []
    for col, data in aligned_data.items():
        stacked = data.stack(future_stack=True).reset_index()
        stacked.columns = ['timestamp', 'symbol', col]
        result_dfs.append(stacked.set_index(['timestamp', 'symbol']))
    
    result = pd.concat(result_dfs, axis=1)
    return result.sort_index()


def check_data_quality(data: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Check data quality and return summary.
    
    Parameters
    ----------
    data : DataFrame
        OHLCV data to check
    verbose : bool, default True
        Print summary to console
        
    Returns
    -------
    dict
        Quality report
    """
    if isinstance(data, str):
        data = pd.read_csv(data, index_col=[0, 1], parse_dates=[0])
    
    report = {
        'shape': data.shape,
        'symbols': list(data.index.get_level_values('symbol').unique()),
        'time_range': (data.index.get_level_values('timestamp').min(),
                      data.index.get_level_values('timestamp').max()),
        'missing_values': data.isnull().sum().to_dict(),
        'duplicates': data.index.duplicated().sum()
    }
    
    if verbose:
        print(f"Data shape: {report['shape']}")
        print(f"Symbols: {len(report['symbols'])}")
        print(f"Time range: {report['time_range'][0]} to {report['time_range'][1]}")
        if any(report['missing_values'].values()):
            print("Missing values:", {k: v for k, v in report['missing_values'].items() if v > 0})
        if report['duplicates'] > 0:
            print(f"Duplicate timestamps: {report['duplicates']}")
    
    return report
