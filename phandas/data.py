"""Data acquisition and management for cryptocurrency markets via CCXT."""

import pandas as pd
import ccxt
import time
import os
import logging
from typing import List, Optional, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .panel import Panel

logger = logging.getLogger(__name__)

TIMEFRAME_MAP = {
    '1m': 'min', '5m': '5min', '15m': '15min', '30m': '30min',
    '1h': 'h', '4h': '4h', '1d': 'D', '1w': 'W', '1M': 'MS',
}
FETCH_BATCH_SIZE = 1000


def fetch_data(
    symbols: List[str], 
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> 'Panel':
    """Fetch, merge, and align multi-source data.
    
    Defaults to Daily resolution ('1d') and Binance OHLCV data.
    """
    if sources is None:
        sources = ['binance']
        
    return fetch_panel_core(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        sources=sources,
        output_path=output_path
    )


def fetch_panel_core(
    symbols: List[str], 
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> 'Panel':
    """Core function to fetch, merge, and align multi-source data (Any resolution).
    
    This function is the internal engine that supports arbitrary timeframes.
    It orchestrates fetching from individual sources and aligning them into a Panel.
    """
    if sources is None:
        sources = ['binance']
    
    source_map = {
        'binance': fetch_binance,
        'benchmark': fetch_benchmark,
        'calendar': fetch_calendar,
        'vwap': fetch_vwap,
    }
    
    raw_dfs = []
    binance_end_date = None
    
    for source in sources:
        if source not in source_map:
            logger.warning(f"Unknown source: {source}. Available: {list(source_map.keys())}")
            continue
        
        logger.info(f"Fetching from {source}...")
        
        try:
            if source == 'binance':
                df = source_map[source](symbols, timeframe, start_date, end_date)
                if df is not None and 'timestamp' in df.columns:
                    binance_end_date = df['timestamp'].max().strftime('%Y-%m-%d')
                    logger.info(f"  Binance data ends at: {binance_end_date}")
            else:
                source_end_date = binance_end_date or end_date
                df = source_map[source](symbols, timeframe, start_date, source_end_date)
            
            if df is not None:
                if not isinstance(df.index, pd.MultiIndex) and 'timestamp' in df.columns and 'symbol' in df.columns:
                    df = df.set_index(['timestamp', 'symbol'])
                raw_dfs.append(df)
            else:
                logger.warning(f"  No data returned from {source}")
        
        except Exception as e:
            logger.error(f"  Failed to fetch from {source}: {e}")
    
    if not raw_dfs:
        raise ValueError("No data fetched from any source")
    
    combined = pd.concat(raw_dfs, axis=1)
    
    if combined.columns.duplicated().any():
        combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]
    
    logger.info(f"Combined columns: {list(combined.columns)}")
    
    combined_reset = combined.reset_index()
    if 'index' in combined_reset.columns:
        logger.warning("Unexpected 'index' column found, removing it")
        combined_reset = combined_reset.drop(columns=['index'])
    logger.info(f"After reset_index columns: {list(combined_reset.columns)}")
    
    processed = _process_data(combined_reset, timeframe, symbols)
    logger.info(f"After _process_data columns: {list(processed.columns)}")
    
    int_cols = ['year', 'month', 'day']
    for col in int_cols:
        if col in processed.columns:
            processed[col] = processed[col].astype('Int64')
    
    from .panel import Panel
    result = Panel(processed)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path)
        logger.info(f"Data saved to {output_path}")
    
    return result


def _fetch_ohlcv_data(
    exchange, 
    symbols: List[str], 
    timeframe: str, 
    since, 
    until=None,
    columns_post_process: Optional[Callable] = None
) -> Optional[pd.DataFrame]:
    """Generic OHLCV fetcher with optional column post-processing.
    
    Parameters
    ----------
    exchange : ccxt.Exchange
        Exchange instance with fetchOHLCV support
    symbols : List[str]
        Symbols to fetch
    timeframe : str
        Timeframe (e.g., '1d', '1h')
    since : int
        Milliseconds timestamp
    until : int, optional
        Upper bound timestamp
    columns_post_process : Callable, optional
        Function to select/rename columns from raw OHLCV
    """
    SYMBOL_RENAMES = {'MATIC': ['MATIC', 'POL'], 'POL': ['MATIC', 'POL']}
    
    def _fetch_single(sym: str) -> Optional[pd.DataFrame]:
        try:
            market_sym = f'{sym}/USDT'
            exchange.load_markets()
            if market_sym not in exchange.symbols:
                logger.warning(f"{market_sym} not available")
                return None
            
            all_candles = []
            cursor = since
            
            while True:
                batch = exchange.fetch_ohlcv(market_sym, timeframe, since=cursor, limit=FETCH_BATCH_SIZE)
                if not batch:
                    break
                
                original_batch_len = len(batch)
                if until:
                    batch = [c for c in batch if c[0] <= until]
                    all_candles.extend(batch)
                    if original_batch_len < FETCH_BATCH_SIZE:
                        break
                else:
                    all_candles.extend(batch)
                
                if batch:
                    cursor = batch[-1][0] + 1
                time.sleep(exchange.rateLimit / 1000)
            
            if not all_candles:
                return None
            
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = sym
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch {sym}: {e}")
            return None
    
    dfs = []
    for symbol in symbols:
        df = _fetch_single(symbol)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        return None
    
    result = pd.concat(dfs, ignore_index=True)
    
    if columns_post_process:
        result = columns_post_process(result)
    
    return result


def fetch_binance(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from Binance.
    
    Handles symbol renames (e.g., MATIC -> POL): fetches MATIC historical data
    before 2024-09 and POL data after, then merges them.
    """
    try:
        exchange = ccxt.binance()
        if not exchange.has['fetchOHLCV']:
            logger.error("Binance does not support OHLCV")
            return None
        
        since = exchange.parse8601(f'{start_date}T00:00:00Z') if start_date else None
        until = exchange.parse8601(f'{end_date}T00:00:00Z') if end_date else None
        
        symbols_to_fetch = list(set(symbols))
        
        if 'POL' in symbols_to_fetch:
            pol_since = exchange.parse8601('2024-09-01T00:00:00Z')
            
            if since is None or since < pol_since:
                matic_until = pol_since - 1
                logger.info("Fetching MATIC historical data for POL before 2024-09")
                matic_data = _fetch_ohlcv_data(
                    exchange, 
                    ['MATIC'] + [s for s in symbols_to_fetch if s != 'POL'],
                    timeframe, 
                    since, 
                    matic_until
                )
                
                pol_data = _fetch_ohlcv_data(
                    exchange,
                    symbols_to_fetch,
                    timeframe,
                    pol_since,
                    until
                )
                
                if matic_data is not None and pol_data is not None:
                    matic_data.loc[matic_data['symbol'] == 'MATIC', 'symbol'] = 'POL'
                    result = pd.concat([matic_data, pol_data], ignore_index=True)
                    result = result.sort_values('timestamp').reset_index(drop=True)
                    
                    pol_rows = result[result['symbol'] == 'POL'].copy()
                    if len(pol_rows) > 0:
                        pol_rows = pol_rows.set_index('timestamp').sort_index()
                        pol_rows = pol_rows.reindex(
                            pd.date_range(pol_rows.index.min(), pol_rows.index.max(), freq='D')
                        ).ffill()
                        pol_rows = pol_rows.reset_index().rename(columns={'index': 'timestamp'})
                        pol_rows['volume'] = pol_rows['volume'].fillna(0)
                        result = pd.concat([
                            result[result['symbol'] != 'POL'],
                            pol_rows
                        ], ignore_index=True)
                        result = result.sort_values('timestamp').reset_index(drop=True)
                    
                    return result
                elif matic_data is not None:
                    matic_data.loc[matic_data['symbol'] == 'MATIC', 'symbol'] = 'POL'
                    return matic_data
                elif pol_data is not None:
                    return pol_data
                else:
                    return None
        
        return _fetch_ohlcv_data(exchange, symbols_to_fetch, timeframe, since, until)
        
    except Exception as e:
        logger.error(f"Failed to initialize Binance: {e}")
        return None


def fetch_benchmark(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Fetch market benchmark data (BTC, ETH) for all symbols."""
    try:
        exchange = ccxt.binance()
        if not exchange.has['fetchOHLCV']:
            logger.error("Binance does not support OHLCV")
            return None
        
        since = exchange.parse8601(f'{start_date}T00:00:00Z') if start_date else None
        until = exchange.parse8601(f'{end_date}T00:00:00Z') if end_date else None
        
        def extract_close(df):
            return df[['timestamp', 'close']]
        
        factor_data = {}
        for factor in ['BTC', 'ETH']:
            df = _fetch_ohlcv_data(exchange, [factor], timeframe, since, until, extract_close)
            if df is not None:
                df = df.rename(columns={'close': f'{factor}_close'})
                df = df.set_index('timestamp')
                factor_data[factor] = df
        
        if not factor_data:
            logger.warning("No factor data fetched")
            return None
        
        combined = pd.concat(factor_data.values(), axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]
        
        combined = combined.reset_index()
        
        rows = [
            {
                'timestamp': ts,
                'symbol': sym,
                **row.to_dict()
            }
            for sym in symbols
            for ts, row in combined.iterrows()
        ]
        
        return pd.DataFrame(rows) if rows else None
        
    except Exception as e:
        logger.error(f"Failed to fetch benchmark: {e}")
        return None


def fetch_calendar(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Fetch time-based features (year, month, day, dayofweek)."""
    if not start_date or not end_date:
        logger.warning("Calendar requires both start_date and end_date")
        return None
    
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        freq = TIMEFRAME_MAP.get(timeframe, 'D')
        dates = pd.date_range(start=start, end=end, freq=freq)
        
        rows = [
            {
                'timestamp': date,
                'symbol': sym,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'dayofweek': date.dayofweek + 1,
                'dayofmonth_position': 1 + (date.day - 1) // 10,
                'is_week_end': int(date.dayofweek >= 5),
            }
            for sym in symbols
            for date in dates
        ]
        
        return pd.DataFrame(rows) if rows else None
        
    except Exception as e:
        logger.error(f"Failed to generate calendar: {e}")
        return None


def _process_data(df: pd.DataFrame, timeframe: str, user_symbols: List[str]) -> pd.DataFrame:
    """Align multi-source data to common time range and forward fill gaps."""
    pivoted = df.pivot_table(index='timestamp', columns='symbol', values='close')
    common_start = pivoted.apply(lambda s: s.first_valid_index()).max()
    end_date = df['timestamp'].max()
    freq = TIMEFRAME_MAP.get(timeframe, 'D')
    full_range = pd.date_range(start=common_start, end=end_date, freq=freq)
    
    result_dfs = []
    for col in df.columns:
        if col not in ['timestamp', 'symbol']:
            pivot = df.pivot_table(index='timestamp', columns='symbol', values=col)
            pivot = pivot[pivot.index >= common_start].reindex(full_range).ffill()
            stacked = pivot.stack(future_stack=True).reset_index()
            stacked.columns = ['timestamp', 'symbol', col]
            result_dfs.append(stacked.set_index(['timestamp', 'symbol']))
    
    result = pd.concat(result_dfs, axis=1).sort_index()
    return result[result.index.get_level_values('symbol').isin(user_symbols)]


def fetch_vwap(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Fetch and calculate Volume Weighted Average Price (VWAP).
    
    For daily ('1d'): Aggregates hourly data for accurate daily VWAP.
    For intraday: Cumulative VWAP anchored to trading day start (UTC).
    """
    try:
        is_daily = timeframe == '1d'
        fetch_tf = '1h' if is_daily else timeframe
        
        if start_date:
            extended_start = pd.to_datetime(start_date).normalize().strftime('%Y-%m-%d %H:%M:%S')
        else:
            extended_start = None

        df = fetch_binance(symbols, fetch_tf, extended_start, end_date)
        if df is None or df.empty:
            return None

        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['pv'] = df['typical_price'] * df['volume']
        df['date'] = df['timestamp'].dt.date

        if is_daily:
            agg = df.groupby(['symbol', 'date']).agg({
                'pv': 'sum',
                'volume': 'sum',
                'timestamp': 'first'
            }).reset_index()
            agg['vwap'] = agg['pv'] / agg['volume']
            agg['timestamp'] = pd.to_datetime(agg['date'])
            result_df = agg[['timestamp', 'symbol', 'vwap']]
        else:
            df['pv_cumsum'] = df.groupby(['symbol', 'date'])['pv'].cumsum()
            df['vol_cumsum'] = df.groupby(['symbol', 'date'])['volume'].cumsum()
            df['vwap'] = df['pv_cumsum'] / df['vol_cumsum']
            result_df = df[['timestamp', 'symbol', 'vwap']]

        if start_date:
            result_df = result_df[result_df['timestamp'] >= pd.to_datetime(start_date)]

        return result_df

    except Exception as e:
        logger.error(f"Failed to calculate VWAP: {e}")
        return None