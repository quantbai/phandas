"""Data acquisition and management for cryptocurrency markets via CCXT."""

import warnings
import pandas as pd
import ccxt
import time
import os
from typing import List, Optional, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .panel import Panel

from .constants import SYMBOL_RENAMES

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
    """Fetch, merge, and align multi-source cryptocurrency data.
    
    Parameters
    ----------
    symbols : List[str]
        List of cryptocurrency symbols (e.g., ['BTC', 'ETH'])
    timeframe : str, default '1d'
        OHLCV timeframe ('1m', '5m', '15m', '1h', '4h', '1d', '1w')
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    sources : List[str], optional
        Data sources to fetch from. Default is ['binance']
    output_path : str, optional
        Path to save CSV output
    
    Returns
    -------
    Panel
        Merged and aligned data from all sources
    
    Notes
    -----
    Defaults to daily resolution and Binance OHLCV data.
    Multi-source data is aligned to common time range.
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
            warnings.warn(f"Unknown source: {source}. Available: {list(source_map.keys())}")
            continue
        
        try:
            if source == 'binance':
                df = source_map[source](symbols, timeframe, start_date, end_date)
                if df is not None and 'timestamp' in df.columns:
                    binance_end_date = df['timestamp'].max().strftime('%Y-%m-%d')
            else:
                source_end_date = binance_end_date or end_date
                df = source_map[source](symbols, timeframe, start_date, source_end_date)
            
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
    
    combined = raw_dfs[0]
    for df in raw_dfs[1:]:
        combined = pd.merge(combined, df, on=['timestamp', 'symbol'], how='outer')
    
    if combined.columns.duplicated().any():
        combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]
    
    combined_reset = combined.copy()
    if 'index' in combined_reset.columns:
        combined_reset = combined_reset.drop(columns=['index'])
    
    processed = _process_data(combined_reset, timeframe, symbols)
    
    int_cols = ['year', 'month', 'day']
    for col in int_cols:
        if col in processed.columns:
            processed[col] = processed[col].astype('Int64')
    
    from .panel import Panel
    result = Panel(processed)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path)
    
    return result


def _fetch_ohlcv_data(
    exchange,
    symbols: List[str], 
    timeframe: str, 
    since: Optional[int],
    until: Optional[int] = None,
    columns_post_process: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
) -> Optional[pd.DataFrame]:
    def _fetch_single(sym: str) -> Optional[pd.DataFrame]:
        try:
            market_sym = f'{sym}/USDT'
            exchange.load_markets()
            if market_sym not in exchange.symbols:
                warnings.warn(f"{market_sym} not available")
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
            warnings.warn(f"Failed to fetch {sym}: {e}")
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
    try:
        exchange = ccxt.binance()
        if not exchange.has['fetchOHLCV']:
            raise RuntimeError("Binance does not support OHLCV")
        
        since = exchange.parse8601(f'{start_date}T00:00:00Z') if start_date else None
        until = exchange.parse8601(f'{end_date}T00:00:00Z') if end_date else None
        
        symbols_to_fetch = list(set(symbols))
        
        for new_sym, rename_info in SYMBOL_RENAMES.items():
            if new_sym not in symbols_to_fetch:
                continue
            
            old_sym = rename_info['old_symbol']
            cutoff_date = rename_info['cutoff_date']
            cutoff_ts = exchange.parse8601(f'{cutoff_date}T00:00:00Z')
            
            if since is None or since < cutoff_ts:
                old_until = cutoff_ts - 1
                
                old_data = _fetch_ohlcv_data(
                    exchange, 
                    [old_sym] + [s for s in symbols_to_fetch if s != new_sym],
                    timeframe, 
                    since, 
                    old_until
                )
                
                new_data = _fetch_ohlcv_data(
                    exchange,
                    symbols_to_fetch,
                    timeframe,
                    cutoff_ts,
                    until
                )
                
                if old_data is not None and new_data is not None:
                    old_data.loc[old_data['symbol'] == old_sym, 'symbol'] = new_sym
                    result = pd.concat([old_data, new_data], ignore_index=True)
                    result = result.sort_values('timestamp').reset_index(drop=True)
                    
                    renamed_rows = result[result['symbol'] == new_sym].copy()
                    if len(renamed_rows) > 0:
                        renamed_rows = renamed_rows.set_index('timestamp').sort_index()
                        renamed_rows = renamed_rows.reindex(
                            pd.date_range(renamed_rows.index.min(), renamed_rows.index.max(), freq='D')
                        ).ffill()
                        renamed_rows = renamed_rows.reset_index().rename(columns={'index': 'timestamp'})
                        renamed_rows['volume'] = renamed_rows['volume'].fillna(0)
                        result = pd.concat([
                            result[result['symbol'] != new_sym],
                            renamed_rows
                        ], ignore_index=True)
                        result = result.sort_values('timestamp').reset_index(drop=True)
                    
                    return result
                elif old_data is not None:
                    old_data.loc[old_data['symbol'] == old_sym, 'symbol'] = new_sym
                    return old_data
                elif new_data is not None:
                    return new_data
                else:
                    return None
        
        return _fetch_ohlcv_data(exchange, symbols_to_fetch, timeframe, since, until)
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Binance: {e}")


def fetch_benchmark(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    try:
        exchange = ccxt.binance()
        if not exchange.has['fetchOHLCV']:
            raise RuntimeError("Binance does not support OHLCV")
        
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
            warnings.warn("No factor data fetched")
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
        raise RuntimeError(f"Failed to fetch benchmark: {e}")


def fetch_calendar(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    if not start_date or not end_date:
        raise ValueError("Calendar requires both start_date and end_date")
    
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
        raise RuntimeError(f"Failed to generate calendar: {e}")


def _process_data(df: pd.DataFrame, timeframe: str, user_symbols: List[str]) -> pd.DataFrame:
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
            result_dfs.append(stacked)
    
    result = result_dfs[0]
    for df_part in result_dfs[1:]:
        result = pd.merge(result, df_part, on=['timestamp', 'symbol'], how='outer')
    
    result = result[result['symbol'].isin(user_symbols)]
    return result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)


def fetch_vwap(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
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
        raise RuntimeError(f"Failed to calculate VWAP: {e}")