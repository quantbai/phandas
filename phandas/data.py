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


def fetch_data(
    symbols: List[str], 
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> 'Panel':
    """Fetch, merge, and align multi-source data to common time range.
    
    Parameters
    ----------
    symbols : list of str
        Symbols to fetch
    timeframe : str, default '1d'
        Timeframe ('1m', '5m', '1h', '1d', '1w', '1M')
    start_date : str, optional
        Start date ('YYYY-MM-DD')
    end_date : str, optional
        End date ('YYYY-MM-DD')
    sources : list of str, optional
        Data sources. Default ['binance']
    output_path : str, optional
        Path to save output CSV
    
    Returns
    -------
    Panel
        Merged and aligned data from all sources
    """
    if sources is None:
        sources = ['binance']
    
    source_map = {
        'binance': fetch_binance,
        'benchmark': fetch_benchmark,
        'calendar': fetch_calendar,
    }
    
    raw_dfs = []
    for source in sources:
        if source not in source_map:
            raise ValueError(f"Unknown source: {source}. Available: {list(source_map.keys())}")
        
        logger.info(f"Fetching from {source}...")
        df = source_map[source](symbols, timeframe, start_date, end_date)
        if df is not None:
            if not isinstance(df.index, pd.MultiIndex):
                if 'timestamp' in df.columns and 'symbol' in df.columns:
                    df = df.set_index(['timestamp', 'symbol'])
            raw_dfs.append(df)
    
    if not raw_dfs:
        raise ValueError("No data fetched from any source")
    
    combined = pd.concat(raw_dfs, axis=1)
    
    dup_cols = combined.columns[combined.columns.duplicated()]
    if len(dup_cols) > 0:
        logger.warning(f"Duplicate columns will be dropped: {list(dup_cols)}")
        combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]
    
    logger.info(f"Combined columns: {list(combined.columns)}")
    
    combined_reset = combined.reset_index()
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


def fetch_binance(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from Binance.
    
    Parameters
    ----------
    symbols : list of str
        Symbols to fetch
    timeframe : str, default '1d'
        Timeframe ('1m', '5m', '1h', '1d', '1w', '1M')
    start_date : str, optional
        Start date ('YYYY-MM-DD')
    end_date : str, optional
        End date ('YYYY-MM-DD')
    
    Returns
    -------
    pandas.DataFrame or None
        OHLCV data (timestamp, symbol, open, high, low, close, volume)
        Returns None on failure
    """
    SYMBOL_MAP = {'MATIC': ['MATIC', 'POL'], 'POL': ['MATIC', 'POL']}
    
    def _fetch_single_symbol(exchange, symbol: str, timeframe: str, since, until=None) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a symbol with pagination."""
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
                
                if until:
                    ohlcv = [candle for candle in ohlcv if candle[0] <= until]
                    all_ohlcv.extend(ohlcv)
                    if len(ohlcv) < limit:
                        break
                else:
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
    
    def _fetch_renamed_symbol(exchange, symbols_list: List[str], timeframe: str, since, until=None) -> Optional[pd.DataFrame]:
        """Fetch and merge data for renamed symbols (e.g., MATIC->POL)."""
        dfs = []
        for symbol in symbols_list:
            df = _fetch_single_symbol(exchange, symbol, timeframe, since, until)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            return None
        
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
        return combined
    
    user_symbols = list(set(symbols))
    download_symbols = user_symbols
    
    try:
        exchange_obj = ccxt.binance()
    except Exception as e:
        logger.error(f"Failed to initialize Binance: {e}")
        return None
    
    if not exchange_obj.has['fetchOHLCV']:
        logger.error("Binance does not support OHLCV")
        return None
    
    since = exchange_obj.parse8601(f'{start_date}T00:00:00Z') if start_date else None
    until = exchange_obj.parse8601(f'{end_date}T23:59:59Z') if end_date else None
    
    all_data = []
    for symbol in download_symbols:
        if symbol in SYMBOL_MAP:
            data = _fetch_renamed_symbol(exchange_obj, SYMBOL_MAP[symbol], timeframe, since, until)
            if data is not None:
                data['symbol'] = 'POL'
                all_data.append(data)
        else:
            data = _fetch_single_symbol(exchange_obj, symbol, timeframe, since, until)
            if data is not None:
                all_data.append(data)
    
    if not all_data:
        logger.warning("No data fetched from Binance")
        return None
    
    return pd.concat(all_data, ignore_index=True)


def fetch_benchmark(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Fetch market benchmark data (BTC, ETH close prices).
    
    Parameters
    ----------
    symbols : list of str
        Symbols to attach benchmarks to
    timeframe : str, default '1d'
        Timeframe ('1m', '5m', '1h', '1d', '1w', '1M')
    start_date : str, optional
        Start date ('YYYY-MM-DD')
    end_date : str, optional
        End date ('YYYY-MM-DD')
    
    Returns
    -------
    pandas.DataFrame or None
        Columns (timestamp, symbol, BTC_close, ETH_close)
        Each symbol has identical benchmark values
        Returns None on failure
    """
    SYMBOL_MAP = {'MATIC': ['MATIC', 'POL'], 'POL': ['MATIC', 'POL']}
    
    def _fetch_single_symbol(exchange, symbol: str, timeframe: str, since, until=None) -> Optional[pd.DataFrame]:
        """Fetch close price for a symbol."""
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
                
                if until:
                    ohlcv = [candle for candle in ohlcv if candle[0] <= until]
                    all_ohlcv.extend(ohlcv)
                    if len(ohlcv) < limit:
                        break
                else:
                    all_ohlcv.extend(ohlcv)
                
                since = ohlcv[-1][0] + 1
                time.sleep(exchange.rateLimit / 1000)
            
            if not all_ohlcv:
                return None
                
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df[['timestamp', 'close']]
            
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            return None
    
    def _fetch_renamed_symbol(exchange, symbols_list: List[str], timeframe: str, since, until=None) -> Optional[pd.DataFrame]:
        """Fetch and merge close prices for renamed symbols."""
        dfs = []
        for symbol in symbols_list:
            df = _fetch_single_symbol(exchange, symbol, timeframe, since, until)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            return None
        
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
        return combined[['timestamp', 'close']]
    
    try:
        exchange_obj = ccxt.binance()
    except Exception as e:
        logger.error(f"Failed to initialize Binance: {e}")
        return None
    
    if not exchange_obj.has['fetchOHLCV']:
        logger.error("Binance does not support OHLCV")
        return None
    
    since = exchange_obj.parse8601(f'{start_date}T00:00:00Z') if start_date else None
    until = exchange_obj.parse8601(f'{end_date}T23:59:59Z') if end_date else None
    
    factor_symbols = ['BTC', 'ETH']
    factor_data = {}
    
    for factor in factor_symbols:
        if factor in SYMBOL_MAP:
            data = _fetch_renamed_symbol(exchange_obj, SYMBOL_MAP[factor], timeframe, since, until)
            if data is not None:
                data = data.rename(columns={'close': f'{factor}_close'})
                factor_data[factor] = data
        else:
            data = _fetch_single_symbol(exchange_obj, factor, timeframe, since, until)
            if data is not None:
                data = data.rename(columns={'close': f'{factor}_close'})
                factor_data[factor] = data
    
    if not factor_data:
        logger.warning("No factor data fetched")
        return None
    
    combined = list(factor_data.values())[0]
    for data in list(factor_data.values())[1:]:
        combined = combined.merge(data, on='timestamp', how='outer')
    
    rows = []
    for symbol in symbols:
        for _, row in combined.iterrows():
            rows.append({
                'timestamp': row['timestamp'],
                'symbol': symbol,
                'BTC_close': row['BTC_close'],
                'ETH_close': row['ETH_close'],
            })
    
    if not rows:
        logger.warning("No factor records generated")
        return None
    
    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df)} factor records for {len(symbols)} symbols")
    return df


def fetch_calendar(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Fetch time-based features (year, month, day) for symbols.
    
    Args:
        symbols: List of symbols to generate features for
        timeframe: Timeframe (used to generate correct date range, default: '1d')
        start_date: Start date (e.g. '2020-01-01'). Required.
        end_date: End date (e.g. '2024-01-01'). If None, uses today's date.
    
    Returns:
        DataFrame with columns (timestamp, symbol, year, month, day), or None if start_date missing
    """
    try:
        if not start_date:
            logger.warning("Calendar requires start_date")
            return None
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) if end_date else pd.Timestamp.today()
        
        FREQ_MAP = {
            '1m': 'min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': 'h', '4h': '4h', '1d': 'D', '1w': 'W', '1M': 'MS',
        }
        freq = FREQ_MAP.get(timeframe, 'D')
        dates = pd.date_range(start=start, end=end, freq=freq)
        
        rows = []
        for symbol in symbols:
            for date in dates:
                dayofmonth = date.day
                if dayofmonth <= 10:
                    dayofmonth_position = 1  # 月初
                elif dayofmonth <= 20:
                    dayofmonth_position = 2  # 月中
                else:
                    dayofmonth_position = 3  # 月末
                
                is_week_end = 1 if date.dayofweek + 1 >= 6 else 0
                
                rows.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'dayofweek': date.dayofweek + 1,  # 1-7 (Mon-Sun)
                    'dayofmonth_position': dayofmonth_position,  # 1=月初, 2=月中, 3=月末
                    'is_week_end': is_week_end,  # 1=週末, 0=工作日
                })
        
        if not rows:
            logger.warning("No dates generated from calendar")
            return None
        
        df = pd.DataFrame(rows)
        logger.info(f"Generated {len(df)} calendar records for {len(symbols)} symbols")
        return df
        
    except Exception as e:
        logger.warning(f"Failed to generate calendar: {e}")
        return None


def _process_data(df: pd.DataFrame, timeframe: str, user_symbols: List[str]) -> pd.DataFrame:
    """Align data to common time range (齊頭) and fill gaps forward."""
    FREQ_MAP = {
        '1m': 'min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': 'h', '4h': '4h', '1d': 'D', '1w': 'W', '1M': 'MS',
    }
    
    pivoted = df.pivot_table(index='timestamp', columns='symbol', values='close')
    common_start = pivoted.apply(lambda s: s.first_valid_index()).max()
    end_date = df['timestamp'].max()
    
    freq = FREQ_MAP.get(timeframe, 'D')
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

