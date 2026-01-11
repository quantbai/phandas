"""OKX Rubik API data provider.

Fetches trading big data for perpetual swaps including:
- Open interest (USD and contract units)
- Long/short account ratio
- Elite trader position and account ratios
- Taker buy/sell volume
"""

import warnings
import pandas as pd
from typing import List, Optional

from .base import RestClient, paginate_ts, to_timestamp_ms


OKX_BASE_URL = "https://www.okx.com"
OKX_RATE_LIMIT_MS = 450

PERIOD_MAP = {
    '1d': '1Dutc',
    '1h': '1H',
    '4h': '4H',
    '12h': '12Hutc',
}


def fetch_okx(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch OKX perpetual swap trading metrics.
    
    Parameters
    ----------
    symbols : List[str]
        Cryptocurrency symbols (e.g., ['ETH', 'SOL'])
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    
    Returns
    -------
    pd.DataFrame or None
        Columns: timestamp, symbol, okx_oi_usd, okx_oi_ccy, okx_ls_acct_ratio,
                 okx_elite_ls_pos_ratio, okx_elite_ls_acct_ratio,
                 okx_taker_buy_vol, okx_taker_sell_vol
    
    Notes
    -----
    Data availability: 2024-01-01 onwards for daily granularity.
    Rate limit: 450ms between requests.
    OKX removed MATIC-USDT-SWAP after POL rename; POL data starts ~2024-09-25.
    """
    client = RestClient(OKX_BASE_URL, rate_limit_ms=OKX_RATE_LIMIT_MS)
    start_ts = to_timestamp_ms(start_date)
    end_ts = to_timestamp_ms(end_date)
    
    all_dfs = []
    
    for symbol in symbols:
        df = _fetch_symbol_metrics(client, symbol, start_ts, end_ts)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        warnings.warn("No OKX data fetched for any symbol")
        return None
    
    result = pd.concat(all_dfs, ignore_index=True)
    return result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)


def _fetch_symbol_metrics(
    client: RestClient,
    symbol: str,
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> Optional[pd.DataFrame]:
    """Fetch all metrics for a single symbol and merge."""
    inst_id = f"{symbol}-USDT-SWAP"
    period = PERIOD_MAP.get('1d', '1Dutc')
    
    fetchers = [
        (_fetch_open_interest, ['okx_oi_usd', 'okx_oi_ccy']),
        (_fetch_ls_ratio, ['okx_ls_acct_ratio']),
        (_fetch_elite_ls_pos_ratio, ['okx_elite_ls_pos_ratio']),
        (_fetch_elite_ls_acct_ratio, ['okx_elite_ls_acct_ratio']),
        (_fetch_taker_volume, ['okx_taker_buy_vol', 'okx_taker_sell_vol']),
    ]
    
    result = None
    
    for fetcher, columns in fetchers:
        df = fetcher(client, inst_id, period, start_ts, end_ts)
        if df is None:
            continue
        
        df['symbol'] = symbol
        
        if result is None:
            result = df
        else:
            result = pd.merge(result, df, on=['timestamp', 'symbol'], how='outer')
    
    return result


def _okx_response_extractor(response: dict) -> list:
    """Extract data array from OKX API response."""
    if response.get("code") != "0":
        raise ValueError(f"OKX API error: {response.get('msg', 'Unknown')}")
    return response.get("data", [])


def _fetch_open_interest(
    client: RestClient,
    inst_id: str,
    period: str,
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> Optional[pd.DataFrame]:
    """Fetch open interest history."""
    params = {"instId": inst_id, "period": period}
    
    data = paginate_ts(
        client,
        "/api/v5/rubik/stat/contracts/open-interest-history",
        params,
        start_ts,
        end_ts,
        response_extractor=_okx_response_extractor,
    )
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=["timestamp", "oi", "oi_ccy", "oi_usd"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df["okx_oi_usd"] = df["oi_usd"].astype(float)
    df["okx_oi_ccy"] = df["oi_ccy"].astype(float)
    
    return df[["timestamp", "okx_oi_usd", "okx_oi_ccy"]].drop_duplicates(subset=["timestamp"])


def _fetch_ls_ratio(
    client: RestClient,
    inst_id: str,
    period: str,
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> Optional[pd.DataFrame]:
    """Fetch long/short account ratio."""
    params = {"instId": inst_id, "period": period}
    
    data = paginate_ts(
        client,
        "/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
        params,
        start_ts,
        end_ts,
        response_extractor=_okx_response_extractor,
    )
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=["timestamp", "ls_ratio"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df["okx_ls_acct_ratio"] = df["ls_ratio"].astype(float)
    
    return df[["timestamp", "okx_ls_acct_ratio"]].drop_duplicates(subset=["timestamp"])


def _fetch_elite_ls_pos_ratio(
    client: RestClient,
    inst_id: str,
    period: str,
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> Optional[pd.DataFrame]:
    """Fetch elite trader long/short position ratio (top 5% by position value)."""
    params = {"instId": inst_id, "period": period}
    
    data = paginate_ts(
        client,
        "/api/v5/rubik/stat/contracts/long-short-position-ratio-contract-top-trader",
        params,
        start_ts,
        end_ts,
        response_extractor=_okx_response_extractor,
    )
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=["timestamp", "ls_pos_ratio"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df["okx_elite_ls_pos_ratio"] = df["ls_pos_ratio"].astype(float)
    
    return df[["timestamp", "okx_elite_ls_pos_ratio"]].drop_duplicates(subset=["timestamp"])


def _fetch_elite_ls_acct_ratio(
    client: RestClient,
    inst_id: str,
    period: str,
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> Optional[pd.DataFrame]:
    """Fetch elite trader long/short account ratio (top 5% by position value)."""
    params = {"instId": inst_id, "period": period}
    
    data = paginate_ts(
        client,
        "/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader",
        params,
        start_ts,
        end_ts,
        response_extractor=_okx_response_extractor,
    )
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=["timestamp", "ls_acct_ratio"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df["okx_elite_ls_acct_ratio"] = df["ls_acct_ratio"].astype(float)
    
    return df[["timestamp", "okx_elite_ls_acct_ratio"]].drop_duplicates(subset=["timestamp"])


def _fetch_taker_volume(
    client: RestClient,
    inst_id: str,
    period: str,
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> Optional[pd.DataFrame]:
    """Fetch taker buy/sell volume in USD."""
    params = {"instId": inst_id, "period": period, "unit": "2"}
    
    data = paginate_ts(
        client,
        "/api/v5/rubik/stat/taker-volume-contract",
        params,
        start_ts,
        end_ts,
        response_extractor=_okx_response_extractor,
    )
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=["timestamp", "sell_vol", "buy_vol"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df["okx_taker_buy_vol"] = df["buy_vol"].astype(float)
    df["okx_taker_sell_vol"] = df["sell_vol"].astype(float)
    
    return df[["timestamp", "okx_taker_buy_vol", "okx_taker_sell_vol"]].drop_duplicates(
        subset=["timestamp"]
    )
