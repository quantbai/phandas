"""Base utilities for data providers.

This module provides shared infrastructure for all data providers:
- RestClient: Rate-limited HTTP client for REST APIs
- Pagination utilities for cursor-based APIs
- Symbol mapping dictionaries for cross-provider normalization
"""

import time
import warnings
import requests
import pandas as pd
from typing import List, Optional, Dict, Any, Callable


class RestClient:
    """Rate-limited REST API client.
    
    Parameters
    ----------
    base_url : str
        Base URL for API requests
    rate_limit_ms : int, default 300
        Minimum milliseconds between requests
    timeout : int, default 30
        Request timeout in seconds
    headers : dict, optional
        Additional HTTP headers
    """
    
    def __init__(
        self,
        base_url: str,
        rate_limit_ms: int = 300,
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.rate_limit_ms = rate_limit_ms
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        if headers:
            self.session.headers.update(headers)
        self._last_request_time = 0
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute rate-limited GET request.
        
        Parameters
        ----------
        endpoint : str
            API endpoint path
        params : dict, optional
            Query parameters
        
        Returns
        -------
        Any
            Parsed JSON response
        
        Raises
        ------
        RuntimeError
            If request fails
        """
        self._apply_rate_limit()
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {url} - {e}")
    
    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = (time.time() * 1000) - self._last_request_time
        if elapsed < self.rate_limit_ms:
            time.sleep((self.rate_limit_ms - elapsed) / 1000)
        self._last_request_time = time.time() * 1000


def paginate_ts(
    client: RestClient,
    endpoint: str,
    params: Dict[str, Any],
    start_ts: Optional[int],
    end_ts: Optional[int],
    limit: int = 100,
    response_extractor: Callable[[Dict], List] = lambda r: r.get("data", []),
    ts_extractor: Callable[[Any], int] = lambda row: int(row[0]),
    direction: str = "backward",
) -> List[Any]:
    """Paginate API with timestamp-based cursor.
    
    Parameters
    ----------
    client : RestClient
        API client instance
    endpoint : str
        API endpoint
    params : dict
        Base query parameters
    start_ts : int, optional
        Start timestamp in milliseconds
    end_ts : int, optional
        End timestamp in milliseconds
    limit : int, default 100
        Records per page
    response_extractor : callable
        Function to extract data list from response
    ts_extractor : callable
        Function to extract timestamp from record
    direction : str, default "backward"
        Pagination direction ("backward" or "forward")
    
    Returns
    -------
    list
        All fetched records
    """
    all_data = []
    request_params = {**params, "limit": str(limit)}
    
    if start_ts is not None:
        request_params["begin"] = str(start_ts)
    if end_ts is not None:
        request_params["end"] = str(end_ts)
    
    while True:
        try:
            response = client.get(endpoint, request_params)
            data = response_extractor(response)
            
            if not data:
                break
            
            all_data.extend(data)
            
            if len(data) < limit:
                break
            
            if direction == "backward":
                oldest_ts = min(ts_extractor(row) for row in data)
                request_params["end"] = str(oldest_ts - 1)
            else:
                newest_ts = max(ts_extractor(row) for row in data)
                request_params["begin"] = str(newest_ts + 1)
                
        except Exception as e:
            warnings.warn(f"Pagination error on {endpoint}: {e}")
            break
    
    return all_data


def to_timestamp_ms(date_str: Optional[str]) -> Optional[int]:
    """Convert date string to millisecond timestamp.
    
    Parameters
    ----------
    date_str : str, optional
        Date in YYYY-MM-DD format
    
    Returns
    -------
    int or None
        Millisecond timestamp or None if input is None
    """
    if date_str is None:
        return None
    return int(pd.Timestamp(date_str).timestamp() * 1000)


SYMBOL_TO_CHAIN = {
    'ETH': 'Ethereum',
    'SOL': 'Solana',
    'ARB': 'Arbitrum',
    'OP': 'Optimism',
    'POL': 'Polygon',
    'SUI': 'Sui',
    'AVAX': 'Avalanche',
    'BNB': 'BSC',
    'FTM': 'Fantom',
    'MATIC': 'Polygon',
    'NEAR': 'Near',
    'APT': 'Aptos',
    'INJ': 'Injective',
    'SEI': 'Sei',
    'HYPE': 'Hyperliquid L1',
    'TON': 'TON',
    'TRX': 'Tron',
}

SYMBOL_TO_STABLECOIN_CHAIN = {
    'ETH': 'Ethereum',
    'SOL': 'Solana',
    'ARB': 'Arbitrum',
    'OP': 'OP Mainnet',
    'POL': 'Polygon',
    'SUI': 'Sui',
    'AVAX': 'Avalanche',
    'BNB': 'BSC',
    'FTM': 'Fantom',
    'MATIC': 'Polygon',
    'NEAR': 'Near',
    'APT': 'Aptos',
    'INJ': 'Injective',
    'SEI': 'Sei',
    'HYPE': 'Hyperliquid L1',
    'TON': 'TON',
    'TRX': 'Tron',
}

SYMBOL_RENAMES = {
    'POL': {
        'old_symbol': 'MATIC',
        'new_symbol': 'POL',
        'cutoff_date': '2024-09-01',
    }
}

TIMEFRAME_MAP = {
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
