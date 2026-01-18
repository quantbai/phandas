"""CoinGecko API data provider.

Fetches historical market data including:
- Market Cap (coingecko_mcap)
- Total Volume (coingecko_volume)
- Price (coingecko_price)

Limit: 30 calls/min (Demo Plan)
History: Past 365 days only (Demo Plan)
"""

import warnings
import pandas as pd
import time
from typing import List, Optional, Dict
from .base import RestClient

# Demo API Key provided by user
API_KEY = "CG-ZgCHaeesQdTihhnJPDTz****"
API_BASE_URL = "https://api.coingecko.com/api/v3"

# Symbol to CoinGecko ID mapping
# TODO: Move this to a central config or dynamic lookup
SYMBOL_TO_ID = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
    'ARB': 'arbitrum',
    'OP': 'optimism',
    'POL': 'polygon-ecosystem-token',
    'MATIC': 'matic-network', # Old MATIC
    'SUI': 'sui',
    'BNB': 'binancecoin',
    'DOGE': 'dogecoin',
    'XRP': 'ripple',
    'ADA': 'cardano',
    'AVAX': 'avalanche-2',
}

def fetch_coingecko(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch CoinGecko market chart data.
    
    Parameters
    ----------
    symbols : List[str]
        Cryptocurrency symbols (e.g., ['ETH', 'SOL'])
    start_date : str, optional
        Start date (YYYY-MM-DD). Note: Demo API only supports last 365 days.
    end_date : str, optional
        End date (YYYY-MM-DD)
        
    Returns
    -------
    pd.DataFrame
        Columns: timestamp, symbol, coingecko_mcap, coingecko_volume, coingecko_price
    """
    client = RestClient(API_BASE_URL, rate_limit_ms=2000) # Conservative 2s delay (30 calls/min)
    
    all_dfs = []
    
    for symbol in symbols:
        coin_id = SYMBOL_TO_ID.get(symbol)
        if not coin_id:
            warnings.warn(f"No CoinGecko ID mapping for {symbol}, skipping.")
            continue
            
        df = _fetch_single_coin(client, coin_id, symbol, API_KEY)
        if df is not None:
            all_dfs.append(df)
            
    if not all_dfs:
        return None
        
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Filter by date if provided
    if start_date:
        combined = combined[combined['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        combined = combined[combined['timestamp'] <= pd.to_datetime(end_date)]
        
    return combined.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

def _fetch_single_coin(client, coin_id, symbol, api_key):
    """Helper to fetch data for a single coin ID."""
    try:
        # Fetch daily data (Demo API restricted to 365 days)
        params = {
            "vs_currency": "usd",
            "days": "365", 
            "interval": "daily",
            "x_cg_demo_api_key": api_key
        }
        
        data = client.get(f"/coins/{coin_id}/market_chart", params=params)
        
        if not data:
            return None
            
        # Process Market Caps
        mcaps = data.get('market_caps', [])
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        if not mcaps:
            return None
            
        # Create DataFrame
        df = pd.DataFrame(mcaps, columns=['timestamp', 'coingecko_mcap'])
        
        # Merge Prices if available
        if prices:
            df_p = pd.DataFrame(prices, columns=['timestamp', 'coingecko_price'])
            df = pd.merge(df, df_p, on='timestamp', how='left')
            
        # Merge Volumes if available
        if volumes:
            df_v = pd.DataFrame(volumes, columns=['timestamp', 'coingecko_volume'])
            df = pd.merge(df, df_v, on='timestamp', how='left')
        
        # Clean up timestamp (ms to datetime)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        
        return df
        
    except Exception as e:
        warnings.warn(f"Failed to fetch CoinGecko data for {symbol} ({coin_id}): {e}")
        return None
