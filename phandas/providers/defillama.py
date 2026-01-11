"""DefiLlama API data provider.

Fetches on-chain metrics by blockchain including:
- Total Value Locked (TVL)
- Stablecoin supply
- DEX trading volume
- Protocol fees and revenue
"""

import warnings
import pandas as pd
from typing import List, Optional, Dict

from .base import RestClient, SYMBOL_TO_CHAIN, SYMBOL_TO_STABLECOIN_CHAIN


STABLECOINS_BASE_URL = "https://stablecoins.llama.fi"
API_BASE_URL = "https://api.llama.fi"
RATE_LIMIT_MS = 300


def fetch_defillama(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch DefiLlama on-chain metrics by blockchain.
    
    Parameters
    ----------
    symbols : List[str]
        Cryptocurrency symbols (e.g., ['ARB', 'OP', 'SUI'])
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    
    Returns
    -------
    pd.DataFrame or None
        Columns: timestamp, symbol, defillama_tvl, defillama_stablecoin_supply,
                 defillama_dex_volume, defillama_fees, defillama_revenue
    
    Notes
    -----
    Maps token symbols to blockchain names (e.g., ARB -> Arbitrum).
    Only daily resolution is supported.
    """
    api_client = RestClient(API_BASE_URL, rate_limit_ms=RATE_LIMIT_MS)
    stablecoin_client = RestClient(STABLECOINS_BASE_URL, rate_limit_ms=RATE_LIMIT_MS)
    
    chains_to_fetch = _build_chain_mapping(symbols, SYMBOL_TO_CHAIN)
    stablecoin_chains = _build_chain_mapping(symbols, SYMBOL_TO_STABLECOIN_CHAIN)
    
    if not chains_to_fetch and not stablecoin_chains:
        warnings.warn("No valid chain mappings found for any symbol")
        return None
    
    all_dfs = []
    
    for chain, chain_symbols in stablecoin_chains.items():
        df = _fetch_stablecoin_supply(stablecoin_client, chain, start_date, end_date)
        if df is not None:
            for symbol in chain_symbols:
                symbol_df = df.copy()
                symbol_df["symbol"] = symbol
                all_dfs.append(symbol_df)
    
    for chain, chain_symbols in chains_to_fetch.items():
        tvl_df = _fetch_tvl(api_client, chain, start_date, end_date)
        dex_df = _fetch_dex_volume(api_client, chain, start_date, end_date)
        fees_df = _fetch_fees_revenue(api_client, chain, start_date, end_date)
        
        for symbol in chain_symbols:
            if tvl_df is not None:
                df = tvl_df.copy()
                df["symbol"] = symbol
                all_dfs.append(df)
            
            if dex_df is not None:
                df = dex_df.copy()
                df["symbol"] = symbol
                all_dfs.append(df)
            
            if fees_df is not None:
                df = fees_df.copy()
                df["symbol"] = symbol
                all_dfs.append(df)
    
    if not all_dfs:
        warnings.warn("No DefiLlama data fetched for any chain")
        return None
    
    combined = pd.concat(all_dfs, ignore_index=True)
    result = combined.groupby(["timestamp", "symbol"], as_index=False).first()
    
    return result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _build_chain_mapping(symbols: List[str], mapping: Dict[str, str]) -> Dict[str, List[str]]:
    """Build chain -> symbols mapping from symbol -> chain mapping."""
    result: Dict[str, List[str]] = {}
    
    for symbol in symbols:
        chain = mapping.get(symbol)
        if chain:
            if chain not in result:
                result[chain] = []
            result[chain].append(symbol)
        else:
            warnings.warn(f"No chain mapping for symbol {symbol}")
    
    return result


def _fetch_stablecoin_supply(
    client: RestClient,
    chain: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[pd.DataFrame]:
    """Fetch historical stablecoin supply for a chain."""
    try:
        data = client.get(f"/stablecoincharts/{chain}")
        
        if not data:
            return None
        
        records = []
        for entry in data:
            ts = entry.get("date")
            total = entry.get("totalCirculating", {})
            
            supply = (
                total.get("peggedUSD", 0) +
                total.get("peggedEUR", 0) +
                total.get("peggedVAR", 0)
            )
            
            records.append({
                "timestamp": pd.to_datetime(int(ts), unit="s"),
                "defillama_stablecoin_supply": supply,
            })
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        return _filter_date_range(df, start_date, end_date)
        
    except Exception as e:
        warnings.warn(f"Failed to fetch stablecoin data for {chain}: {e}")
        return None


def _fetch_tvl(
    client: RestClient,
    chain: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[pd.DataFrame]:
    """Fetch historical TVL for a chain."""
    try:
        data = client.get(f"/v2/historicalChainTvl/{chain}")
        
        if not data:
            return None
        
        records = []
        for entry in data:
            records.append({
                "timestamp": pd.to_datetime(int(entry.get("date")), unit="s"),
                "defillama_tvl": entry.get("tvl", 0),
            })
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        return _filter_date_range(df, start_date, end_date)
        
    except Exception as e:
        warnings.warn(f"Failed to fetch TVL for {chain}: {e}")
        return None


def _fetch_dex_volume(
    client: RestClient,
    chain: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[pd.DataFrame]:
    """Fetch historical DEX trading volume for a chain."""
    try:
        params = {
            "excludeTotalDataChart": "false",
            "excludeTotalDataChartBreakdown": "true",
        }
        
        data = client.get(f"/overview/dexs/{chain.lower()}", params)
        
        if not data:
            return None
        
        chart_data = data.get("totalDataChart", [])
        
        if not chart_data:
            return None
        
        records = []
        for entry in chart_data:
            if isinstance(entry, list) and len(entry) >= 2:
                records.append({
                    "timestamp": pd.to_datetime(int(entry[0]), unit="s"),
                    "defillama_dex_volume": entry[1],
                })
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        return _filter_date_range(df, start_date, end_date)
        
    except Exception as e:
        warnings.warn(f"Failed to fetch DEX volume for {chain}: {e}")
        return None


def _fetch_fees_revenue(
    client: RestClient,
    chain: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[pd.DataFrame]:
    """Fetch historical fees and revenue for a chain."""
    base_params = {
        "excludeTotalDataChart": "false",
        "excludeTotalDataChartBreakdown": "true",
    }
    
    fees_df = _fetch_chart_metric(
        client,
        f"/overview/fees/{chain.lower()}",
        {**base_params, "dataType": "dailyFees"},
        "defillama_fees",
    )
    
    revenue_df = _fetch_chart_metric(
        client,
        f"/overview/fees/{chain.lower()}",
        {**base_params, "dataType": "dailyRevenue"},
        "defillama_revenue",
    )
    
    if fees_df is None and revenue_df is None:
        return None
    
    if fees_df is not None and revenue_df is not None:
        df = pd.merge(fees_df, revenue_df, on="timestamp", how="outer")
    elif fees_df is not None:
        df = fees_df
    else:
        df = revenue_df
    
    return _filter_date_range(df, start_date, end_date)


def _fetch_chart_metric(
    client: RestClient,
    endpoint: str,
    params: dict,
    column_name: str,
) -> Optional[pd.DataFrame]:
    """Fetch a single chart metric from DefiLlama."""
    try:
        data = client.get(endpoint, params)
        
        if not data:
            return None
        
        chart = data.get("totalDataChart", [])
        
        if not chart:
            return None
        
        records = []
        for entry in chart:
            if isinstance(entry, list) and len(entry) >= 2:
                records.append({
                    "timestamp": pd.to_datetime(int(entry[0]), unit="s"),
                    column_name: entry[1],
                })
        
        return pd.DataFrame(records) if records else None
        
    except Exception:
        return None


def _filter_date_range(
    df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    """Filter DataFrame to date range."""
    if start_date:
        df = df[df["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.to_datetime(end_date)]
    
    return df.sort_values("timestamp").reset_index(drop=True)
