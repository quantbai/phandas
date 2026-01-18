"""Data provider registry.

This module exports the SOURCE_REGISTRY mapping data source names to fetcher functions.
All fetchers follow the same interface:

    def fetch_xxx(
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        '''Fetch data for symbols.
        
        Returns DataFrame with columns: timestamp, symbol, <metric_columns>
        '''

To add a new data source:
1. Create a new module in providers/ (e.g., coinglass.py)
2. Implement fetch function following the interface above
3. Import and register in SOURCE_REGISTRY below
"""

from typing import Callable, Dict, List, Optional
import pandas as pd

from .binance import fetch_binance
from .okx import fetch_okx
from .defillama import fetch_defillama
from .coingecko import fetch_coingecko
from .base import TIMEFRAME_MAP, SYMBOL_RENAMES, SYMBOL_TO_CHAIN, SYMBOL_TO_STABLECOIN_CHAIN

FetcherType = Callable[
    [List[str], Optional[str], Optional[str]],
    Optional[pd.DataFrame]
]

SOURCE_REGISTRY: Dict[str, FetcherType] = {
    'binance': fetch_binance,
    'okx': fetch_okx,
    'defillama': fetch_defillama,
    'coingecko': fetch_coingecko,
}

__all__ = [
    'SOURCE_REGISTRY',
    'fetch_binance',
    'fetch_okx',
    'fetch_defillama',
    'fetch_coingecko',
]

