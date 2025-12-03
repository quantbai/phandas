"""Shared pytest fixtures for phandas tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_dates():
    """Generate 100 consecutive dates starting from 2024-01-01."""
    return pd.date_range('2024-01-01', periods=100, freq='D')


@pytest.fixture
def sample_symbols():
    """Standard test symbols matching real usage patterns."""
    return ['BTC', 'ETH', 'SOL', 'ARB', 'OP', 'POL']


@pytest.fixture
def sample_factor_data(sample_dates, sample_symbols):
    """Create sample factor DataFrame with realistic structure.
    
    Returns DataFrame with columns: timestamp, symbol, factor
    100 dates x 6 symbols = 600 rows
    """
    n_dates = len(sample_dates)
    n_symbols = len(sample_symbols)
    
    data = []
    np.random.seed(42)
    
    for symbol in sample_symbols:
        base_value = np.random.randn()
        values = base_value + np.cumsum(np.random.randn(n_dates) * 0.1)
        
        for i, date in enumerate(sample_dates):
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'factor': values[i]
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_panel_data(sample_dates, sample_symbols):
    """Create sample OHLCV Panel data with realistic price structure.
    
    Returns DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    n_dates = len(sample_dates)
    data = []
    np.random.seed(42)
    
    base_prices = {'BTC': 40000, 'ETH': 2000, 'SOL': 100, 'ARB': 1.5, 'OP': 2.0, 'POL': 0.8}
    
    for symbol in sample_symbols:
        base = base_prices.get(symbol, 100)
        price = base
        
        for date in sample_dates:
            ret = np.random.randn() * 0.03
            price = price * (1 + ret)
            
            high = price * (1 + abs(np.random.randn()) * 0.01)
            low = price * (1 - abs(np.random.randn()) * 0.01)
            open_price = low + (high - low) * np.random.random()
            volume = base * 1000 * (1 + np.random.randn() * 0.3)
            
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': max(volume, 0)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_factor(sample_factor_data):
    """Create Factor instance from sample data."""
    from phandas import Factor
    return Factor(sample_factor_data, name='test_factor')


@pytest.fixture
def sample_panel(sample_panel_data):
    """Create Panel instance from sample OHLCV data."""
    from phandas import Panel
    return Panel(sample_panel_data)


@pytest.fixture
def close_factor(sample_panel):
    """Extract close price as Factor from Panel."""
    return sample_panel['close']


@pytest.fixture
def volume_factor(sample_panel):
    """Extract volume as Factor from Panel."""
    return sample_panel['volume']

