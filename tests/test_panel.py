"""Unit tests for phandas Panel class."""

import pytest
import pandas as pd
import numpy as np
from phandas import Panel, Factor


class TestPanelInit:
    """Tests for Panel initialization."""
    
    def test_init_from_flat_dataframe(self, sample_panel_data):
        """Panel should initialize from flat DataFrame with timestamp/symbol columns."""
        panel = Panel(sample_panel_data)
        
        assert 'timestamp' in panel.data.columns
        assert 'symbol' in panel.data.columns
        assert 'close' in panel.data.columns
    
    def test_init_from_multiindex(self, sample_panel_data):
        """Panel should accept MultiIndex DataFrame and flatten it."""
        df = sample_panel_data.set_index(['timestamp', 'symbol'])
        panel = Panel(df)
        
        assert 'timestamp' in panel.data.columns
        assert 'symbol' in panel.data.columns
    
    def test_init_missing_columns_raises(self):
        """Panel should raise ValueError if timestamp/symbol missing."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="timestamp.*symbol"):
            Panel(df)


class TestPanelFromCSV:
    """Tests for Panel.from_csv class method."""
    
    def test_from_csv_roundtrip(self, sample_panel, tmp_path):
        """Panel should round-trip through CSV correctly."""
        csv_path = tmp_path / 'test_panel.csv'
        sample_panel.to_csv(str(csv_path))
        
        loaded = Panel.from_csv(str(csv_path))
        
        assert len(loaded.data) == len(sample_panel.data)
        assert set(loaded.columns) == set(sample_panel.columns)
    
    def test_from_df(self, sample_panel_data):
        """Panel.from_df should work as constructor alias."""
        panel = Panel.from_df(sample_panel_data)
        
        assert isinstance(panel, Panel)
        assert 'close' in panel.columns


class TestPanelAccess:
    """Tests for Panel column extraction."""
    
    def test_getitem_string(self, sample_panel):
        """Indexing with string should return Factor."""
        close = sample_panel['close']
        
        assert isinstance(close, Factor)
        assert close.name == 'close'
    
    def test_getitem_list(self, sample_panel):
        """Indexing with list should return Panel subset."""
        subset = sample_panel[['open', 'close']]
        
        assert isinstance(subset, Panel)
        assert set(subset.columns) == {'open', 'close'}
    
    def test_missing_column_raises(self, sample_panel):
        """Accessing non-existent column should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            sample_panel['nonexistent']
    
    def test_to_df(self, sample_panel):
        """to_df should return DataFrame copy."""
        df = sample_panel.to_df()
        
        assert isinstance(df, pd.DataFrame)
        assert 'close' in df.columns


class TestPanelSlice:
    """Tests for Panel slicing operations."""
    
    def test_slice_time(self, sample_panel):
        """slice_time should filter by date range."""
        result = sample_panel.slice_time(start='2024-01-10', end='2024-01-20')
        
        assert result.data['timestamp'].min() >= pd.Timestamp('2024-01-10')
        assert result.data['timestamp'].max() <= pd.Timestamp('2024-01-20')
    
    def test_slice_symbols(self, sample_panel):
        """slice_symbols should filter by symbol list."""
        result = sample_panel.slice_symbols(['BTC', 'ETH'])
        
        assert set(result.symbols) == {'BTC', 'ETH'}
    
    def test_slice_single_symbol(self, sample_panel):
        """slice_symbols should accept single string."""
        result = sample_panel.slice_symbols('BTC')
        
        assert result.symbols == ['BTC']


class TestPanelProperties:
    """Tests for Panel properties."""
    
    def test_columns_property(self, sample_panel):
        """columns property should exclude timestamp and symbol."""
        cols = sample_panel.columns
        
        assert 'timestamp' not in cols
        assert 'symbol' not in cols
        assert 'close' in cols
    
    def test_symbols_property(self, sample_panel):
        """symbols property should return unique symbols."""
        symbols = sample_panel.symbols
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
    
    def test_timestamps_property(self, sample_panel):
        """timestamps property should return DatetimeIndex."""
        ts = sample_panel.timestamps
        
        assert isinstance(ts, pd.DatetimeIndex)
    
    def test_len(self, sample_panel):
        """len() should return number of rows."""
        assert len(sample_panel) == len(sample_panel.data)


class TestPanelRepr:
    """Tests for Panel string representations."""
    
    def test_repr(self, sample_panel):
        """__repr__ should include key statistics."""
        repr_str = repr(sample_panel)
        
        assert 'Panel' in repr_str
        assert 'rows' in repr_str
        assert 'symbols' in repr_str
