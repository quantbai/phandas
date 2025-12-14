"""Unit tests for phandas data module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


class TestDataProcessing:
    """Tests for data processing functions."""
    
    def test_panel_from_csv(self, sample_panel, tmp_path):
        """Panel should load correctly from CSV."""
        from phandas import Panel
        
        csv_path = tmp_path / 'test_data.csv'
        sample_panel.to_csv(str(csv_path))
        
        loaded = Panel.from_csv(str(csv_path))
        
        assert loaded.data.shape == sample_panel.data.shape
    
    def test_factor_from_csv(self, sample_factor, tmp_path):
        """Factor should load correctly from CSV."""
        from phandas import Factor
        
        csv_path = tmp_path / 'test_factor.csv'
        sample_factor.to_csv(str(csv_path))
        
        loaded = Factor(str(csv_path))
        
        assert len(loaded.data) == len(sample_factor.data)


class TestSymbolRenames:
    """Tests for symbol rename handling."""
    
    def test_symbol_renames_defined(self):
        """SYMBOL_RENAMES should be defined in constants."""
        from phandas.constants import SYMBOL_RENAMES
        
        assert isinstance(SYMBOL_RENAMES, dict)
        assert 'POL' in SYMBOL_RENAMES


class TestTimeframeMapping:
    """Tests for timeframe mapping."""
    
    def test_timeframe_map_exists(self):
        """TIMEFRAME_MAP should contain standard timeframes."""
        from phandas.data import TIMEFRAME_MAP
        
        assert '1d' in TIMEFRAME_MAP
        assert '1h' in TIMEFRAME_MAP
        assert '1m' in TIMEFRAME_MAP

