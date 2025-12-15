"""Unit tests for phandas FactorAnalyzer."""

import pytest
import pandas as pd
import numpy as np
from phandas import Factor, analyze, FactorAnalyzer


class TestFactorAnalyzer:
    """Test FactorAnalyzer class."""
    
    def test_analyze_creates_analyzer(self, sample_factor_data):
        """Test analyze() convenience function."""
        factor1 = Factor(sample_factor_data, "alpha1")
        
        factor2_data = sample_factor_data.copy()
        factor2_data['factor'] = factor2_data['factor'] * 2
        factor2 = Factor(factor2_data, "alpha2")
        
        price_data = sample_factor_data.copy()
        price_data['factor'] = 100 + np.random.randn(len(price_data)) * 10
        price = Factor(price_data, "close")
        
        result = analyze([factor1, factor2], price)
        
        assert isinstance(result, FactorAnalyzer)
        assert len(result.factors) == 2
        assert result.horizons == [1, 7, 30]
    
    def test_analyze_single_factor(self, sample_factor_data):
        """Test analyze() with single factor."""
        factor = Factor(sample_factor_data, "test")
        price = Factor(sample_factor_data.copy(), "price")
        
        result = analyze(factor, price)
        
        assert len(result.factors) == 1
    
    def test_correlation_returns_dataframe(self, sample_factor_data):
        """Test correlation() returns proper DataFrame."""
        factor1 = Factor(sample_factor_data, "alpha1")
        
        factor2_data = sample_factor_data.copy()
        factor2_data['factor'] = factor2_data['factor'] * 2
        factor2 = Factor(factor2_data, "alpha2")
        
        price = Factor(sample_factor_data.copy(), "price")
        analyzer = analyze([factor1, factor2], price)
        
        corr = analyzer.correlation()
        
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (2, 2)
    
    def test_ic_returns_dict(self, sample_factor_data):
        """Test ic() returns proper dict structure."""
        factor = Factor(sample_factor_data, "alpha1")
        price = Factor(sample_factor_data.copy(), "price")
        
        analyzer = analyze(factor, price, horizons=[1])
        ic = analyzer.ic()
        
        assert isinstance(ic, dict)
        assert "alpha1" in ic
        assert 1 in ic["alpha1"]
        assert "ic_mean" in ic["alpha1"][1]
    
    def test_stats_returns_dict(self, sample_factor_data):
        """Test stats() returns proper dict structure."""
        factor = Factor(sample_factor_data, "alpha1")
        price = Factor(sample_factor_data.copy(), "price")
        
        analyzer = analyze(factor, price)
        stats = analyzer.stats()
        
        assert isinstance(stats, dict)
        assert "alpha1" in stats
        assert "coverage" in stats["alpha1"]
        assert "turnover" in stats["alpha1"]
    
    def test_print_summary_returns_self(self, sample_factor_data):
        """Test print_summary() returns self for chaining."""
        factor = Factor(sample_factor_data, "alpha1")
        price = Factor(sample_factor_data.copy(), "price")
        
        analyzer = analyze(factor, price, horizons=[1])
        result = analyzer.print_summary()
        
        assert result is analyzer
    
    def test_empty_factors_raises(self, sample_factor_data):
        """Test empty factors list raises error."""
        price = Factor(sample_factor_data.copy(), "price")
        
        with pytest.raises(ValueError):
            analyze([], price)
    
    def test_custom_horizons(self, sample_factor_data):
        """Test custom horizons parameter."""
        factor = Factor(sample_factor_data, "test")
        price = Factor(sample_factor_data.copy(), "price")
        
        analyzer = analyze(factor, price, horizons=[1, 3, 5])
        
        assert analyzer.horizons == [1, 3, 5]
