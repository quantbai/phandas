"""Unit tests for phandas.operators functional API."""

import pytest
import pandas as pd
import numpy as np
from phandas import (
    Factor, Panel,
    ts_rank, ts_mean, ts_std_dev, ts_delay, ts_delta, ts_skewness, ts_corr,
    rank, zscore, signal, vector_neut,
    log, sqrt, sign, reverse, add, subtract, multiply, divide
)


class TestTimeSeriesOperators:
    """Tests for time series operator functions."""
    
    def test_ts_mean_function(self, close_factor):
        """ts_mean function should match Factor method."""
        result = ts_mean(close_factor, 10)
        expected = close_factor.ts_mean(10)
        
        pd.testing.assert_frame_equal(result.data, expected.data)
    
    def test_ts_delay_function(self, close_factor):
        """ts_delay function should match Factor method."""
        result = ts_delay(close_factor, 5)
        expected = close_factor.ts_delay(5)
        
        pd.testing.assert_frame_equal(result.data, expected.data)
    
    def test_ts_skewness_function(self, close_factor):
        """ts_skewness function should match Factor method."""
        result = ts_skewness(close_factor, 20)
        expected = close_factor.ts_skewness(20)
        
        pd.testing.assert_frame_equal(result.data, expected.data)
    
    def test_ts_corr_function(self, close_factor, volume_factor):
        """ts_corr function should compute rolling correlation."""
        result = ts_corr(close_factor, volume_factor, 20)
        
        assert 'ts_corr' in result.name
        valid = result.data['factor'].dropna()
        assert (valid >= -1).all() and (valid <= 1).all()


class TestCrossSectionalOperators:
    """Tests for cross-sectional operator functions."""
    
    def test_rank_function(self, sample_factor):
        """rank function should match Factor method."""
        result = rank(sample_factor)
        expected = sample_factor.rank()
        
        pd.testing.assert_frame_equal(result.data, expected.data)
    
    def test_zscore_function(self, sample_factor):
        """zscore function should match Factor method."""
        result = zscore(sample_factor)
        expected = sample_factor.zscore()
        
        pd.testing.assert_frame_equal(result.data, expected.data)
    
    def test_signal_function(self, sample_factor):
        """signal function should produce dollar-neutral weights."""
        result = signal(sample_factor)
        
        for ts in result.data['timestamp'].unique():
            ts_data = result.data[result.data['timestamp'] == ts]['factor']
            valid = ts_data.dropna()
            if len(valid) > 0:
                assert abs(valid.sum()) < 1e-6


class TestMathOperators:
    """Tests for mathematical operator functions."""
    
    def test_log_function(self, close_factor):
        """log function should compute natural logarithm."""
        result = log(close_factor)
        
        assert 'log' in result.name
        assert result.data['factor'].notna().any()
    
    def test_sqrt_function(self, close_factor):
        """sqrt function should compute square root."""
        result = sqrt(close_factor)
        
        assert 'sqrt' in result.name
    
    def test_sign_function(self, sample_factor):
        """sign function should return sign of values."""
        result = sign(sample_factor)
        
        valid = result.data['factor'].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})
    
    def test_reverse_function(self, sample_factor):
        """reverse function should negate values."""
        result = reverse(sample_factor)
        
        np.testing.assert_array_almost_equal(
            result.data['factor'].values,
            -sample_factor.data['factor'].values
        )


class TestArithmeticOperators:
    """Tests for arithmetic operator functions."""
    
    def test_add_function(self, sample_factor):
        """add function should add two factors."""
        result = add(sample_factor, sample_factor)
        
        expected = sample_factor.data['factor'] * 2
        np.testing.assert_array_almost_equal(
            result.data['factor'].values,
            expected.values
        )
    
    def test_subtract_function(self, sample_factor):
        """subtract function should subtract factors."""
        result = subtract(sample_factor, sample_factor)
        
        assert (result.data['factor'].dropna() == 0).all()
    
    def test_multiply_function(self, sample_factor):
        """multiply function should multiply factors."""
        result = multiply(sample_factor, sample_factor)
        
        expected = sample_factor.data['factor'] ** 2
        np.testing.assert_array_almost_equal(
            result.data['factor'].values,
            expected.values
        )
    
    def test_divide_function(self, sample_factor):
        """divide function should divide factors."""
        result = divide(sample_factor, sample_factor)
        
        valid = result.data['factor'].dropna()
        assert (abs(valid - 1) < 1e-10).all()


class TestNeutralizationOperators:
    """Tests for neutralization operator functions."""
    
    def test_vector_neut_function(self, sample_factor, volume_factor):
        """vector_neut function should orthogonalize factors."""
        result = vector_neut(sample_factor, volume_factor)
        
        assert 'vector_neut' in result.name


class TestRealWorldUsage:
    """Tests based on real usage patterns from okx/skewness.py."""
    
    def test_skewness_factor_pipeline(self, sample_panel):
        """Test complete skewness factor pipeline."""
        close = sample_panel['close']
        volume = sample_panel['volume']
        
        log_returns = log(close) - ts_delay(log(close), 1)
        skewness = ts_skewness(log_returns, 20).rank()
        skewness = vector_neut(skewness, -rank(volume))
        
        assert skewness.data['factor'].notna().any()
        assert 'vector_neut' in skewness.name
    
    def test_factor_chain_operations(self, sample_panel):
        """Test chained factor operations."""
        close = sample_panel['close']
        
        result = close.ts_mean(10).rank().zscore()
        
        assert result.data['factor'].notna().any()
        for ts in result.data['timestamp'].unique():
            ts_data = result.data[result.data['timestamp'] == ts]['factor']
            valid = ts_data.dropna()
            if len(valid) > 1:
                assert abs(valid.mean()) < 1e-10

