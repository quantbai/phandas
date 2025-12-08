"""Unit tests for phandas.operators functional API."""

import pytest
import pandas as pd
import numpy as np
from phandas import (
    Factor, Panel,
    ts_rank, ts_mean, ts_std_dev, ts_delay, ts_delta, ts_skewness, ts_corr,
    rank, zscore, signal, vector_neut,
    log, sqrt, sign, reverse, add, subtract, multiply, divide,
    group, group_neutralize, group_mean, group_median,
    group_rank, group_scale, group_zscore, group_normalize
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


class TestGroupOperators:
    """Tests for group-related operator functions."""
    
    def test_group_mapping_constants(self, close_factor):
        """Test mapping using predefined constant name."""
        g_factor = group(close_factor, 'SECTOR_L1_L2')
        
        assert isinstance(g_factor, Factor)
        
        df = g_factor.data
        eth_val = df[df['symbol'] == 'ETH']['factor'].iloc[0]
        arb_val = df[df['symbol'] == 'ARB']['factor'].iloc[0]
        
        assert eth_val == 1
        assert arb_val == 2

    def test_group_mapping_dict(self, close_factor):
        """Test mapping using custom dictionary."""
        mapping = {'BTC': 10, 'ETH': 20}
        g_factor = group(close_factor, mapping)
        
        df = g_factor.data
        btc_val = df[df['symbol'] == 'BTC']['factor'].iloc[0]
        eth_val = df[df['symbol'] == 'ETH']['factor'].iloc[0]
        sol_val = df[df['symbol'] == 'SOL']['factor'].iloc[0]
        
        assert btc_val == 10
        assert eth_val == 20
        assert np.isnan(sol_val)

    def test_group_neutralize_logic(self):
        """Verify mathematical correctness of group neutralization."""
        data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [10.0, 20.0, 30.0]
        })
        x = Factor(data, 'x')
        
        group_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [1, 1, 2]
        })
        g = Factor(group_data, 'g')
        
        neut = group_neutralize(x, g)
        res = neut.data.set_index('symbol')['factor']
        
        np.testing.assert_almost_equal(res['SymA'], -5.0)
        np.testing.assert_almost_equal(res['SymB'], 5.0)
        np.testing.assert_almost_equal(res['SymC'], 0.0)

    def test_group_mean_logic(self):
        """Verify group_mean calculation."""
        data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [10.0, 20.0, 30.0]
        })
        x = Factor(data, 'x')
        
        group_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [1, 1, 2]
        })
        g = Factor(group_data, 'g')
        
        gm = group_mean(x, g)
        res = gm.data.set_index('symbol')['factor']
        
        np.testing.assert_almost_equal(res['SymA'], 15.0)
        np.testing.assert_almost_equal(res['SymB'], 15.0)
        np.testing.assert_almost_equal(res['SymC'], 30.0)
        
    def test_group_median_logic(self):
        """Verify group_median calculation."""
        data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [10.0, 20.0, 500.0]
        })
        x = Factor(data, 'x')
        
        group_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [1, 1, 1]
        })
        g = Factor(group_data, 'g')
        
        gmed = group_median(x, g)
        res = gmed.data.iloc[0]['factor']
        
        np.testing.assert_almost_equal(res, 20.0)

    def test_group_rank_logic(self):
        """Verify group_rank calculation."""
        data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [10.0, 20.0, 50.0]
        })
        x = Factor(data, 'x')
        
        group_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [1, 1, 1]
        })
        g = Factor(group_data, 'g')
        
        gr = group_rank(x, g)
        res = gr.data.set_index('symbol')['factor']
        
        np.testing.assert_almost_equal(res['SymA'], 1/3)
        np.testing.assert_almost_equal(res['SymB'], 2/3)
        np.testing.assert_almost_equal(res['SymC'], 1.0)

    def test_group_scale_logic(self):
        """Verify group_scale calculation."""
        data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [10.0, 20.0, 50.0]
        })
        x = Factor(data, 'x')
        
        group_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [1, 1, 1]
        })
        g = Factor(group_data, 'g')
        
        gs = group_scale(x, g)
        res = gs.data.set_index('symbol')['factor']
        
        np.testing.assert_almost_equal(res['SymA'], 0.0)
        np.testing.assert_almost_equal(res['SymB'], 0.25)
        np.testing.assert_almost_equal(res['SymC'], 1.0)

    def test_group_zscore_logic(self):
        """Verify group_zscore calculation."""
        data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [10.0, 20.0, 30.0]
        })
        x = Factor(data, 'x')
        
        group_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [1, 1, 1]
        })
        g = Factor(group_data, 'g')
        
        gz = group_zscore(x, g)
        res = gz.data.set_index('symbol')['factor']
        
        np.testing.assert_almost_equal(res['SymA'], -1.0)
        np.testing.assert_almost_equal(res['SymB'], 0.0)
        np.testing.assert_almost_equal(res['SymC'], 1.0)

    def test_group_normalize_logic(self):
        """Verify group_normalize calculation."""
        data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [10.0, -20.0, 20.0]
        })
        x = Factor(data, 'x')
        
        group_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')] * 3,
            'symbol': ['SymA', 'SymB', 'SymC'],
            'factor': [1, 1, 1]
        })
        g = Factor(group_data, 'g')
        
        gn = group_normalize(x, g, scale=1.0)
        res = gn.data.set_index('symbol')['factor']
        
        np.testing.assert_almost_equal(res['SymA'], 0.2)
        np.testing.assert_almost_equal(res['SymB'], -0.4)
        np.testing.assert_almost_equal(res['SymC'], 0.4)
        np.testing.assert_almost_equal(res.abs().sum(), 1.0)


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

