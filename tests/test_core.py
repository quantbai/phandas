"""Unit tests for phandas Factor class."""

import pytest
import pandas as pd
import numpy as np
from phandas import Factor


class TestFactorInit:
    """Tests for Factor initialization and data validation."""
    
    def test_init_from_dataframe(self, sample_factor_data):
        """Factor should initialize from DataFrame with correct columns."""
        factor = Factor(sample_factor_data, name='test')
        
        assert factor.name == 'test'
        assert list(factor.data.columns) == ['timestamp', 'symbol', 'factor']
        assert len(factor.data) == len(sample_factor_data)
    
    def test_init_auto_column_rename(self):
        """Factor should auto-rename columns if 3 columns present."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'ticker': ['BTC'] * 10,
            'value': np.random.randn(10)
        })
        factor = Factor(df)
        
        assert list(factor.data.columns) == ['timestamp', 'symbol', 'factor']
    
    def test_init_sorted_by_symbol_timestamp(self, sample_factor_data):
        """Factor data should be sorted by symbol then timestamp."""
        shuffled = sample_factor_data.sample(frac=1, random_state=42)
        factor = Factor(shuffled)
        
        for symbol in factor.data['symbol'].unique():
            symbol_data = factor.data[factor.data['symbol'] == symbol]
            assert symbol_data['timestamp'].is_monotonic_increasing
    
    def test_init_missing_factor_column_raises(self):
        """Factor should raise ValueError if no factor column found."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'symbol': ['BTC'] * 10
        })
        
        with pytest.raises(ValueError, match="No factor column found"):
            Factor(df)


class TestFactorTimeSeries:
    """Tests for time series operators."""
    
    def test_ts_mean(self, sample_factor):
        """ts_mean should compute rolling mean with correct window."""
        result = sample_factor.ts_mean(5)
        
        assert result.name == f'ts_mean({sample_factor.name},5)'
        for symbol in result.data['symbol'].unique():
            symbol_data = result.data[result.data['symbol'] == symbol]
            assert symbol_data['factor'].iloc[:4].isna().all()
            assert symbol_data['factor'].iloc[4:].notna().all()
    
    def test_ts_delay(self, sample_factor):
        """ts_delay should lag values by specified periods."""
        result = sample_factor.ts_delay(3)
        
        assert result.name == f'ts_delay({sample_factor.name},3)'
        for symbol in result.data['symbol'].unique():
            symbol_data = result.data[result.data['symbol'] == symbol]
            assert symbol_data['factor'].iloc[:3].isna().all()
    
    def test_ts_skewness(self, sample_factor):
        """ts_skewness should compute rolling skewness."""
        result = sample_factor.ts_skewness(20)
        
        assert result.name == f'ts_skewness({sample_factor.name},20)'
        for symbol in result.data['symbol'].unique():
            symbol_data = result.data[result.data['symbol'] == symbol]
            assert symbol_data['factor'].iloc[:19].isna().all()
    
    def test_ts_std_dev(self, sample_factor):
        """ts_std_dev should compute rolling standard deviation."""
        result = sample_factor.ts_std_dev(10)
        
        assert result.name == f'ts_std_dev({sample_factor.name},10)'
        assert result.data['factor'].iloc[9:].notna().any()
    
    def test_ts_rank(self, sample_factor):
        """ts_rank should compute rolling percentile rank."""
        result = sample_factor.ts_rank(10)
        
        assert result.name == f'ts_rank({sample_factor.name},10)'
        valid_values = result.data['factor'].dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 1).all()
    
    def test_invalid_window_raises(self, sample_factor):
        """Negative window should raise ValueError."""
        with pytest.raises(ValueError, match="Window must be positive"):
            sample_factor.ts_mean(-1)


class TestFactorCrossSection:
    """Tests for cross-sectional operators."""
    
    def test_rank(self, sample_factor):
        """rank should compute cross-sectional percentile rank."""
        result = sample_factor.rank()
        
        assert result.name == f'rank({sample_factor.name})'
        for ts in result.data['timestamp'].unique():
            ts_data = result.data[result.data['timestamp'] == ts]['factor']
            valid = ts_data.dropna()
            if len(valid) > 0:
                assert (valid >= 0).all()
                assert (valid <= 1).all()
    
    def test_zscore(self, sample_factor):
        """zscore should standardize cross-sectionally."""
        result = sample_factor.zscore()
        
        assert 'normalize' in result.name  # zscore uses normalize internally
        for ts in result.data['timestamp'].unique():
            ts_data = result.data[result.data['timestamp'] == ts]['factor']
            valid = ts_data.dropna()
            if len(valid) > 1:
                assert abs(valid.mean()) < 1e-10
                assert abs(valid.std() - 1) < 0.1
    
    def test_signal(self, sample_factor):
        """signal should produce dollar-neutral weights."""
        result = sample_factor.signal()
        
        for ts in result.data['timestamp'].unique():
            ts_data = result.data[result.data['timestamp'] == ts]['factor']
            valid = ts_data.dropna()
            if len(valid) > 0:
                long_sum = valid[valid > 0].sum()
                short_sum = valid[valid < 0].sum()
                if abs(long_sum) > 1e-6:
                    assert abs(long_sum - 0.5) < 0.1
                if abs(short_sum) > 1e-6:
                    assert abs(short_sum + 0.5) < 0.1


class TestFactorArithmetic:
    """Tests for arithmetic operations."""
    
    def test_add_scalar(self, sample_factor):
        """Adding scalar should work element-wise."""
        result = sample_factor + 10
        
        diff = result.data['factor'] - sample_factor.data['factor']
        assert (diff.dropna() == 10).all()
    
    def test_add_factor(self, sample_factor):
        """Adding Factor should align and sum."""
        result = sample_factor + sample_factor
        
        expected = sample_factor.data['factor'] * 2
        np.testing.assert_array_almost_equal(
            result.data['factor'].values,
            expected.values
        )
    
    def test_subtract(self, sample_factor):
        """Subtraction should work with Factor and scalar."""
        result = sample_factor - sample_factor
        
        assert (result.data['factor'].dropna() == 0).all()
    
    def test_multiply(self, sample_factor):
        """Multiplication should work element-wise."""
        result = sample_factor * 2
        
        expected = sample_factor.data['factor'] * 2
        np.testing.assert_array_almost_equal(
            result.data['factor'].values,
            expected.values
        )
    
    def test_divide(self, sample_factor):
        """Division should handle zero correctly."""
        result = sample_factor / sample_factor
        
        valid = result.data['factor'].dropna()
        assert (abs(valid - 1) < 1e-10).all()


class TestFactorTransform:
    """Tests for mathematical transforms."""
    
    def test_log(self, close_factor):
        """log should compute natural logarithm of positive values."""
        result = close_factor.log()
        
        assert result.name == f'log({close_factor.name})'
        assert result.data['factor'].notna().any()
    
    def test_sqrt(self, close_factor):
        """sqrt should compute square root of non-negative values."""
        result = close_factor.sqrt()
        
        squared = result * result
        np.testing.assert_array_almost_equal(
            squared.data['factor'].dropna().values,
            close_factor.data['factor'].dropna().values,
            decimal=5
        )
    
    def test_sign(self, sample_factor):
        """sign should return -1, 0, or 1."""
        result = sample_factor.sign()
        
        valid = result.data['factor'].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})
    
    def test_reverse(self, sample_factor):
        """reverse should negate values."""
        result = sample_factor.reverse()
        
        np.testing.assert_array_almost_equal(
            result.data['factor'].values,
            -sample_factor.data['factor'].values
        )


class TestFactorNeutralization:
    """Tests for factor neutralization."""
    
    def test_vector_neut(self, sample_factor, volume_factor):
        """vector_neut should remove projection onto another factor."""
        result = sample_factor.vector_neut(volume_factor)
        
        assert 'vector_neut' in result.name
    
    def test_regression_neut(self, sample_factor, volume_factor):
        """regression_neut should return OLS residuals."""
        result = sample_factor.regression_neut(volume_factor)
        
        assert 'regression_neut' in result.name

