"""Unit tests for phandas Backtester."""

import pytest
import pandas as pd
import numpy as np
from phandas import Panel, Factor, backtest, Backtester


class TestBacktester:
    """Tests for Backtester class."""
    
    def test_init(self, sample_panel, sample_factor):
        """Backtester should initialize with valid inputs."""
        open_factor = sample_panel['open']
        
        bt = Backtester(
            entry_price_factor=open_factor,
            strategy_factor=sample_factor
        )
        
        assert bt is not None
    
    def test_run_basic(self, sample_panel, sample_factor):
        """Backtester.run should execute without errors."""
        open_factor = sample_panel['open']
        
        bt = Backtester(
            entry_price_factor=open_factor,
            strategy_factor=sample_factor,
            transaction_cost=(0.0003, 0.0003)
        )
        bt.run()
        
        assert bt.portfolio is not None
    
    def test_metrics_calculation(self, sample_panel, sample_factor):
        """Backtester should calculate performance metrics after run."""
        open_factor = sample_panel['open']
        
        bt = Backtester(
            entry_price_factor=open_factor,
            strategy_factor=sample_factor
        )
        bt.run().calculate_metrics()
        
        assert bt.metrics is not None
        assert 'total_return' in bt.metrics
        assert 'sharpe_ratio' in bt.metrics
        assert 'max_drawdown' in bt.metrics


class TestBacktestFunction:
    """Tests for backtest convenience function."""
    
    def test_backtest_function(self, sample_panel, sample_factor):
        """backtest function should return configured Backtester."""
        open_factor = sample_panel['open']
        
        result = backtest(
            entry_price_factor=open_factor,
            strategy_factor=sample_factor,
            transaction_cost=(0.0003, 0.0003)
        )
        
        assert isinstance(result, Backtester)
        assert result.metrics is not None
    
    def test_backtest_with_full_rebalance(self, sample_panel, sample_factor):
        """backtest should handle full_rebalance option."""
        open_factor = sample_panel['open']
        
        result = backtest(
            entry_price_factor=open_factor,
            strategy_factor=sample_factor,
            full_rebalance=True
        )
        
        assert result is not None


class TestBacktestMetrics:
    """Tests for backtest performance metrics."""
    
    def test_total_return_range(self, sample_panel, sample_factor):
        """Total return should be reasonable value."""
        open_factor = sample_panel['open']
        
        result = backtest(
            entry_price_factor=open_factor,
            strategy_factor=sample_factor
        )
        
        assert result.metrics['total_return'] > -1.0
    
    def test_sharpe_ratio_exists(self, sample_panel, sample_factor):
        """Sharpe ratio should be calculated."""
        open_factor = sample_panel['open']
        
        result = backtest(
            entry_price_factor=open_factor,
            strategy_factor=sample_factor
        )
        
        assert 'sharpe_ratio' in result.metrics
        assert not np.isnan(result.metrics['sharpe_ratio'])
    
    def test_max_drawdown_negative(self, sample_panel, sample_factor):
        """Max drawdown should be non-positive."""
        open_factor = sample_panel['open']
        
        result = backtest(
            entry_price_factor=open_factor,
            strategy_factor=sample_factor
        )
        
        assert result.metrics['max_drawdown'] <= 0


class TestRealWorldBacktest:
    """Tests based on real usage patterns."""
    
    def test_skewness_strategy_backtest(self, sample_panel):
        """Test backtest with skewness-based strategy."""
        from phandas import log, ts_delay, ts_skewness, rank, vector_neut
        
        close = sample_panel['close']
        volume = sample_panel['volume']
        open_price = sample_panel['open']
        
        log_returns = log(close) - ts_delay(log(close), 1)
        skewness = ts_skewness(log_returns, 20).rank()
        alpha = vector_neut(skewness, -rank(volume))
        
        result = backtest(
            entry_price_factor=open_price,
            strategy_factor=alpha,
            transaction_cost=(0.0003, 0.0003),
            full_rebalance=False
        )
        
        assert result.metrics is not None
        assert 'total_return' in result.metrics

