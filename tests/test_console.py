"""Tests for console output module."""

import pytest
import warnings


class TestConsoleImports:
    def test_print_import(self):
        from phandas.console import print
        assert callable(print)
    
    def test_console_import(self):
        from phandas.console import console
        from rich.console import Console
        assert isinstance(console, Console)
    
    def test_table_import(self):
        from phandas.console import Table
        from rich.table import Table as RichTable
        assert Table is RichTable


class TestWarningsUsage:
    def test_analysis_correlation_warning(self):
        from phandas import Factor
        import pandas as pd
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'symbol': ['BTC'] * 10,
            'factor': range(10),
        })
        factor = Factor(df)
        
        from phandas.analysis import FactorAnalyzer
        analyzer = FactorAnalyzer([factor], factor)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyzer.correlation()
            assert len(w) == 1
            assert "at least 2 factors" in str(w[0].message)
    
    def test_plot_no_data_warning(self):
        from phandas import Factor
        from phandas.plot import FactorPlotter
        import pandas as pd
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'symbol': ['BTC'] * 10,
            'factor': range(10),
        })
        factor = Factor(df)
        plotter = FactorPlotter(factor)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plotter._plot_single_symbol('INVALID_SYMBOL', (12, 5), None)
            assert len(w) == 1
            assert "No data found" in str(w[0].message)


class TestRichTableOutput:
    def test_table_creation(self):
        from phandas.console import Table
        
        table = Table(title="Test")
        table.add_column("Col1")
        table.add_column("Col2")
        table.add_row("a", "b")
        
        assert table.row_count == 1
    
    def test_console_print_no_error(self):
        from phandas.console import console
        from io import StringIO
        
        console.print("Test message", highlight=False)
