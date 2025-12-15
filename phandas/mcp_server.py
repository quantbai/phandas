"""
MCP (Model Context Protocol) server for phandas.

Provides a bridge for AI IDEs (Cursor, Claude Desktop) to access phandas
as a pip-installed Python module. This allows AI agents to fetch market data,
browse operators, read source code, and execute backtests without manual coding.

Available MCP Tools:
    fetch_market_data : Fetch cryptocurrency OHLCV data
    list_operators : List all available alpha factor operators
    read_source : Get source code of phandas functions
    execute_factor_backtest : Run factor backtest with custom Python code

Usage:
    Configure in Cursor/Claude Desktop MCP settings:
    {"command": "python", "args": ["-m", "phandas.mcp_server"]}
"""

from typing import List, Optional
from mcp.server.fastmcp import FastMCP
from .data import fetch_data
from .backtest import backtest
import pandas as pd
import json
import warnings

mcp = FastMCP("phandas")

@mcp.tool()
def fetch_market_data(
    symbols: List[str], 
    timeframe: str = '1d',
    limit: int = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None
) -> str:
    """
    Fetch cryptocurrency market data. Returns the latest data points by default.
    
    Args:
        symbols: List of trading pairs (e.g., ['BTC', 'ETH'])
        timeframe: Time interval (e.g., '1d', '1h', '15m')
        limit: Number of recent data points to return per symbol (default: 5)
        start_date: Start date (YYYY-MM-DD). If None, fetches recent data.
        end_date: End date (YYYY-MM-DD).
        sources: Data sources (default: ['binance'])
        
    Returns:
        JSON string containing a list of the latest market data records.
    """
    try:
        panel = fetch_data(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            sources=sources
        )
        
        df = panel.data
        
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        if 'symbol' in df.columns:
            latest_df = df.groupby('symbol').tail(limit)
        else:
            latest_df = df.tail(limit)
        
        records = latest_df.to_dict(orient='records')
        for record in records:
            for k, v in record.items():
                if isinstance(v, pd.Timestamp):
                    record[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                    
        return json.dumps(records, indent=2)
        
    except Exception as e:
        return f"Error fetching data: {str(e)}"

@mcp.tool()
def list_operators() -> str:
    """
    List all available alpha factor operators in phandas.
    Returns a JSON list containing function names, signatures, and docstrings.
    Use this to discover what mathematical and statistical operations are available.
    
    All operators are imported at the top level, use: from phandas import ts_mean, rank, etc.
    """
    import inspect
    from . import operators
    
    ops = []
    for name, func in inspect.getmembers(operators, inspect.isfunction):
        if name.startswith('_'):
            continue
        
        try:
            sig = str(inspect.signature(func))
            doc = inspect.getdoc(func) or ""
            ops.append({
                "name": name,
                "signature": f"{name}{sig}",
                "docstring": doc.split('\n')[0]
            })
        except Exception:
            continue
    
    return json.dumps(ops, indent=2)

@mcp.tool()
def read_source(object_path: str) -> str:
    """
    Get the source code of a specific Phandas function or class.
    
    Args:
        object_path: Dot-separated path to the object (e.g., 'ts_mean', 'Factor.ts_mean', 'phandas.core.Factor')
        All operators are top-level exports, so 'ts_mean' resolves to 'phandas.operators.ts_mean'
        
    Returns:
        The source code of the object.
    """
    import inspect
    import importlib
    
    try:
        if '.' not in object_path:
            object_path = f"phandas.operators.{object_path}"
        
        module_name, obj_name = object_path.rsplit('.', 1)
        
        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, obj_name)
        except (ImportError, AttributeError):
            if '.' in module_name:
                mod_name, class_name = module_name.rsplit('.', 1)
                module = importlib.import_module(mod_name)
                cls = getattr(module, class_name)
                obj = getattr(cls, obj_name)
            else:
                raise
        
        source = inspect.getsource(obj)
        return f"Source code for {object_path}:\n\n{source}"
    
    except Exception as e:
        return f"Error reading source for {object_path}: {str(e)}"

@mcp.tool()
def execute_factor_backtest(
    factor_code: str,
    symbols: List[str] = None,
    start_date: str = '2022-01-01',
    transaction_cost: float = 0.0003,
    full_rebalance: bool = False
) -> str:
    """
    Execute factor backtest with custom Python code.
    
    Args:
        factor_code: Python code to calculate factor. 
                    - Pre-defined: close, open, high, low, volume
                    - Operators: ts_rank(), ts_mean(), ts_skewness(), ts_delay(), 
                                log(), rank(), vector_neut(), etc.
                    - Must assign result to variable named 'factor'
        symbols: List of trading symbols (default: ['ETH','SOL','ARB','OP','POL','SUI'])
        start_date: Start date in YYYY-MM-DD format (default: 2022-01-01)
        transaction_cost: Transaction cost rate as decimal (default: 0.0003 = 0.03%)
        full_rebalance: Whether to fully rebalance portfolio each period (default: False)
        
    Returns:
        JSON string with backtest results containing:
        - status: 'success' or 'error'
        - summary: Performance metrics (total_return, annual_return, sharpe_ratio, max_drawdown)
        - factor_expression: Complete factor expression (one-line, including intermediate variables)
        - error: Error message if status is 'error'
    
    Examples:
        factor_code = '''
log_returns = log(close) - ts_delay(log(close), 20)
momentum = log_returns.rank()
alpha = vector_neut(momentum, -rank(volume))
        '''
    """
    try:
        if symbols is None:
            symbols = ['ETH', 'SOL', 'ARB', 'OP', 'POL', 'SUI']
        
        panel = fetch_data(symbols=symbols, start_date=start_date, sources=['binance'])
        
        import phandas
        namespace = {
            'close': panel['close'],
            'open': panel['open'],
            'high': panel['high'],
            'low': panel['low'],
            'volume': panel['volume'],
            **{name: getattr(phandas, name) for name in phandas.__all__ if not name[0].isupper()}
        }
        
        exec(factor_code, namespace)
        
        if 'alpha' not in namespace:
            return json.dumps({
                'status': 'error',
                'summary': {},
                'factor_expression': None,
                'error': "Factor code must assign result to variable named 'alpha'"
            })
        
        bt_results = backtest(
            entry_price_factor=panel['open'],
            strategy_factor=namespace['alpha'],
            transaction_cost=(transaction_cost, transaction_cost),
            full_rebalance=full_rebalance,
            auto_run=True
        )
        
        summary = bt_results.metrics
        key_metrics = {
            'total_return': summary.get('total_return', 0),
            'annual_return': summary.get('annual_return', 0),
            'sharpe_ratio': summary.get('sharpe_ratio', 0),
            'max_drawdown': summary.get('max_drawdown', 0),
        }
        
        factor_expr = namespace['alpha'].name if hasattr(namespace['alpha'], 'name') else 'alpha'
        
        result = {
            'status': 'success',
            'summary': key_metrics,
            'factor_expression': factor_expr,
            'error': None
        }
        
        return json.dumps(result, default=str)
    
    except Exception as e:
        warnings.warn(f"Backtest execution failed: {e}")
        return json.dumps({
            'status': 'error',
            'summary': {},
            'factor_expression': None,
            'error': str(e)
        })

def main():
    """Entry point for the MCP server."""
    mcp.run()

if __name__ == "__main__":
    main()
