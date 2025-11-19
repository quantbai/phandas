from typing import List, Optional
from mcp.server.fastmcp import FastMCP
from .data import fetch_data
import pandas as pd

# Initialize FastMCP server
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
        # If no start_date provided, we still need to fetch enough data to get the 'limit'
        # For simplicity, we let fetch_data handle defaults (usually fetches recent if start_date is None)
        # or we rely on the user providing it if they want specific history.
        # If start_date is None, fetch_data might default to a specific range or error depending on implementation.
        # Assuming fetch_data handles None gracefully or we might need to set a default start_date.
        
        panel = fetch_data(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            sources=sources
        )
        
        df = panel.data
        
        # Sort by timestamp to ensure we get the latest
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            
        # Group by symbol and take the last 'limit' rows
        if 'symbol' in df.columns:
            latest_df = df.groupby('symbol').tail(limit)
        else:
            latest_df = df.tail(limit)
            
        # Convert to list of dicts (records) for clean JSON output
        records = latest_df.to_dict(orient='records')
        
        # Format timestamp to string if needed (to_dict might handle it, but let's be safe for JSON)
        for record in records:
            for k, v in record.items():
                if isinstance(v, pd.Timestamp):
                    record[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                    
        import json
        return json.dumps(records, indent=2)
        
    except Exception as e:
        return f"Error fetching data: {str(e)}"

@mcp.tool()
def list_operators() -> str:
    """
    List all available alpha factor operators in phandas.operators.
    Returns a JSON list containing function names, signatures, and docstrings.
    Use this to discover what mathematical and statistical operations are available.
    """
    import inspect
    import json
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
                "docstring": doc.split('\n')[0] # First line only for brevity
            })
        except Exception:
            continue
            
    return json.dumps(ops, indent=2)

@mcp.tool()
def read_source(object_path: str) -> str:
    """
    Get the source code of a specific Phandas function or class.
    
    Args:
        object_path: Dot-separated path to the object (e.g., 'phandas.operators.ts_mean', 'phandas.core.Factor.ts_mean')
        
    Returns:
        The source code of the object.
    """
    import inspect
    import importlib
    
    try:
        # Handle simple names by defaulting to phandas.operators if not specified
        if '.' not in object_path:
            object_path = f"phandas.operators.{object_path}"
            
        module_name, obj_name = object_path.rsplit('.', 1)
        
        # If it looks like a class method (e.g. phandas.core.Factor.mean), split again
        # This is a simple heuristic; for robust import we might need more logic,
        # but let's try importing the module first.
        
        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, obj_name)
        except (ImportError, AttributeError):
            # Maybe it's a class method: module.Class.method
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

def main():
    """Entry point for the MCP server."""
    mcp.run()

if __name__ == "__main__":
    main()
