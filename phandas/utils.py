"""
Utility functions for unified Factor-based analysis.

Provides helper functions for:
- Saving and loading Factor objects to/from disk
- Factor information and statistics display
- Factor validation and quality checks
"""

from .core import Factor


# ==================== Factor Persistence ====================

def save_factor(factor: Factor, path: str) -> str:
    """Save factor to CSV file."""
    return factor.to_csv(path)


def load_saved_factor(csv_path: str, name: str = None) -> Factor:
    """Load saved factor from CSV."""
    import os
    if name is None:
        name = os.path.splitext(os.path.basename(csv_path))[0]
    return Factor(csv_path, name)


# ==================== Factor Information ====================

def factor_info(factor: Factor, verbose: bool = True) -> dict:
    """
    Get comprehensive factor information.
    
    Parameters
    ----------
    factor : Factor
        Factor to analyze
    verbose : bool, default True
        Print information to console
        
    Returns
    -------
    dict
        Factor information dictionary
    """
    info = factor.info()
    
    if verbose:
        print(f"Factor: {info['name']}")
        print(f"Shape: {info['shape']}")
        print(f"Symbols: {len(info['symbols'])} ({', '.join(info['symbols'])})")
        print(f"Time Range: {info['time_range'][0]} to {info['time_range'][1]}")
        print(f"Valid Data: {info['valid_ratio']:.1%}")
        
        # Additional statistics
        values = factor.data['factor'].dropna()
        if len(values) > 0:
            print(f"Statistics:")
            print(f"  Mean: {values.mean():.4f}")
            print(f"  Std:  {values.std():.4f}")
            print(f"  Min:  {values.min():.4f}")
            print(f"  Max:  {values.max():.4f}")
    
    return info

