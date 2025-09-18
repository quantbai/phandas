"""
This module contains predefined quantitative factors for financial analysis.
"""
from ..core import Factor

def momentum(close: Factor, period: int = 14) -> Factor:
    """
    Calculates the Momentum (Momentum) factor.

    Momentum is the rate of acceleration of a security's price or volume.
    It identifies the strength of a price trend.

    Args:
        close (Factor): Factor of closing prices.
        period (int, optional): The lookback period for momentum calculation. Defaults to 14.

    Returns:
        Factor: A Factor representing the momentum factor.
    """
    return (close / close.ts_delay(period)) - 1
