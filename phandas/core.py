"""Core factor computation engine for alpha factor transformations via operators.

This module re-exports the Factor class from the modular factor subpackage
for backward compatibility. The actual implementation is now split across:
- factor/base.py: Core data structure and utilities
- factor/cross_sectional.py: Cross-sectional operators (rank, mean, zscore...)
- factor/timeseries.py: Time-series operators (ts_rank, ts_mean, ts_kurtosis...)
- factor/neutralization.py: Neutralization operators (vector_neut, regression_neut)
- factor/group.py: Group operators (group_neutralize, group_rank...)
- factor/arithmetic.py: Arithmetic and comparison operators (+, -, *, /, power...)
- factor/io.py: Input/output and utility methods (to_df, show, info...)
"""

from .factor import Factor

__all__ = ['Factor']