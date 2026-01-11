"""Factor sub-package with modular operators using Mixin pattern.

This package provides a modular architecture for the Factor class,
splitting operators into logical categories for maintainability.

The Factor class is composed of multiple Mixins:
- FactorBase: Core data structure and utilities
- CrossSectionalMixin: Cross-sectional operators (rank, mean, zscore...)
- TimeSeriesMixin: Time-series operators (ts_rank, ts_mean, ts_kurtosis...)
- NeutralizationMixin: Neutralization operators (vector_neut, regression_neut)
- GroupMixin: Group operators (group_neutralize, group_rank...)
- ArithmeticMixin: Arithmetic and comparison operators (+, -, *, /, power...)
- IOMixin: Input/output and utility methods (to_df, show, info...)
- MetricsMixin: Factor metrics (crowding, ic, turnover, autocorr...)
"""

from .base import FactorBase
from .cross_sectional import CrossSectionalMixin
from .timeseries import TimeSeriesMixin
from .neutralization import NeutralizationMixin
from .group import GroupMixin
from .arithmetic import ArithmeticMixin
from .io import IOMixin
from .metrics import MetricsMixin


class Factor(
    FactorBase,
    CrossSectionalMixin,
    TimeSeriesMixin,
    NeutralizationMixin,
    GroupMixin,
    ArithmeticMixin,
    IOMixin,
    MetricsMixin
):
    """Factor matrix for quantitative analysis.
    
    Parameters
    ----------
    data : pd.DataFrame or str
        DataFrame with columns (timestamp, symbol, factor) or path to CSV
    name : str, optional
        Factor name for tracking in transformations
    
    Attributes
    ----------
    data : pd.DataFrame
        Sorted factor data (timestamp, symbol, factor)
    name : str
        Factor identifier
    """
    pass


__all__ = ['Factor', 'FactorBase']
