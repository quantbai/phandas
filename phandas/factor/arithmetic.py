"""Arithmetic and comparison operators for Factor."""

import numpy as np
import pandas as pd
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import FactorBase


class ArithmeticMixin:
    """Mixin providing arithmetic and comparison operators.
    
    Includes mathematical functions (log, sqrt, power) and
    Python operator overloads (+, -, *, /, **, <, >, etc.).
    """

    def abs(self: 'FactorBase') -> 'FactorBase':
        """Absolute value of factor."""
        from . import Factor
        
        result = self.data.copy()
        result['factor'] = np.abs(result['factor'])
        return Factor(result, f"abs({self.name})")

    def sign(self: 'FactorBase') -> 'FactorBase':
        """Sign of factor (-1, 0, or 1)."""
        from . import Factor
        
        result = self.data.copy()
        result['factor'] = np.sign(result['factor'])
        return Factor(result, f"sign({self.name})")

    def inverse(self: 'FactorBase') -> 'FactorBase':
        """Multiplicative inverse (1/x), NaN for zero."""
        from . import Factor
        
        result = self.data.copy()
        result['factor'] = np.where(
            result['factor'] != 0,
            1 / result['factor'],
            np.nan
        )
        return Factor(result, f"inverse({self.name})")

    def log(self: 'FactorBase', base: Optional[float] = None) -> 'FactorBase':
        """Logarithm of factor.
        
        Parameters
        ----------
        base : float, optional
            Log base. If None, natural log is used.
        
        Returns
        -------
        Factor
            Log-transformed factor, NaN for non-positive values
        """
        from . import Factor
        
        result = self.data.copy()
        
        if base is None:
            result['factor'] = np.where(
                result['factor'] > 0,
                np.log(result['factor']),
                np.nan
            )
            return Factor(result, f"log({self.name})")
        else:
            if base <= 0 or base == 1:
                raise ValueError(f"Invalid log base: {base}. Base must be positive and not equal to 1.")
            
            result['factor'] = np.where(
                result['factor'] > 0,
                np.log(result['factor']) / np.log(base),
                np.nan
            )
            return Factor(result, f"log({self.name},base={base})")
    
    def ln(self: 'FactorBase') -> 'FactorBase':
        """Natural logarithm (alias for log())."""
        return self.log()

    def sqrt(self: 'FactorBase') -> 'FactorBase':
        """Square root, NaN for negative values."""
        from . import Factor
        
        result = self.data.copy()
        result['factor'] = np.where(
            result['factor'] >= 0,
            np.sqrt(result['factor']),
            np.nan
        )
        return Factor(result, f"sqrt({self.name})")

    def s_log_1p(self: 'FactorBase') -> 'FactorBase':
        """Signed log1p: sign(x) * log(1 + |x|)."""
        from . import Factor
        
        result = self.data.copy()
        result['factor'] = np.sign(result['factor']) * np.log1p(np.abs(result['factor']))
        return Factor(result, f"s_log_1p({self.name})")

    def signed_power(self: 'FactorBase', exponent: Union['FactorBase', float]) -> 'FactorBase':
        """Signed power: sign(x) * |x|^exponent.
        
        Parameters
        ----------
        exponent : Factor or float
            Power exponent
        
        Returns
        -------
        Factor
            Signed power result
        """
        from . import Factor
        from .base import FactorBase
        
        if isinstance(exponent, FactorBase):
            merged = pd.merge(self.data, exponent.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            
            sign = np.sign(merged['factor_x'])
            abs_val = np.abs(merged['factor_x'])
            
            with np.errstate(invalid='ignore', divide='ignore'):
                result_val = sign * (abs_val ** merged['factor_y'])
            
            merged['factor'] = self._replace_inf(result_val)
            
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"signed_power({self.name},{exponent.name})")
        else:
            result = self.data.copy()
            
            sign = np.sign(result['factor'])
            abs_val = np.abs(result['factor'])
            
            with np.errstate(invalid='ignore', divide='ignore'):
                result_val = sign * (abs_val ** exponent)
            
            result['factor'] = self._replace_inf(result_val)
            
            return Factor(result, f"signed_power({self.name},{exponent})")

    def power(self: 'FactorBase', exponent: Union['FactorBase', float]) -> 'FactorBase':
        """Power operation: x^exponent.
        
        Parameters
        ----------
        exponent : Factor or float
            Power exponent
        
        Returns
        -------
        Factor
            Power result
        """
        from . import Factor
        from .base import FactorBase
        
        if isinstance(exponent, FactorBase):
            merged = pd.merge(self.data, exponent.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            
            with np.errstate(invalid='ignore', divide='ignore'):
                merged['factor'] = merged['factor_x'] ** merged['factor_y']
            
            merged['factor'] = self._replace_inf(merged['factor'])
            
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"({self.name}**{exponent.name})")
        else:
            result = self.data.copy()
            
            with np.errstate(invalid='ignore', divide='ignore'):
                result['factor'] = result['factor'] ** exponent
            
            result['factor'] = self._replace_inf(result['factor'])
            
            return Factor(result, f"({self.name}**{exponent})")

    def add(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Add other factor or scalar."""
        return self.__add__(other)
    
    def subtract(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Subtract other factor or scalar."""
        return self.__sub__(other)
    
    def multiply(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Multiply by other factor or scalar."""
        return self.__mul__(other)
    
    def divide(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Divide by other factor or scalar."""
        return self.__truediv__(other)

    def where(self: 'FactorBase', cond: 'FactorBase', other: Union['FactorBase', float] = np.nan) -> 'FactorBase':
        """Select x when condition is True, else y.
        
        Parameters
        ----------
        cond : Factor
            Boolean condition factor
        other : Factor or float, default np.nan
            Values where condition is False
        
        Returns
        -------
        Factor
            Conditional selection result
        """
        from . import Factor
        from .base import FactorBase
        
        if not isinstance(cond, FactorBase):
            raise TypeError("cond must be a Factor object.")

        merged = pd.merge(self.data, cond.data,
                         on=['timestamp', 'symbol'],
                         suffixes=('', '_cond'))
        
        cond_bool = merged['factor_cond'].fillna(False).astype(bool)
        
        if isinstance(other, FactorBase):
            merged = pd.merge(merged, other.data.rename(columns={'factor': 'factor_other'}),
                            on=['timestamp', 'symbol'])
            merged['factor'] = np.where(cond_bool, merged['factor'], merged['factor_other'])
        else:
            merged['factor'] = np.where(cond_bool, merged['factor'], other)
        
        result = merged[['timestamp', 'symbol', 'factor']]
        return Factor(result, name=f"where({self.name})")

    def maximum(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Element-wise maximum with other factor or scalar."""
        from . import Factor
        from .base import FactorBase
        
        if isinstance(other, FactorBase):
            merged = pd.merge(self.data, other.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            merged['factor'] = np.maximum(merged['factor_x'], merged['factor_y'])
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"max({self.name},{other.name})")
        else:
            result = self.data.copy()
            result['factor'] = np.maximum(result['factor'], other)
            return Factor(result, f"max({self.name},{other})")
    
    def minimum(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Element-wise minimum with other factor or scalar."""
        from . import Factor
        from .base import FactorBase
        
        if isinstance(other, FactorBase):
            merged = pd.merge(self.data, other.data,
                             on=['timestamp', 'symbol'],
                             suffixes=('_x', '_y'))
            merged['factor'] = np.minimum(merged['factor_x'], merged['factor_y'])
            result = merged[['timestamp', 'symbol', 'factor']]
            return Factor(result, f"min({self.name},{other.name})")
        else:
            result = self.data.copy()
            result['factor'] = np.minimum(result['factor'], other)
            return Factor(result, f"min({self.name},{other})")

    def reverse(self: 'FactorBase') -> 'FactorBase':
        """Negate factor (multiply by -1)."""
        return self.__neg__()

    def __neg__(self: 'FactorBase') -> 'FactorBase':
        """Unary negation."""
        return self.multiply(-1)
    
    def __add__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Addition operator."""
        return self._binary_op(other, lambda x, y: x + y, '+')
    
    def __radd__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Reflected addition."""
        return self.__add__(other)
    
    def __sub__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Subtraction operator."""
        return self._binary_op(other, lambda x, y: x - y, '-')
    
    def __rsub__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Reflected subtraction."""
        from . import Factor
        from .base import FactorBase
        
        if isinstance(other, FactorBase):
            return other.subtract(self)
        else:
            result = self.data.copy()
            result['factor'] = other - result['factor']
            return Factor(result, f"({other}-{self.name})")
    
    def __mul__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Multiplication operator."""
        return self._binary_op(other, lambda x, y: x * y, '*')
    
    def __rmul__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Reflected multiplication."""
        return self.__mul__(other)

    def __abs__(self: 'FactorBase') -> 'FactorBase':
        """Absolute value."""
        return self.abs()

    def __pow__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Power operator."""
        return self.power(other)
    
    def __rpow__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Reflected power."""
        from . import Factor
        from .base import FactorBase
        
        if isinstance(other, FactorBase):
            return other.power(self)
        else:
            result = self.data.copy()
            
            with np.errstate(invalid='ignore', divide='ignore'):
                result['factor'] = other ** result['factor']
            
            result['factor'] = self._replace_inf(result['factor'])
            
            return Factor(result, f"({other}**{self.name})")

    def __truediv__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Division operator with safe divide by zero handling."""
        def safe_div(x, y):
            if isinstance(y, (int, float)):
                if y == 0:
                    return np.full_like(x, np.nan) if hasattr(x, '__len__') else np.nan
                return x / y
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = x / y
                return np.where(np.isinf(result) | np.isnan(result), np.nan, result)
        
        return self._binary_op(other, safe_div, '/')

    def __rtruediv__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Reflected division."""
        from . import Factor
        from .base import FactorBase
        
        if isinstance(other, FactorBase):
            return other.__truediv__(self)
        else:
            result = self.data.copy()
            
            result['factor'] = np.where(
                result['factor'] != 0,
                other / result['factor'],
                np.nan
            )
            
            return Factor(result, f"({other}/{self.name})")

    def __lt__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Less than comparison."""
        return self._comparison_op(other, lambda x, y: x < y, '<')

    def __le__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Less than or equal comparison."""
        return self._comparison_op(other, lambda x, y: x <= y, '<=')

    def __gt__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Greater than comparison."""
        return self._comparison_op(other, lambda x, y: x > y, '>')

    def __ge__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Greater than or equal comparison."""
        return self._comparison_op(other, lambda x, y: x >= y, '>=')

    def __eq__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Equality comparison."""
        return self._comparison_op(other, lambda x, y: x == y, '==')

    def __ne__(self: 'FactorBase', other: Union['FactorBase', float]) -> 'FactorBase':
        """Not equal comparison."""
        return self._comparison_op(other, lambda x, y: x != y, '!=')
