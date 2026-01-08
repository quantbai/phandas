"""Input/output and utility methods for Factor."""

import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from typing import Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import FactorBase


class IOMixin:
    """Mixin providing I/O and utility methods.
    
    Includes data export, visualization, and introspection methods.
    """

    def to_weights(self: 'FactorBase', date: Optional[Union[str, pd.Timestamp]] = None) -> dict:
        """Convert factor to portfolio weights for a specific date.
        
        If factor is not already a dollar-neutral signal, it will be
        demeaned and scaled automatically.
        
        Parameters
        ----------
        date : str or Timestamp, optional
            Target date. If None, uses latest date.
        
        Returns
        -------
        dict
            Dictionary mapping symbol to weight
        """
        
        if date is None:
            latest_date = self.data['timestamp'].max()
            utc_today = pd.Timestamp(datetime.now(timezone.utc).date())
            
            if latest_date >= utc_today:
                all_dates = sorted(self.data['timestamp'].unique())
                if len(all_dates) >= 2:
                    target_date = all_dates[-2]
                    warnings.warn(
                        f"Latest date {latest_date.strftime('%Y-%m-%d')} appears to be incomplete (UTC today). "
                        f"Automatically using {target_date.strftime('%Y-%m-%d')} instead. "
                        f"To override, explicitly pass date parameter."
                    )
                else:
                    target_date = latest_date
            else:
                target_date = latest_date
        else:
            target_date = pd.to_datetime(date)
        
        pivot = self.data.pivot(index='timestamp', columns='symbol', values='factor')
        
        first_valid_date = pivot.dropna(how='any').index.min()
        
        if first_valid_date is not None and target_date >= first_valid_date:
            target_row = pivot.loc[target_date] if target_date in pivot.index else None
            if target_row is not None:
                nan_symbols = target_row[target_row.isna()].index.tolist()
                if nan_symbols:
                    warnings.warn(
                        f"Factor has NaN values on {target_date.strftime('%Y-%m-%d')}: {nan_symbols}. "
                        f"Consider using ts_backfill() to handle missing values."
                    )
        
        date_data = self.data[self.data['timestamp'] == target_date]
        if date_data.empty:
            return {}
        
        factors = date_data.set_index('symbol')['factor'].dropna()
        if factors.empty:
            return {}
        
        long_sum = factors[factors > 0].sum()
        short_sum = factors[factors < 0].sum()
        total_sum = long_sum + short_sum
        is_already_signal = (np.isclose(long_sum, 0.5, atol=1e-2) and 
                            np.isclose(short_sum, -0.5, atol=1e-2) and
                            np.isclose(total_sum, 0.0, atol=1e-2))
        
        if is_already_signal:
            weights = factors
        else:
            demeaned = factors - factors.mean()
            abs_sum = np.abs(demeaned).sum()
            
            if abs_sum < 1e-10:
                return {}
            
            weights = demeaned / abs_sum
        
        return weights.to_dict()

    def to_df(self: 'FactorBase') -> pd.DataFrame:
        """Export factor data as DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Copy of factor data with columns (timestamp, symbol, factor)
        """
        return self.data.copy()
    
    def to_csv(self: 'FactorBase', path: str) -> str:
        """Export factor data to CSV file.
        
        Parameters
        ----------
        path : str
            Output file path
        
        Returns
        -------
        str
            Path to saved file
        """
        self.data.to_csv(path, index=False)
        return path
    
    @property
    def symbols(self: 'FactorBase') -> List[str]:
        """List of unique symbols in factor data."""
        return self.data['symbol'].unique().tolist()
    
    @property
    def timestamps(self: 'FactorBase') -> pd.DatetimeIndex:
        """DatetimeIndex of unique timestamps in factor data."""
        return pd.DatetimeIndex(self.data['timestamp'].unique())
    
    def __len__(self: 'FactorBase') -> int:
        """Number of observations in factor data."""
        return len(self.data)
    
    def info(self: 'FactorBase') -> None:
        """Print summary information about the factor."""
        from ..console import print
        n_obs = len(self.data)
        n_symbols = self.data['symbol'].nunique()
        n_nan = self.data['factor'].isna().sum()
        nan_ratio = n_nan / n_obs if n_obs > 0 else 0
        time_range = f"{self.data['timestamp'].min().strftime('%Y-%m-%d')} to {self.data['timestamp'].max().strftime('%Y-%m-%d')}"
        
        print(f"Factor: {self.name}")
        print(f"  obs={n_obs}, symbols={n_symbols}, period={time_range}")
        print(f"  NaN: {n_nan} ({nan_ratio:.1%})")
    
    def __repr__(self: 'FactorBase') -> str:
        """String representation for developers."""
        n_obs = len(self.data)
        n_symbols = self.data['symbol'].nunique()
        valid_ratio = self.data['factor'].notna().mean()
        time_range = f"{self.data['timestamp'].min().strftime('%Y-%m-%d')} to {self.data['timestamp'].max().strftime('%Y-%m-%d')}"
        return (f"Factor(name={self.name}, obs={n_obs}, symbols={n_symbols}, "
               f"valid={valid_ratio:.1%}, period={time_range})")
    
    def __str__(self: 'FactorBase') -> str:
        """User-friendly string representation."""
        n_symbols = self.data['symbol'].nunique()
        return f"Factor({self.name}): {len(self.data)} obs, {n_symbols} symbols"

    def show(self: 'FactorBase', symbol: Optional[str] = None, figsize: tuple = (12, 6), 
             title: Optional[str] = None) -> None:
        """Plot factor time series.
        
        Parameters
        ----------
        symbol : str, optional
            Specific symbol to plot. If None, plots all symbols.
        figsize : tuple, default (12, 6)
            Figure size
        title : str, optional
            Plot title
        """
        from ..plot import FactorPlotter
        plotter = FactorPlotter(self)
        plotter.plot(symbol, figsize, title)
