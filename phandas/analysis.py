"""Factor analysis module for quantitative research reports."""

import warnings
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, TYPE_CHECKING
from scipy import stats as scipy_stats

if TYPE_CHECKING:
    from .core import Factor

from .console import print

_DEFAULT_HORIZONS = [1, 7, 30]


class FactorAnalyzer:
    """Multi-factor analysis for quantitative research."""
    
    def __init__(self, factors: List['Factor'], price: 'Factor', 
                 horizons: Optional[List[int]] = None):
        if not factors:
            raise ValueError("Must provide at least one factor")
        
        self.factors = factors if isinstance(factors, list) else [factors]
        self.price = price
        self.horizons = horizons or _DEFAULT_HORIZONS
        self._forward_returns = None
        self._ic_cache = None
        self._stats_cache = None
        self._corr_cache = None
    
    def _compute_forward_returns(self) -> Dict[int, pd.DataFrame]:
        if self._forward_returns is not None:
            return self._forward_returns
        
        price_pivot = self.price.data.pivot(
            index='timestamp', columns='symbol', values='factor'
        )
        
        self._forward_returns = {}
        for h in self.horizons:
            fwd_ret = price_pivot.shift(-h) / price_pivot - 1
            self._forward_returns[h] = fwd_ret
        
        return self._forward_returns
    
    def correlation(self, method: str = 'pearson') -> pd.DataFrame:
        if len(self.factors) < 2:
            warnings.warn("Need at least 2 factors for correlation")
            return pd.DataFrame()
        
        aligned_data = {}
        for f in self.factors:
            signal_factor = f.signal()
            pivot = signal_factor.data.pivot(index='timestamp', columns='symbol', values='factor')
            aligned_data[f.name] = pivot.stack()
        
        df = pd.DataFrame(aligned_data).dropna()
        
        if df.empty or len(df) < 2:
            warnings.warn("Insufficient overlapping data for correlation")
            return pd.DataFrame()
        
        return df.corr(method=method)
    
    def ic(self, method: str = 'spearman') -> Dict[str, Dict]:
        if self._ic_cache is not None:
            return self._ic_cache
        
        fwd_rets = self._compute_forward_returns()
        results = {}
        
        for factor in self.factors:
            factor_pivot = factor.data.pivot(
                index='timestamp', columns='symbol', values='factor'
            )
            
            factor_results = {}
            for h in self.horizons:
                fwd_ret = fwd_rets[h]
                aligned_factor, aligned_ret = factor_pivot.align(fwd_ret, join='inner')
                
                ic_series = self._compute_ic_vectorized(aligned_factor, aligned_ret, method)
                
                if len(ic_series) > 0:
                    ic_arr = ic_series.values
                    ic_mean = np.nanmean(ic_arr)
                    ic_std = np.nanstd(ic_arr)
                    ir = ic_mean / ic_std if ic_std > 0 else 0
                    t_stat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0
                    
                    factor_results[h] = {
                        'ic_mean': ic_mean,
                        'ic_std': ic_std,
                        'ir': ir,
                        't_stat': t_stat,
                        'ic_series': ic_series
                    }
                else:
                    factor_results[h] = {
                        'ic_mean': np.nan,
                        'ic_std': np.nan,
                        'ir': np.nan,
                        't_stat': np.nan,
                        'ic_series': pd.Series(dtype=float)
                    }
            
            results[factor.name] = factor_results
        
        self._ic_cache = results
        return results
    
    def _compute_ic_vectorized(self, factor_pivot: pd.DataFrame, 
                                ret_pivot: pd.DataFrame, method: str) -> pd.Series:
        if method == 'spearman':
            f_data = factor_pivot.rank(axis=1, na_option='keep')
            r_data = ret_pivot.rank(axis=1, na_option='keep')
        else:
            f_data = factor_pivot
            r_data = ret_pivot
        
        valid_mask = factor_pivot.notna() & ret_pivot.notna()
        valid_count = valid_mask.sum(axis=1)
        
        f_std = f_data.std(axis=1, skipna=True)
        r_std = r_data.std(axis=1, skipna=True)
        std_valid = (f_std > 1e-10) & (r_std > 1e-10) & (valid_count >= 3)
        
        f_demean = f_data.sub(f_data.mean(axis=1, skipna=True), axis=0)
        r_demean = r_data.sub(r_data.mean(axis=1, skipna=True), axis=0)
        
        numer = (f_demean * r_demean).sum(axis=1, skipna=True)
        denom = (f_demean.pow(2).sum(axis=1, skipna=True) * 
                 r_demean.pow(2).sum(axis=1, skipna=True)).pow(0.5)
        
        ic = numer / denom
        ic = ic[std_valid]
        
        return ic.dropna()
    
    def stats(self) -> Dict[str, Dict]:
        if self._stats_cache is not None:
            return self._stats_cache
        
        results = {}
        
        for factor in self.factors:
            pivot = factor.data.pivot(
                index='timestamp', columns='symbol', values='factor'
            )
            
            total_cells = pivot.size
            non_nan_cells = pivot.count().sum()
            coverage = non_nan_cells / total_cells if total_cells > 0 else 0
            
            rank_df = pivot.rank(axis=1, pct=True)
            rank_diff = rank_df.diff().abs()
            turnover = rank_diff.mean().mean() * 2 if not rank_diff.empty else 0
            
            autocorr_list = []
            for symbol in pivot.columns:
                series = pivot[symbol].dropna()
                if len(series) > 10:
                    ac = series.autocorr(lag=1)
                    if not np.isnan(ac):
                        autocorr_list.append(ac)
            
            autocorr = np.mean(autocorr_list) if autocorr_list else np.nan
            
            results[factor.name] = {
                'coverage': coverage,
                'turnover': turnover,
                'autocorr': autocorr
            }
        
        self._stats_cache = results
        return results
    
    def summary(self) -> str:
        ic_results = self.ic()
        stats_results = self.stats()
        corr_matrix = self.correlation() if len(self.factors) > 1 else None
        
        lines = [f"FactorAnalyzer(factors={len(self.factors)}, horizons={self.horizons})"]
        lines.append("")
        
        lines.append("IC Analysis (Spearman):")
        header = "  Factor".ljust(20) + "".join([f"{h}D".rjust(16) for h in self.horizons])
        lines.append(header)
        lines.append("  " + "-" * (18 + 16 * len(self.horizons)))
        
        for factor in self.factors:
            name = factor.name[:18].ljust(18)
            ic_vals = []
            for h in self.horizons:
                ic_data = ic_results[factor.name].get(h, {})
                ic_mean = ic_data.get('ic_mean', np.nan)
                t_stat = ic_data.get('t_stat', np.nan)
                if np.isnan(ic_mean):
                    ic_vals.append("N/A".rjust(16))
                else:
                    ic_vals.append(f"{ic_mean:.4f}|{t_stat:.2f}".rjust(16))
            lines.append(f"  {name}" + "".join(ic_vals))
        
        lines.append("")
        lines.append("IR (IC Mean / IC Std):")
        for factor in self.factors:
            name = factor.name[:18].ljust(18)
            ir_vals = []
            for h in self.horizons:
                ic_data = ic_results[factor.name].get(h, {})
                ir = ic_data.get('ir', np.nan)
                if np.isnan(ir):
                    ir_vals.append("N/A".rjust(12))
                else:
                    ir_vals.append(f"{ir:.3f}".rjust(12))
            lines.append(f"  {name}" + "".join(ir_vals))
        
        lines.append("")
        lines.append("Factor Statistics:")
        lines.append("  Factor".ljust(20) + "Coverage".rjust(12) + "Turnover".rjust(12) + "Autocorr".rjust(12))
        lines.append("  " + "-" * 54)
        for factor in self.factors:
            name = factor.name[:18].ljust(18)
            s = stats_results[factor.name]
            lines.append(f"  {name}" + 
                        f"{s['coverage']:.2%}".rjust(12) +
                        f"{s['turnover']:.4f}".rjust(12) +
                        f"{s['autocorr']:.4f}".rjust(12) if not np.isnan(s['autocorr']) 
                        else f"  {name}" + f"{s['coverage']:.2%}".rjust(12) + 
                             f"{s['turnover']:.4f}".rjust(12) + "N/A".rjust(12))
        
        if corr_matrix is not None and not corr_matrix.empty:
            lines.append("")
            lines.append("Correlation Matrix:")
            corr_str = corr_matrix.to_string(float_format=lambda x: f'{x:.4f}')
            for line in corr_str.split('\n'):
                lines.append(f"  {line}")
        
        return "\n".join(lines)
    
    def print_summary(self) -> 'FactorAnalyzer':
        print(self.summary())
        return self
    
    def plot(self, factor_idx: int = 0, horizon_idx: int = 0,
             figsize: tuple = (14, 10), rolling_window: int = 20) -> 'FactorAnalyzer':
        """Generate 4-panel factor analysis chart.
        
        Parameters
        ----------
        factor_idx : int
            Index of factor to plot (default first factor)
        horizon_idx : int
            Index of horizon to use (default first horizon)
        figsize : tuple
            Figure size
        rolling_window : int
            Window for rolling IC calculation
            
        Returns
        -------
        FactorAnalyzer
            Self for method chaining
        """
        from .plot import FactorAnalysisPlotter
        plotter = FactorAnalysisPlotter(self)
        plotter.plot(factor_idx=factor_idx, horizon_idx=horizon_idx,
                    figsize=figsize, rolling_window=rolling_window)
        return self
    
    def __repr__(self) -> str:
        factor_names = [f.name for f in self.factors]
        return f"FactorAnalyzer(factors={factor_names}, horizons={self.horizons})"


def analyze(factors: Union['Factor', List['Factor']], 
            price: 'Factor',
            horizons: Optional[List[int]] = None) -> FactorAnalyzer:
    """Create FactorAnalyzer for multi-factor analysis.
    
    Parameters
    ----------
    factors : Factor or List[Factor]
        Factor(s) to analyze
    price : Factor
        Price Factor for computing forward returns
    horizons : List[int], optional
        Holding periods to analyze, default [1, 7, 30]
    
    Returns
    -------
    FactorAnalyzer
        Analyzer instance with ic(), stats(), correlation(), print_summary()
    
    Examples
    --------
    >>> report = analyze([alpha1, alpha2], price=close)
    >>> report.print_summary()
    >>> corr = report.correlation()
    >>> ic = report.ic()
    """
    factor_list = factors if isinstance(factors, list) else [factors]
    return FactorAnalyzer(factor_list, price, horizons)
