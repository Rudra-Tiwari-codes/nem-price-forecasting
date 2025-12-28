"""
Dynamic Programming Strategy - Wrapper for backward compatibility.

This module now wraps run_perfect_foresight since they use the same algorithm.
The Perfect Foresight strategy uses true dynamic programming to find the
globally optimal trading sequence.
"""

from .perfect_foresight import run_perfect_foresight


def run_dp_strategy(
    df,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90,
    num_soc_states: int = 21
):
    """
    Run dynamic programming strategy simulation.
    
    This is now an alias for run_perfect_foresight since both use
    the same O(n*m) backward-recursion dynamic programming algorithm.
    
    Args:
        df: DataFrame with SETTLEMENTDATE and RRP columns
        capacity_mwh: Battery capacity
        power_mw: Max power rating
        efficiency: Round-trip efficiency
        num_soc_states: Number of discrete SoC states
        
    Returns:
        DataFrame with simulation results
    """
    return run_perfect_foresight(
        df, 
        capacity_mwh=capacity_mwh, 
        power_mw=power_mw, 
        efficiency=efficiency,
        soc_levels=num_soc_states
    )


# Keep old function name for backward compatibility
optimize_soc_resolution = None  # Deprecated
