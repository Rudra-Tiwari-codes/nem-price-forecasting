"""
Sliding Window Strategy.

Finds local minima and maxima within a sliding window to identify
optimal charge/discharge opportunities.

Algorithm: O(n * k) where k is window size
"""

import numpy as np
import pandas as pd
from typing import Tuple
from collections import deque


def find_local_extrema(
    prices: np.ndarray,
    window_size: int = 12  # 1 hour at 5-min intervals
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local minima and maxima using sliding window.
    
    Args:
        prices: Array of prices
        window_size: Number of intervals to look back/forward
        
    Returns:
        Tuple of (is_local_min array, is_local_max array)
    """
    n = len(prices)
    is_min = np.zeros(n, dtype=bool)
    is_max = np.zeros(n, dtype=bool)
    
    half_window = window_size // 2
    
    for i in range(half_window, n - half_window):
        window = prices[i - half_window:i + half_window + 1]
        current = prices[i]
        
        if current == np.min(window):
            is_min[i] = True
        if current == np.max(window):
            is_max[i] = True
    
    return is_min, is_max


def run_sliding_window_strategy(
    df: pd.DataFrame,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90,
    window_size: int = 24,  # 2 hours
    min_profit_margin: float = 20.0  # Minimum $/MWh profit to trade
) -> pd.DataFrame:
    """
    Run sliding window strategy simulation.
    
    The strategy:
    1. Identifies local price minima within the window
    2. Charges at local minima if expected profit > margin
    3. Identifies local price maxima within the window
    4. Discharges at local maxima
    
    Args:
        df: DataFrame with SETTLEMENTDATE and RRP columns
        capacity_mwh: Battery capacity
        power_mw: Max power rating
        efficiency: Round-trip efficiency
        window_size: Size of sliding window (intervals)
        min_profit_margin: Minimum profit margin to trigger trade
        
    Returns:
        DataFrame with simulation results
    """
    prices = df['RRP'].values
    n = len(prices)
    interval_hours = 5/60
    max_energy = power_mw * interval_hours
    efficiency_factor = np.sqrt(efficiency)
    
    # Find local extrema
    is_min, is_max = find_local_extrema(prices, window_size)
    
    # Initialize arrays
    actions = np.full(n, 'hold', dtype=object)
    energy = np.zeros(n)
    soc = np.zeros(n)
    profit = np.zeros(n)
    cumulative_profit = np.zeros(n)
    
    current_soc = 0.0
    total_profit = 0.0
    last_buy_price = 0.0
    
    half_window = window_size // 2
    
    for i in range(n):
        price = prices[i]
        
        # Look ahead to find max price in upcoming window
        future_end = min(i + window_size, n)
        future_max = np.max(prices[i:future_end]) if i < future_end else price
        
        # Calculate expected profit from current price to future max
        expected_profit = (future_max * efficiency_factor) - (price / efficiency_factor)
        
        if is_min[i] and current_soc < capacity_mwh and expected_profit > min_profit_margin:
            # Local minimum - good time to charge
            charge_amount = min(max_energy, capacity_mwh - current_soc)
            grid_energy = charge_amount / efficiency_factor
            cost = grid_energy * price
            
            last_buy_price = price
            actions[i] = 'charge'
            energy[i] = charge_amount
            current_soc += charge_amount
            profit[i] = -cost
            
        elif is_max[i] and current_soc > 0:
            # Local maximum - good time to discharge
            # Only sell if profitable
            effective_sell = price * efficiency_factor
            effective_buy = last_buy_price / efficiency_factor if last_buy_price > 0 else 0
            
            if effective_sell > effective_buy:
                discharge_amount = min(max_energy, current_soc)
                grid_energy = discharge_amount * efficiency_factor
                revenue = grid_energy * price
                
                actions[i] = 'discharge'
                energy[i] = discharge_amount
                current_soc -= discharge_amount
                profit[i] = revenue
        
        soc[i] = current_soc
        total_profit += profit[i]
        cumulative_profit[i] = total_profit
    
    result = df.copy()
    result['action'] = actions
    result['energy_mwh'] = energy
    result['soc'] = soc
    result['profit'] = profit
    result['cumulative_profit'] = cumulative_profit
    result['is_local_min'] = is_min
    result['is_local_max'] = is_max
    
    return result


def optimize_window_size(
    df: pd.DataFrame,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90,
    window_sizes: list = [6, 12, 24, 48, 96]  # 30min to 8 hours
) -> dict:
    """
    Find optimal window size via parameter sweep.
    
    Returns:
        Dict with optimal window size and profit
    """
    best_profit = float('-inf')
    best_params = {}
    
    for window in window_sizes:
        result = run_sliding_window_strategy(
            df, capacity_mwh, power_mw, efficiency, window_size=window
        )
        profit = result['cumulative_profit'].iloc[-1]
        
        if profit > best_profit:
            best_profit = profit
            best_params = {
                'window_size': window,
                'window_hours': window * 5 / 60,
                'profit': profit
            }
    
    return best_params


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace('strategies/sliding_window.py', ''))
    from data_loader import load_dispatch_data
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "data" / "combined_dispatch_prices.csv"
    
    if data_path.exists():
        df = load_dispatch_data(str(data_path), regions=['SA1'])
        print(f"Loaded {len(df)} price intervals for SA1")
        
        # Run with default window
        result = run_sliding_window_strategy(df, capacity_mwh=100, power_mw=50, window_size=24)
        
        total_profit = result['cumulative_profit'].iloc[-1]
        num_charges = (result['action'] == 'charge').sum()
        num_discharges = (result['action'] == 'discharge').sum()
        local_mins = result['is_local_min'].sum()
        local_maxs = result['is_local_max'].sum()
        
        print(f"\nSliding Window Strategy Results (2-hour window):")
        print(f"  Local minima found: {local_mins}")
        print(f"  Local maxima found: {local_maxs}")
        print(f"  Total Profit: ${total_profit:,.2f}")
        print(f"  Charge cycles: {num_charges}")
        print(f"  Discharge cycles: {num_discharges}")
        
        # Optimize window size
        print("\nOptimizing window size...")
        optimal = optimize_window_size(df)
        print(f"  Optimal window: {optimal['window_size']} intervals ({optimal['window_hours']:.1f} hours)")
        print(f"  Optimal profit: ${optimal['profit']:,.2f}")
    else:
        print(f"Data not found: {data_path}")
