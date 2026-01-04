"""
Perfect Foresight Strategy - Upper bound benchmark.

This strategy has complete knowledge of future prices, making it the
theoretical maximum profit achievable. Uses dynamic programming to find
the globally optimal trading sequence.

Algorithm: O(n * m) where n = time steps, m = discretized SoC levels
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from functools import lru_cache


def run_perfect_foresight(
    df: pd.DataFrame,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90,
    soc_levels: int = 21
) -> pd.DataFrame:
    """
    Run perfect foresight simulation using dynamic programming.
    
    This is the theoretical upper bound - no other strategy can beat it
    because it has complete knowledge of all future prices.
    
    Args:
        df: DataFrame with SETTLEMENTDATE and RRP columns
        capacity_mwh: Battery capacity
        power_mw: Max power rating
        efficiency: Round-trip efficiency
        soc_levels: Number of discrete SoC levels (more = accurate but slower)
        
    Returns:
        DataFrame with simulation results including actions and cumulative profit
    """
    prices = df['RRP'].values.astype(np.float64)
    n = len(prices)
    interval_hours = 5/60  # 5-minute intervals
    max_energy = power_mw * interval_hours
    efficiency_factor = np.sqrt(efficiency)
    
    # Discretize SoC into levels (0, 1, 2, ..., soc_levels-1)
    # Each level represents capacity_mwh / (soc_levels - 1) MWh
    soc_step = capacity_mwh / (soc_levels - 1)
    
    # Maximum SoC change per interval (in discrete levels)
    max_level_change = max(1, int(np.ceil(max_energy / soc_step)))
    
    # DP table: value[t][s] = maximum profit achievable from time t to end
    # starting with SoC level s
    # We compute this backwards from the end
    
    INF = float('inf')
    
    # value[s] = max profit from current time to end with SoC level s
    # We'll iterate backwards through time
    value = np.zeros(soc_levels)
    next_value = np.zeros(soc_levels)
    
    # best_action[t][s] = best action at time t with SoC level s
    # 0 = hold, positive = charge that many levels, negative = discharge
    best_action = np.zeros((n, soc_levels), dtype=np.int32)
    
    # Backward pass: compute optimal value and action at each step
    for t in range(n - 1, -1, -1):
        price = prices[t]
        
        for s in range(soc_levels):
            current_soc = s * soc_step
            
            # Option 1: Hold
            best_val = next_value[s] if t < n - 1 else 0.0
            best_act = 0
            
            # Option 2: Charge (if not full)
            if s < soc_levels - 1:
                # How many levels can we charge?
                max_charge_levels = min(max_level_change, soc_levels - 1 - s)
                
                for delta in range(1, max_charge_levels + 1):
                    new_s = s + delta
                    charge_energy = delta * soc_step
                    
                    # Cost = energy from grid * price
                    # Energy from grid is higher due to charging losses
                    grid_energy = charge_energy / efficiency_factor
                    cost = grid_energy * price
                    
                    future_val = next_value[new_s] if t < n - 1 else 0.0
                    total_val = -cost + future_val
                    
                    if total_val > best_val:
                        best_val = total_val
                        best_act = delta  # positive = charge
            
            # Option 3: Discharge (if not empty)
            if s > 0:
                # How many levels can we discharge?
                max_discharge_levels = min(max_level_change, s)
                
                for delta in range(1, max_discharge_levels + 1):
                    new_s = s - delta
                    discharge_energy = delta * soc_step
                    
                    # Revenue = energy to grid * price
                    # Energy to grid is lower due to discharge losses
                    grid_energy = discharge_energy * efficiency_factor
                    revenue = grid_energy * price
                    
                    future_val = next_value[new_s] if t < n - 1 else 0.0
                    total_val = revenue + future_val
                    
                    if total_val > best_val:
                        best_val = total_val
                        best_act = -delta  # negative = discharge
            
            value[s] = best_val
            best_action[t, s] = best_act
        
        # Swap arrays for next iteration
        value, next_value = next_value, value
    
    # Forward pass: extract the optimal trajectory
    actions = np.full(n, 'hold', dtype=object)
    energy = np.zeros(n)
    soc = np.zeros(n)
    profit = np.zeros(n)
    cumulative_profit = np.zeros(n)
    
    current_level = 0  # Start empty
    total_profit = 0.0
    
    for t in range(n):
        price = prices[t]
        act = best_action[t, current_level]
        
        if act > 0:
            # Charge
            charge_energy = act * soc_step
            grid_energy = charge_energy / efficiency_factor
            cost = grid_energy * price
            
            actions[t] = 'charge'
            energy[t] = charge_energy
            profit[t] = -cost
            current_level += act
            
        elif act < 0:
            # Discharge
            discharge_energy = (-act) * soc_step
            grid_energy = discharge_energy * efficiency_factor
            revenue = grid_energy * price
            
            actions[t] = 'discharge'
            energy[t] = discharge_energy
            profit[t] = revenue
            current_level += act  # act is negative
        
        soc[t] = current_level * soc_step
        total_profit += profit[t]
        cumulative_profit[t] = total_profit
    
    result = df.copy()
    result['action'] = actions
    result['energy_mwh'] = energy
    result['soc'] = soc
    result['profit'] = profit
    result['cumulative_profit'] = cumulative_profit
    
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace('strategies/perfect_foresight.py', ''))
    from data_loader import load_dispatch_data
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "data" / "combined_dispatch_prices.csv"
    
    if data_path.exists():
        df = load_dispatch_data(str(data_path), regions=['SA1'])
        print(f"Loaded {len(df)} price intervals for SA1")
        
        result = run_perfect_foresight(df, capacity_mwh=100, power_mw=50, efficiency=0.90)
        
        total_profit = result['cumulative_profit'].iloc[-1]
        num_charges = (result['action'] == 'charge').sum()
        num_discharges = (result['action'] == 'discharge').sum()
        
        print(f"\nPerfect Foresight Results (Dynamic Programming):")
        print(f"  Total Profit: ${total_profit:,.2f}")
        print(f"  Charge cycles: {num_charges}")
        print(f"  Discharge cycles: {num_discharges}")
    else:
        print(f"Data not found: {data_path}")
