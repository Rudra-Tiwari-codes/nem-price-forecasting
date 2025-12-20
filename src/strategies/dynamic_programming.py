"""
Dynamic Programming Strategy for Battery Arbitrage.

Optimal solution using discrete SoC states. Computes maximum achievable 
profit by considering all possible charge/discharge/hold decisions at 
each time step.

Algorithm: O(n * m) where:
    n = number of time intervals
    m = number of discrete SoC states
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def run_dp_strategy(
    df: pd.DataFrame,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90,
    num_soc_states: int = 21  # 0%, 5%, 10%, ..., 100%
) -> pd.DataFrame:
    """
    Run dynamic programming strategy simulation.
    
    Uses backward induction to find optimal trading decisions at each
    time step given m discrete State of Charge levels.
    
    Args:
        df: DataFrame with SETTLEMENTDATE and RRP columns
        capacity_mwh: Battery capacity
        power_mw: Max power rating
        efficiency: Round-trip efficiency
        num_soc_states: Number of discrete SoC states (more = higher accuracy)
        
    Returns:
        DataFrame with simulation results
    """
    prices = df['RRP'].values
    n = len(prices)
    interval_hours = 5/60  # 5 minutes
    max_energy = power_mw * interval_hours
    efficiency_factor = np.sqrt(efficiency)
    
    # Discrete SoC levels: 0, delta, 2*delta, ..., capacity
    soc_levels = np.linspace(0, capacity_mwh, num_soc_states)
    delta_soc = soc_levels[1] - soc_levels[0] if num_soc_states > 1 else capacity_mwh
    
    # Limit energy per interval to discretization step(s)
    max_charge_steps = max(1, int(np.ceil(max_energy / delta_soc)))
    max_discharge_steps = max_charge_steps
    
    # DP table: dp[t][s] = max profit achievable from time t onwards, starting at SoC state s
    dp = np.full((n + 1, num_soc_states), -np.inf)
    dp[n, :] = 0  # Terminal condition: no future profit at end
    
    # Decision table: stores optimal action at each (time, state)
    # -k = discharge k steps, 0 = hold, +k = charge k steps
    decision = np.zeros((n, num_soc_states), dtype=int)
    
    # Backward induction
    for t in range(n - 1, -1, -1):
        price = prices[t]
        
        for s in range(num_soc_states):
            current_soc = soc_levels[s]
            best_value = -np.inf
            best_action = 0
            
            # Option 1: Hold
            hold_value = dp[t + 1, s]
            if hold_value > best_value:
                best_value = hold_value
                best_action = 0
            
            # Option 2: Charge (buy power from grid)
            for k in range(1, max_charge_steps + 1):
                new_s = s + k
                if new_s >= num_soc_states:
                    break
                    
                # Energy stored in battery
                energy_stored = k * delta_soc
                # Energy drawn from grid (accounting for charging losses)
                energy_from_grid = energy_stored / efficiency_factor
                cost = energy_from_grid * price
                
                charge_value = -cost + dp[t + 1, new_s]
                if charge_value > best_value:
                    best_value = charge_value
                    best_action = k  # Positive = charge
            
            # Option 3: Discharge (sell power to grid)
            for k in range(1, max_discharge_steps + 1):
                new_s = s - k
                if new_s < 0:
                    break
                    
                # Energy discharged from battery
                energy_discharged = k * delta_soc
                # Energy delivered to grid (accounting for discharge losses)
                energy_to_grid = energy_discharged * efficiency_factor
                revenue = energy_to_grid * price
                
                discharge_value = revenue + dp[t + 1, new_s]
                if discharge_value > best_value:
                    best_value = discharge_value
                    best_action = -k  # Negative = discharge
            
            dp[t, s] = best_value
            decision[t, s] = best_action
    
    # Forward pass to reconstruct optimal path
    actions = np.full(n, 'hold', dtype=object)
    energy = np.zeros(n)
    soc = np.zeros(n)
    profit = np.zeros(n)
    cumulative_profit = np.zeros(n)
    
    current_state = 0  # Start at 0% SoC
    current_soc = 0.0
    total_profit = 0.0
    
    for t in range(n):
        price = prices[t]
        action = decision[t, current_state]
        
        if action > 0:
            # Charge
            energy_stored = action * delta_soc
            energy_from_grid = energy_stored / efficiency_factor
            cost = energy_from_grid * price
            
            actions[t] = 'charge'
            energy[t] = min(energy_stored, max_energy)
            current_state += action
            current_soc = soc_levels[current_state]
            profit[t] = -cost
            
        elif action < 0:
            # Discharge
            k = abs(action)
            energy_discharged = k * delta_soc
            energy_to_grid = energy_discharged * efficiency_factor
            revenue = energy_to_grid * price
            
            actions[t] = 'discharge'
            energy[t] = min(energy_discharged, max_energy)
            current_state -= k
            current_soc = soc_levels[current_state]
            profit[t] = revenue
            
        # Hold: no change to state
        
        soc[t] = current_soc
        total_profit += profit[t]
        cumulative_profit[t] = total_profit
    
    result = df.copy()
    result['action'] = actions
    result['energy_mwh'] = energy
    result['soc'] = soc
    result['profit'] = profit
    result['cumulative_profit'] = cumulative_profit
    
    return result


def optimize_soc_resolution(
    df: pd.DataFrame,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90,
    resolutions: list = [11, 21, 41, 51]
) -> dict:
    """
    Find optimal SoC resolution by testing different discretization levels.
    
    Args:
        df: Price data (use a subset for speed)
        capacity_mwh: Battery capacity
        power_mw: Max power
        efficiency: Efficiency
        resolutions: List of num_soc_states to try
        
    Returns:
        Dict with results for each resolution
    """
    results = {}
    
    for res in resolutions:
        result = run_dp_strategy(
            df, capacity_mwh, power_mw, efficiency, num_soc_states=res
        )
        profit = result['cumulative_profit'].iloc[-1]
        results[res] = {
            'profit': profit,
            'charges': (result['action'] == 'charge').sum(),
            'discharges': (result['action'] == 'discharge').sum()
        }
    
    return results


if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, str(__file__).replace('strategies/dynamic_programming.py', ''))
    from data_loader import load_dispatch_data
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "data" / "combined_dispatch_prices.csv"
    
    if data_path.exists():
        df = load_dispatch_data(str(data_path), regions=['SA1'])
        print(f"Loaded {len(df)} price intervals for SA1")
        
        # Use subset for faster testing
        test_df = df.head(2000)  # ~7 days of data
        print(f"Testing on {len(test_df)} intervals...")
        
        start = time.perf_counter()
        result = run_dp_strategy(test_df, capacity_mwh=100, power_mw=50, num_soc_states=21)
        elapsed = time.perf_counter() - start
        
        total_profit = result['cumulative_profit'].iloc[-1]
        num_charges = (result['action'] == 'charge').sum()
        num_discharges = (result['action'] == 'discharge').sum()
        
        print(f"\nDynamic Programming Results (21 SoC states):")
        print(f"  Total Profit: ${total_profit:,.2f}")
        print(f"  Charge cycles: {num_charges}")
        print(f"  Discharge cycles: {num_discharges}")
        print(f"  Time: {elapsed*1000:.1f}ms")
        
        # Test different resolutions
        print("\nTesting different SoC resolutions...")
        resolutions = optimize_soc_resolution(test_df)
        for res, data in resolutions.items():
            print(f"  {res} states: ${data['profit']:,.2f} ({data['charges']}/{data['discharges']} cycles)")
    else:
        print(f"Data not found: {data_path}")
