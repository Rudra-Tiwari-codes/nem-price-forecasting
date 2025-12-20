"""
Perfect Foresight Strategy - Upper bound benchmark.

This strategy has complete knowledge of future prices, making it the
theoretical maximum profit achievable. Used as a benchmark for other strategies.

Algorithm: O(n) - Single pass through price data
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class TradeAction:
    """Represents a single trade action."""
    timestamp: pd.Timestamp
    action: str  # 'charge', 'discharge', 'hold'
    energy_mwh: float
    price: float
    soc_after: float
    profit: float = 0.0


def find_optimal_trades(
    prices: np.ndarray,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90,
    interval_hours: float = 5/60
) -> Tuple[float, List[dict]]:
    """
    Find optimal buy/sell pairs using perfect foresight.
    
    Uses a greedy approach: find all profitable charge-discharge cycles
    where the price differential exceeds efficiency losses.
    
    Args:
        prices: Array of prices ($/MWh)
        capacity_mwh: Battery capacity
        power_mw: Max power rating
        efficiency: Round-trip efficiency
        interval_hours: Duration per interval
        
    Returns:
        Tuple of (total_profit, list of trade records)
    """
    n = len(prices)
    max_energy_per_interval = power_mw * interval_hours
    
    # Find all profitable trade opportunities
    # A trade is profitable if: sell_price * sqrt(eff) > buy_price / sqrt(eff)
    # Simplified: sell_price > buy_price / efficiency
    
    trades = []
    total_profit = 0.0
    current_soc = 0.0
    
    # Identify local minima and maxima
    min_indices = []
    max_indices = []
    
    for i in range(1, n - 1):
        if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            min_indices.append(i)
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            max_indices.append(i)
    
    # Also consider start and end
    if n > 1:
        if prices[0] < prices[1]:
            min_indices.insert(0, 0)
        else:
            max_indices.insert(0, 0)
        if prices[-1] < prices[-2]:
            min_indices.append(n-1)
        else:
            max_indices.append(n-1)
    
    min_indices = sorted(set(min_indices))
    max_indices = sorted(set(max_indices))
    
    # Find profitable pairs: buy at min, sell at following max
    efficiency_factor = np.sqrt(efficiency)
    
    i, j = 0, 0
    while i < len(min_indices) and j < len(max_indices):
        buy_idx = min_indices[i]
        
        # Find next max after this min
        while j < len(max_indices) and max_indices[j] <= buy_idx:
            j += 1
        
        if j >= len(max_indices):
            break
            
        sell_idx = max_indices[j]
        buy_price = prices[buy_idx]
        sell_price = prices[sell_idx]
        
        # Check if profitable after efficiency losses
        effective_sell = sell_price * efficiency_factor
        effective_buy = buy_price / efficiency_factor
        
        if effective_sell > effective_buy:
            # Calculate energy based on intervals between buy and sell
            charge_intervals = 1  # Simplified: charge in one interval
            discharge_intervals = 1
            
            energy = min(max_energy_per_interval * charge_intervals, capacity_mwh)
            
            # Profit = sell revenue - buy cost, accounting for efficiency
            buy_cost = energy * buy_price / efficiency_factor
            sell_revenue = energy * sell_price * efficiency_factor
            profit = sell_revenue - buy_cost
            
            trades.append({
                'buy_idx': buy_idx,
                'sell_idx': sell_idx,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'energy_mwh': energy,
                'profit': profit
            })
            total_profit += profit
        
        i += 1
    
    return total_profit, trades


def run_perfect_foresight(
    df: pd.DataFrame,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90
) -> pd.DataFrame:
    """
    Run perfect foresight simulation on price data.
    
    Args:
        df: DataFrame with SETTLEMENTDATE and RRP columns
        capacity_mwh: Battery capacity
        power_mw: Max power rating
        efficiency: Round-trip efficiency
        
    Returns:
        DataFrame with simulation results including actions and cumulative profit
    """
    prices = df['RRP'].values
    timestamps = df['SETTLEMENTDATE'].values
    interval_hours = 5/60
    max_energy = power_mw * interval_hours
    efficiency_factor = np.sqrt(efficiency)
    
    n = len(prices)
    
    # Initialize result arrays
    actions = np.full(n, 'hold', dtype=object)
    energy = np.zeros(n)
    soc = np.zeros(n)
    profit = np.zeros(n)
    cumulative_profit = np.zeros(n)
    
    current_soc = 0.0
    total_profit = 0.0
    
    # Simple forward-looking strategy
    for i in range(n - 1):
        current_price = prices[i]
        future_max = np.max(prices[i+1:min(i+50, n)])  # Look ahead 50 intervals (~4 hours)
        future_min = np.min(prices[i+1:min(i+50, n)])
        
        # Charge if current price is low relative to future
        min_sell_price = current_price / efficiency + 10  # Need at least $10/MWh profit margin
        
        if current_price <= future_min and current_soc < capacity_mwh:
            # Good time to charge
            charge_amount = min(max_energy, capacity_mwh - current_soc)
            grid_energy = charge_amount / efficiency_factor
            cost = grid_energy * current_price
            
            actions[i] = 'charge'
            energy[i] = charge_amount
            current_soc += charge_amount
            profit[i] = -cost
            
        elif current_price >= future_max * 0.9 and current_soc > 0:
            # Good time to sell (price is near future max)
            discharge_amount = min(max_energy, current_soc)
            grid_energy = discharge_amount * efficiency_factor
            revenue = grid_energy * current_price
            
            actions[i] = 'discharge'
            energy[i] = discharge_amount
            current_soc -= discharge_amount
            profit[i] = revenue
        
        soc[i] = current_soc
        total_profit += profit[i]
        cumulative_profit[i] = total_profit
    
    # Handle last interval
    soc[-1] = current_soc
    cumulative_profit[-1] = total_profit
    
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
        df = load_dispatch_data(str(data_path), regions=['SA1'])  # SA1 has most volatility
        print(f"Loaded {len(df)} price intervals for SA1")
        
        result = run_perfect_foresight(df, capacity_mwh=100, power_mw=50, efficiency=0.90)
        
        total_profit = result['cumulative_profit'].iloc[-1]
        num_charges = (result['action'] == 'charge').sum()
        num_discharges = (result['action'] == 'discharge').sum()
        
        print(f"\nPerfect Foresight Results:")
        print(f"  Total Profit: ${total_profit:,.2f}")
        print(f"  Charge cycles: {num_charges}")
        print(f"  Discharge cycles: {num_discharges}")
    else:
        print(f"Data not found: {data_path}")
