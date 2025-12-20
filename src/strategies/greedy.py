"""
Greedy Threshold Strategy.

Simple but effective: buy when price is below threshold, sell when above.
Thresholds can be static or dynamically calculated from historical data.

Algorithm: O(n) - Single pass through price data
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_thresholds(
    prices: np.ndarray,
    buy_percentile: float = 25,
    sell_percentile: float = 75
) -> Tuple[float, float]:
    """
    Calculate buy/sell thresholds from historical prices.
    
    Args:
        prices: Array of historical prices
        buy_percentile: Percentile for buy threshold (lower = more selective)
        sell_percentile: Percentile for sell threshold (higher = more selective)
        
    Returns:
        Tuple of (buy_threshold, sell_threshold)
    """
    buy_threshold = np.percentile(prices, buy_percentile)
    sell_threshold = np.percentile(prices, sell_percentile)
    return buy_threshold, sell_threshold


def run_greedy_strategy(
    df: pd.DataFrame,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90,
    buy_threshold: Optional[float] = None,
    sell_threshold: Optional[float] = None,
    buy_percentile: float = 25,
    sell_percentile: float = 75
) -> pd.DataFrame:
    """
    Run greedy threshold strategy simulation.
    
    Args:
        df: DataFrame with SETTLEMENTDATE and RRP columns
        capacity_mwh: Battery capacity
        power_mw: Max power rating
        efficiency: Round-trip efficiency
        buy_threshold: Fixed buy threshold ($/MWh), or None to calculate
        sell_threshold: Fixed sell threshold ($/MWh), or None to calculate
        buy_percentile: Percentile for dynamic buy threshold
        sell_percentile: Percentile for dynamic sell threshold
        
    Returns:
        DataFrame with simulation results
    """
    prices = df['RRP'].values
    n = len(prices)
    interval_hours = 5/60
    max_energy = power_mw * interval_hours
    efficiency_factor = np.sqrt(efficiency)
    
    # Calculate thresholds if not provided
    if buy_threshold is None or sell_threshold is None:
        auto_buy, auto_sell = calculate_thresholds(
            prices, buy_percentile, sell_percentile
        )
        buy_threshold = buy_threshold or auto_buy
        sell_threshold = sell_threshold or auto_sell
    
    # Ensure sell threshold accounts for round-trip efficiency
    min_profitable_sell = buy_threshold / efficiency + 5  # Add $5 margin
    sell_threshold = max(sell_threshold, min_profitable_sell)
    
    # Initialize arrays
    actions = np.full(n, 'hold', dtype=object)
    energy = np.zeros(n)
    soc = np.zeros(n)
    profit = np.zeros(n)
    cumulative_profit = np.zeros(n)
    
    current_soc = 0.0
    total_profit = 0.0
    avg_buy_price = 0.0
    total_bought = 0.0
    
    for i in range(n):
        price = prices[i]
        
        if price <= buy_threshold and current_soc < capacity_mwh:
            # Buy signal - charge the battery
            charge_amount = min(max_energy, capacity_mwh - current_soc)
            grid_energy = charge_amount / efficiency_factor
            cost = grid_energy * price
            
            # Track weighted average buy price
            total_bought += charge_amount
            avg_buy_price = (avg_buy_price * (total_bought - charge_amount) + 
                           price * charge_amount) / total_bought if total_bought > 0 else price
            
            actions[i] = 'charge'
            energy[i] = charge_amount
            current_soc += charge_amount
            profit[i] = -cost
            
        elif price >= sell_threshold and current_soc > 0:
            # Sell signal - discharge the battery
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
    
    return result, {'buy_threshold': buy_threshold, 'sell_threshold': sell_threshold}


def optimize_thresholds(
    df: pd.DataFrame,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90,
    percentile_range: range = range(10, 50, 5)
) -> dict:
    """
    Find optimal buy/sell percentile thresholds via grid search.
    
    Args:
        df: Price data
        capacity_mwh: Battery capacity
        power_mw: Max power
        efficiency: Efficiency
        percentile_range: Range of percentiles to try
        
    Returns:
        Dict with optimal thresholds and profit
    """
    best_profit = float('-inf')
    best_params = {}
    
    for buy_pct in percentile_range:
        for sell_pct in range(100 - buy_pct, 100, 5):
            result, thresholds = run_greedy_strategy(
                df, capacity_mwh, power_mw, efficiency,
                buy_percentile=buy_pct, sell_percentile=sell_pct
            )
            profit = result['cumulative_profit'].iloc[-1]
            
            if profit > best_profit:
                best_profit = profit
                best_params = {
                    'buy_percentile': buy_pct,
                    'sell_percentile': sell_pct,
                    'buy_threshold': thresholds['buy_threshold'],
                    'sell_threshold': thresholds['sell_threshold'],
                    'profit': profit
                }
    
    return best_params


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace('strategies/greedy.py', ''))
    from data_loader import load_dispatch_data
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "data" / "combined_dispatch_prices.csv"
    
    if data_path.exists():
        df = load_dispatch_data(str(data_path), regions=['SA1'])
        print(f"Loaded {len(df)} price intervals for SA1")
        
        # Run with default thresholds
        result, thresholds = run_greedy_strategy(df, capacity_mwh=100, power_mw=50)
        
        total_profit = result['cumulative_profit'].iloc[-1]
        num_charges = (result['action'] == 'charge').sum()
        num_discharges = (result['action'] == 'discharge').sum()
        
        print(f"\nGreedy Strategy Results:")
        print(f"  Buy threshold: ${thresholds['buy_threshold']:.2f}/MWh")
        print(f"  Sell threshold: ${thresholds['sell_threshold']:.2f}/MWh")
        print(f"  Total Profit: ${total_profit:,.2f}")
        print(f"  Charge cycles: {num_charges}")
        print(f"  Discharge cycles: {num_discharges}")
        
        # Optimize thresholds
        print("\nOptimizing thresholds...")
        optimal = optimize_thresholds(df)
        print(f"  Optimal buy percentile: {optimal['buy_percentile']}%")
        print(f"  Optimal sell percentile: {optimal['sell_percentile']}%")
        print(f"  Optimal profit: ${optimal['profit']:,.2f}")
    else:
        print(f"Data not found: {data_path}")
