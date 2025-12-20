"""
Vectorized Simulation Engine.

High-performance simulation using NumPy vectorized operations.
Processes 100k+ data points efficiently.
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable
import time


def vectorized_greedy_simulation(
    prices: np.ndarray,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90,
    buy_threshold: float = 50.0,
    sell_threshold: float = 150.0,
    interval_hours: float = 5/60
) -> Dict[str, np.ndarray]:
    """
    Ultra-fast vectorized greedy simulation.
    
    Uses NumPy operations to minimize Python overhead.
    
    Args:
        prices: Price array
        capacity_mwh: Battery capacity
        power_mw: Max power
        efficiency: Round-trip efficiency
        buy_threshold: Price to buy below
        sell_threshold: Price to sell above
        interval_hours: Duration per interval
        
    Returns:
        Dict with simulation arrays
    """
    n = len(prices)
    max_energy = power_mw * interval_hours
    eff_factor = np.sqrt(efficiency)
    
    # Preallocate arrays
    soc = np.zeros(n)
    actions = np.zeros(n)  # 0=hold, 1=charge, -1=discharge
    energy = np.zeros(n)
    profit = np.zeros(n)
    
    # Vectorized signal generation
    buy_signals = prices <= buy_threshold
    sell_signals = prices >= sell_threshold
    
    # Sequential simulation (SoC depends on previous state)
    current_soc = 0.0
    
    for i in range(n):
        if buy_signals[i] and current_soc < capacity_mwh:
            # Charge
            charge = min(max_energy, capacity_mwh - current_soc)
            grid_energy = charge / eff_factor
            current_soc += charge
            actions[i] = 1
            energy[i] = charge
            profit[i] = -grid_energy * prices[i]
            
        elif sell_signals[i] and current_soc > 0:
            # Discharge
            discharge = min(max_energy, current_soc)
            grid_energy = discharge * eff_factor
            current_soc -= discharge
            actions[i] = -1
            energy[i] = discharge
            profit[i] = grid_energy * prices[i]
        
        soc[i] = current_soc
    
    cumulative_profit = np.cumsum(profit)
    
    return {
        'soc': soc,
        'actions': actions,
        'energy': energy,
        'profit': profit,
        'cumulative_profit': cumulative_profit
    }


def run_all_strategies(
    df: pd.DataFrame,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90
) -> Dict[str, pd.DataFrame]:
    """
    Run all strategies and return comparison results.
    
    Args:
        df: Price DataFrame
        capacity_mwh: Battery capacity
        power_mw: Max power
        efficiency: Round-trip efficiency
        
    Returns:
        Dict mapping strategy name to result DataFrame
    """
    from strategies.perfect_foresight import run_perfect_foresight
    from strategies.greedy import run_greedy_strategy
    from strategies.sliding_window import run_sliding_window_strategy
    
    results = {}
    timings = {}
    
    # Perfect Foresight
    start = time.perf_counter()
    results['perfect_foresight'] = run_perfect_foresight(df, capacity_mwh, power_mw, efficiency)
    timings['perfect_foresight'] = time.perf_counter() - start
    
    # Greedy
    start = time.perf_counter()
    greedy_result, _ = run_greedy_strategy(df, capacity_mwh, power_mw, efficiency)
    results['greedy'] = greedy_result
    timings['greedy'] = time.perf_counter() - start
    
    # Sliding Window
    start = time.perf_counter()
    results['sliding_window'] = run_sliding_window_strategy(df, capacity_mwh, power_mw, efficiency)
    timings['sliding_window'] = time.perf_counter() - start
    
    return results, timings


def benchmark_performance(
    df: pd.DataFrame,
    iterations: int = 10
) -> pd.DataFrame:
    """
    Benchmark strategy performance.
    
    Args:
        df: Test data
        iterations: Number of runs for timing
        
    Returns:
        DataFrame with benchmark results
    """
    prices = df['RRP'].values
    n = len(prices)
    
    results = []
    
    # Vectorized greedy
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        vectorized_greedy_simulation(prices)
        times.append(time.perf_counter() - start)
    
    results.append({
        'strategy': 'vectorized_greedy',
        'avg_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'intervals_per_second': n / np.mean(times)
    })
    
    # Regular strategies
    from strategies.greedy import run_greedy_strategy
    from strategies.sliding_window import run_sliding_window_strategy
    from strategies.perfect_foresight import run_perfect_foresight
    
    for name, func in [
        ('greedy', lambda: run_greedy_strategy(df)),
        ('sliding_window', lambda: run_sliding_window_strategy(df)),
        ('perfect_foresight', lambda: run_perfect_foresight(df))
    ]:
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            times.append(time.perf_counter() - start)
        
        results.append({
            'strategy': name,
            'avg_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'intervals_per_second': n / np.mean(times)
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add src to path
    src_path = Path(__file__).parent
    sys.path.insert(0, str(src_path))
    
    from data_loader import load_dispatch_data
    
    data_path = src_path.parent / "data" / "combined_dispatch_prices.csv"
    
    if data_path.exists():
        df = load_dispatch_data(str(data_path))
        print(f"Loaded {len(df)} price intervals")
        
        # Quick vectorized test
        prices = df['RRP'].values
        start = time.perf_counter()
        result = vectorized_greedy_simulation(prices)
        elapsed = time.perf_counter() - start
        
        print(f"\nVectorized Greedy Simulation:")
        print(f"  Time: {elapsed*1000:.2f} ms")
        print(f"  Throughput: {len(prices)/elapsed:,.0f} intervals/second")
        print(f"  Final profit: ${result['cumulative_profit'][-1]:,.2f}")
        
        # Full benchmark
        print("\nRunning full benchmark...")
        bench = benchmark_performance(df, iterations=5)
        print(bench.to_string(index=False))
    else:
        print(f"Data not found: {data_path}")
