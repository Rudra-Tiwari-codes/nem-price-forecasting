"""
NEM Arbitrage Engine - Main Entry Point

Runs complete arbitrage simulation with all strategies,
generates visualizations, and outputs performance report.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data_loader import load_dispatch_data, get_price_statistics, validate_data
from battery import Battery
from strategies.perfect_foresight import run_perfect_foresight
from strategies.greedy import run_greedy_strategy, optimize_thresholds
from strategies.sliding_window import run_sliding_window_strategy, optimize_window_size
from strategies.dynamic_programming import run_dp_strategy
from visualizer import generate_all_charts, plot_strategy_comparison
import time


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text: str):
    """Print section divider."""
    print(f"\n--- {text} ---")


def run_simulation(
    data_path: str = "data/combined_dispatch_prices.csv",
    region: str = None,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90,
    generate_charts: bool = True
):
    """
    Run complete arbitrage simulation.
    
    Args:
        data_path: Path to AEMO data CSV
        region: Region to simulate (None for all)
        capacity_mwh: Battery capacity
        power_mw: Battery power rating
        efficiency: Round-trip efficiency
        generate_charts: Whether to generate visualization charts
    """
    print_header("NEM ARBITRAGE ENGINE")
    print(f"\nBattery Configuration:")
    print(f"  Capacity: {capacity_mwh} MWh")
    print(f"  Power Rating: {power_mw} MW")
    print(f"  Efficiency: {efficiency:.0%}")
    print(f"  C-Rate: {capacity_mwh/power_mw:.1f} hours")
    
    # Load data
    print_section("Loading Data")
    df = load_dispatch_data(data_path, regions=[region] if region else None)
    
    validation = validate_data(df)
    print(f"  Rows loaded: {validation['row_count']:,}")
    print(f"  Date range: {validation['date_range'][0]} to {validation['date_range'][1]}")
    print(f"  Valid: {'Yes' if validation['valid'] else 'No'}")
    
    if 'REGIONID' in df.columns:
        regions = df['REGIONID'].unique()
        print(f"  Regions: {', '.join(regions)}")
    
    # Price statistics
    print_section("Price Statistics")
    stats = get_price_statistics(df)
    print(f"  Mean:   ${stats['mean']:.2f}/MWh")
    print(f"  Median: ${stats['median']:.2f}/MWh")
    print(f"  Std:    ${stats['std']:.2f}/MWh")
    print(f"  Min:    ${stats['min']:.2f}/MWh")
    print(f"  Max:    ${stats['max']:.2f}/MWh")
    print(f"  Negative prices: {stats['negative_percent']:.1f}%")
    print(f"  Spike prices (>$300): {stats['spike_percent']:.1f}%")
    
    # Run strategies
    print_section("Running Strategies")
    results = {}
    timings = {}
    
    # Perfect Foresight
    print("  Running Perfect Foresight...")
    start = time.perf_counter()
    results['perfect_foresight'] = run_perfect_foresight(df, capacity_mwh, power_mw, efficiency)
    timings['perfect_foresight'] = time.perf_counter() - start
    
    # Greedy
    print("  Running Greedy Strategy...")
    start = time.perf_counter()
    results['greedy'], thresholds = run_greedy_strategy(df, capacity_mwh, power_mw, efficiency)
    timings['greedy'] = time.perf_counter() - start
    
    # Sliding Window
    print("  Running Sliding Window...")
    start = time.perf_counter()
    results['sliding_window'] = run_sliding_window_strategy(df, capacity_mwh, power_mw, efficiency)
    timings['sliding_window'] = time.perf_counter() - start
    
    # Dynamic Programming
    print("  Running Dynamic Programming...")
    start = time.perf_counter()
    results['dynamic_programming'] = run_dp_strategy(df, capacity_mwh, power_mw, efficiency)
    timings['dynamic_programming'] = time.perf_counter() - start
    
    # Results summary
    print_section("Results Summary")
    print("\n  Strategy              | Total Profit  | Cycles | Time (ms)")
    print("  " + "-" * 58)
    
    for name, result_df in results.items():
        profit = result_df['cumulative_profit'].iloc[-1]
        charges = (result_df['action'] == 'charge').sum()
        discharges = (result_df['action'] == 'discharge').sum()
        time_ms = timings[name] * 1000
        
        display_name = name.replace('_', ' ').title()
        print(f"  {display_name:<22} | ${profit:>11,.0f} | {charges:>3}/{discharges:<3} | {time_ms:>7.1f}")
    
    # Best result
    best_name = max(results.keys(), key=lambda k: results[k]['cumulative_profit'].iloc[-1])
    best_profit = results[best_name]['cumulative_profit'].iloc[-1]
    
    print(f"\n  Best Strategy: {best_name.replace('_', ' ').title()}")
    print(f"     Total Profit: ${best_profit:,.2f}")
    
    # Calculate annualized ROI
    date_range = df['SETTLEMENTDATE'].max() - df['SETTLEMENTDATE'].min()
    days = date_range.days if hasattr(date_range, 'days') else 1
    annualized = (best_profit / max(days, 1)) * 365
    print(f"     Annualized: ${annualized:,.0f}/year")
    
    # Generate charts
    if generate_charts:
        print_section("Generating Charts")
        try:
            chart_dir = Path(data_path).parent.parent / "charts"
            saved = generate_all_charts(df, results, str(chart_dir))
            print(f"  Saved {len(saved)} charts to {chart_dir}/")
            for path in saved:
                print(f"    - {Path(path).name}")
        except Exception as e:
            print(f"  Warning: Could not generate charts: {e}")
    
    print_header("SIMULATION COMPLETE")
    
    return results, stats


def main():
    """Main entry point with command line support."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NEM Arbitrage Engine - Battery Trading Simulator"
    )
    parser.add_argument(
        "--data", "-d",
        default="data/combined_dispatch_prices.csv",
        help="Path to AEMO price data CSV"
    )
    parser.add_argument(
        "--region", "-r",
        default=None,
        help="Region to simulate (e.g., SA1, VIC1)"
    )
    parser.add_argument(
        "--capacity", "-c",
        type=float, default=100.0,
        help="Battery capacity in MWh"
    )
    parser.add_argument(
        "--power", "-p",
        type=float, default=50.0,
        help="Battery power rating in MW"
    )
    parser.add_argument(
        "--efficiency", "-e",
        type=float, default=0.90,
        help="Round-trip efficiency (0.0-1.0)"
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation"
    )
    
    args = parser.parse_args()
    
    run_simulation(
        data_path=args.data,
        region=args.region,
        capacity_mwh=args.capacity,
        power_mw=args.power,
        efficiency=args.efficiency,
        generate_charts=not args.no_charts
    )


if __name__ == "__main__":
    main()
