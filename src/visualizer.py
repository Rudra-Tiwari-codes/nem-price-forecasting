"""
Visualization module for NEM Arbitrage results.

Creates compelling charts for price analysis and simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional, Dict, List


# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_price_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot price distribution histogram with key statistics.
    
    Args:
        df: DataFrame with RRP column
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    prices = df['RRP'].values
    
    # Main histogram
    ax1 = axes[0]
    ax1.hist(prices, bins=100, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='$0/MWh')
    ax1.axvline(x=np.median(prices), color='orange', linestyle='-', linewidth=2, 
                label=f'Median: ${np.median(prices):.0f}')
    ax1.set_xlabel('Price ($/MWh)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Price Distribution (Full Range)', fontsize=14, fontweight='bold')
    ax1.legend()
    
    # Zoomed histogram (exclude extreme spikes)
    ax2 = axes[1]
    filtered = prices[(prices > -100) & (prices < 500)]
    ax2.hist(filtered, bins=80, color='seagreen', edgecolor='white', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=np.percentile(prices, 25), color='purple', linestyle=':', 
                label=f'25th: ${np.percentile(prices, 25):.0f}')
    ax2.axvline(x=np.percentile(prices, 75), color='purple', linestyle=':', 
                label=f'75th: ${np.percentile(prices, 75):.0f}')
    ax2.set_xlabel('Price ($/MWh)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Price Distribution (Zoomed -$100 to $500)', fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_battery_operation(
    result_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Plot battery operation with price, SoC, and actions.
    
    Args:
        result_df: Simulation result DataFrame
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Ensure we have datetime index
    if 'SETTLEMENTDATE' in result_df.columns:
        x = pd.to_datetime(result_df['SETTLEMENTDATE'])
    else:
        x = range(len(result_df))
    
    # Price plot
    ax1 = axes[0]
    ax1.plot(x, result_df['RRP'], color='steelblue', linewidth=0.8, alpha=0.8)
    ax1.fill_between(x, result_df['RRP'], alpha=0.3, color='steelblue')
    ax1.set_ylabel('Price ($/MWh)', fontsize=11)
    ax1.set_title('Electricity Price', fontsize=13, fontweight='bold')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Mark charge/discharge points
    if 'action' in result_df.columns:
        charge_mask = result_df['action'] == 'charge'
        discharge_mask = result_df['action'] == 'discharge'
        
        if isinstance(x, pd.Series):
            charge_x = x[charge_mask]
            discharge_x = x[discharge_mask]
        else:
            charge_x = np.array(x)[charge_mask]
            discharge_x = np.array(x)[discharge_mask]
        
        ax1.scatter(charge_x, result_df.loc[charge_mask, 'RRP'], 
                   color='green', s=20, alpha=0.7, label='Charge', zorder=5)
        ax1.scatter(discharge_x, result_df.loc[discharge_mask, 'RRP'], 
                   color='red', s=20, alpha=0.7, label='Discharge', zorder=5)
        ax1.legend(loc='upper right')
    
    # SoC plot
    ax2 = axes[1]
    ax2.fill_between(x, result_df['soc'], color='orange', alpha=0.6)
    ax2.plot(x, result_df['soc'], color='darkorange', linewidth=1)
    ax2.set_ylabel('State of Charge (MWh)', fontsize=11)
    ax2.set_title('Battery State of Charge', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, result_df['soc'].max() * 1.1 if result_df['soc'].max() > 0 else 100)
    
    # Cumulative profit plot
    ax3 = axes[2]
    ax3.plot(x, result_df['cumulative_profit'], color='green', linewidth=1.5)
    ax3.fill_between(x, result_df['cumulative_profit'], alpha=0.3, color='green')
    ax3.set_ylabel('Cumulative Profit ($)', fontsize=11)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_title('Cumulative Profit', fontsize=13, fontweight='bold')
    
    # Format x-axis for datetime
    if isinstance(x, pd.Series):
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_strategy_comparison(
    results: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Compare multiple strategy results.
    
    Args:
        results: Dict mapping strategy name to result DataFrame
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    # Profit curves
    ax1 = axes[0]
    for i, (name, df) in enumerate(results.items()):
        if 'SETTLEMENTDATE' in df.columns:
            x = pd.to_datetime(df['SETTLEMENTDATE'])
        else:
            x = range(len(df))
        
        label = name.replace('_', ' ').title()
        ax1.plot(x, df['cumulative_profit'], label=label, 
                color=colors[i % len(colors)], linewidth=2)
    
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Cumulative Profit ($)', fontsize=11)
    ax1.set_title('Strategy Profit Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    if 'SETTLEMENTDATE' in list(results.values())[0].columns:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Final profit bar chart
    ax2 = axes[1]
    names = [name.replace('_', ' ').title() for name in results.keys()]
    profits = [df['cumulative_profit'].iloc[-1] for df in results.values()]
    
    bars = ax2.bar(names, profits, color=colors[:len(names)], edgecolor='white', linewidth=2)
    ax2.set_ylabel('Total Profit ($)', fontsize=11)
    ax2.set_title('Final Profit by Strategy', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, profit in zip(bars, profits):
        height = bar.get_height()
        ax2.annotate(f'${profit:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_region_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Compare price volatility and opportunities across regions.
    
    Args:
        df: DataFrame with REGIONID and RRP columns
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if 'REGIONID' not in df.columns:
        raise ValueError("DataFrame must have REGIONID column")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    regions = df['REGIONID'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(regions)))
    
    # Box plot
    ax1 = axes[0]
    region_data = [df[df['REGIONID'] == r]['RRP'].values for r in regions]
    bp = ax1.boxplot(region_data, labels=regions, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Price ($/MWh)', fontsize=11)
    ax1.set_title('Price Distribution by Region', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Volatility comparison
    ax2 = axes[1]
    stats = []
    for region in regions:
        region_prices = df[df['REGIONID'] == region]['RRP']
        stats.append({
            'region': region,
            'std': region_prices.std(),
            'range': region_prices.max() - region_prices.min(),
            'negative_pct': (region_prices < 0).mean() * 100
        })
    
    stats_df = pd.DataFrame(stats)
    x = np.arange(len(regions))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, stats_df['std'], width, label='Std Dev', color='steelblue')
    ax2.set_ylabel('Standard Deviation ($/MWh)', fontsize=11)
    
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, stats_df['negative_pct'], width, 
                         label='Negative %', color='coral')
    ax2_twin.set_ylabel('Negative Price %', fontsize=11)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions)
    ax2.set_title('Volatility Metrics by Region', fontsize=14, fontweight='bold')
    
    # Combined legend
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_all_charts(
    df: pd.DataFrame,
    results: Dict[str, pd.DataFrame],
    output_dir: str = 'charts'
) -> List[str]:
    """
    Generate all charts and save to output directory.
    
    Args:
        df: Price DataFrame
        results: Strategy results dict
        output_dir: Directory to save charts
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    saved_files = []
    
    # Price distribution
    path = str(output_path / 'price_distribution.png')
    plot_price_distribution(df, save_path=path)
    saved_files.append(path)
    plt.close()
    
    # Strategy comparison
    path = str(output_path / 'strategy_comparison.png')
    plot_strategy_comparison(results, save_path=path)
    saved_files.append(path)
    plt.close()
    
    # Battery operation for best strategy
    best_strategy = max(results.items(), 
                       key=lambda x: x[1]['cumulative_profit'].iloc[-1])
    path = str(output_path / f'battery_operation_{best_strategy[0]}.png')
    plot_battery_operation(best_strategy[1], save_path=path)
    saved_files.append(path)
    plt.close()
    
    # Region comparison (if multiple regions)
    if 'REGIONID' in df.columns and df['REGIONID'].nunique() > 1:
        path = str(output_path / 'region_comparison.png')
        plot_region_comparison(df, save_path=path)
        saved_files.append(path)
        plt.close()
    
    return saved_files


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    src_path = Path(__file__).parent
    sys.path.insert(0, str(src_path))
    
    from data_loader import load_dispatch_data
    from strategies.greedy import run_greedy_strategy
    from strategies.sliding_window import run_sliding_window_strategy
    from strategies.perfect_foresight import run_perfect_foresight
    
    data_path = src_path.parent / "data" / "combined_dispatch_prices.csv"
    
    if data_path.exists():
        df = load_dispatch_data(str(data_path), regions=['SA1'])
        print(f"Loaded {len(df)} price intervals")
        
        # Run strategies
        print("Running strategies...")
        results = {}
        results['perfect_foresight'] = run_perfect_foresight(df)
        results['greedy'], _ = run_greedy_strategy(df)
        results['sliding_window'] = run_sliding_window_strategy(df)
        
        # Generate charts
        print("Generating charts...")
        chart_dir = src_path.parent / "charts"
        saved = generate_all_charts(df, results, str(chart_dir))
        
        print(f"\nSaved {len(saved)} charts:")
        for path in saved:
            print(f"  - {path}")
    else:
        print(f"Data not found: {data_path}")
