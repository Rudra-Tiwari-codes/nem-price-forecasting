"""
Evaluation metrics for trading strategy performance.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_baseline(
    df: pd.DataFrame, 
    capacity_mwh: float = 100.0, 
    power_mw: float = 50.0, 
    efficiency: float = 0.90
) -> float:
    """Simple daily charge/discharge baseline. Charge at lowest, discharge at highest."""
    if 'SETTLEMENTDATE' not in df.columns or 'RRP' not in df.columns:
        raise ValueError("DataFrame must contain 'SETTLEMENTDATE' and 'RRP' columns")
    
    df = df.copy()
    df['date'] = df['SETTLEMENTDATE'].dt.date
    df['hour'] = df['SETTLEMENTDATE'].dt.hour
    
    interval_hours = 5/60
    max_energy = power_mw * interval_hours
    efficiency_factor = np.sqrt(efficiency)
    total_profit = 0.0
    
    for date, group in df.groupby('date'):
        if len(group) < 12:
            continue
        
        min_idx = group['RRP'].idxmin()
        max_idx = group['RRP'].idxmax()
        buy_price = group.loc[min_idx, 'RRP']
        sell_price = group.loc[max_idx, 'RRP']
        energy = min(capacity_mwh, max_energy * 12)
        cost = (energy / efficiency_factor) * buy_price
        revenue = (energy * efficiency_factor) * sell_price
        total_profit += revenue - cost
    
    return float(total_profit)


def profit_vs_baseline(strategy_profit: float, baseline_profit: float) -> Dict[str, float]:
    """Compare strategy profit to baseline."""
    if baseline_profit == 0:
        return {
            'strategy_profit': strategy_profit,
            'baseline_profit': baseline_profit,
            'absolute_diff': strategy_profit,
            'ratio': float('inf') if strategy_profit > 0 else 0.0,
            'improvement_pct': float('inf') if strategy_profit > 0 else 0.0
        }
    
    return {
        'strategy_profit': strategy_profit,
        'baseline_profit': baseline_profit,
        'absolute_diff': strategy_profit - baseline_profit,
        'ratio': strategy_profit / baseline_profit,
        'improvement_pct': (strategy_profit - baseline_profit) / abs(baseline_profit) * 100
    }


def sharpe_ratio(
    returns: np.ndarray, 
    risk_free_rate: float = 0.0, 
    periods_per_year: int = 365 * 288
) -> float:
    """Calculate annualized Sharpe ratio. Assumes 5-minute intervals by default."""
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    std = returns.std()
    if std == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    return float(np.sqrt(periods_per_year) * excess_returns.mean() / std)


def win_rate(result_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate precision of buy/sell decisions."""
    if 'action' not in result_df.columns or 'profit' not in result_df.columns:
        raise ValueError("DataFrame must contain 'action' and 'profit' columns")
    
    charges = result_df[result_df['action'] == 'charge']
    discharges = result_df[result_df['action'] == 'discharge']
    
    total_trades = len(charges) + len(discharges)
    profitable_trades = int((result_df['profit'] > 0).sum())
    
    if total_trades == 0:
        return {'win_rate': 0.0, 'total_trades': 0, 'profitable_trades': 0, 
                'charge_count': 0, 'discharge_count': 0}
    
    return {
        'win_rate': profitable_trades / total_trades * 100,
        'profitable_trades': profitable_trades,
        'total_trades': total_trades,
        'charge_count': len(charges),
        'discharge_count': len(discharges)
    }


def profit_factor(result_df: pd.DataFrame) -> float:
    """Ratio of gross profit to gross loss."""
    if 'profit' not in result_df.columns:
        raise ValueError("DataFrame must contain 'profit' column")
    
    profits = result_df['profit']
    gross_profit = profits[profits > 0].sum()
    gross_loss = abs(profits[profits < 0].sum())
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return float(gross_profit / gross_loss)


def max_drawdown(cumulative_profits: np.ndarray) -> float:
    """Maximum peak to trough decline."""
    cumulative = np.array(cumulative_profits)
    if len(cumulative) == 0:
        return 0.0
    
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(drawdown.max())


def generate_metrics_report(
    results_dict: Dict[str, pd.DataFrame], 
    df: pd.DataFrame, 
    capacity_mwh: float = 100.0, 
    power_mw: float = 50.0, 
    efficiency: float = 0.90
) -> Dict[str, Dict[str, float]]:
    """Generate comprehensive metrics for all strategies."""
    try:
        baseline = calculate_baseline(df, capacity_mwh, power_mw, efficiency)
    except Exception as e:
        logger.warning(f"Failed to calculate baseline: {e}")
        baseline = 0.0
    
    print("Strategy Performance Metrics")
    print("=" * 70)
    print(f"\nBaseline (daily min/max): ${baseline:,.2f}")
    print()
    
    header = f"{'Strategy':<20} {'Profit':<12} {'vs Base':<10} {'Sharpe':<8} {'Win%':<8} {'MaxDD':<10}"
    print(header)
    print("-" * 70)
    
    metrics_data: Dict[str, Dict[str, float]] = {}
    
    for name, result_df in results_dict.items():
        try:
            strategy_profit = float(result_df['cumulative_profit'].iloc[-1])
            returns = result_df['profit'].values
            sr = sharpe_ratio(returns)
            wr = win_rate(result_df)
            mdd = max_drawdown(result_df['cumulative_profit'].values)
            vs_base = profit_vs_baseline(strategy_profit, baseline)
            
            display_name = name.replace('_', ' ').title()
            print(f"{display_name:<20} ${strategy_profit:<11,.0f} {vs_base['ratio']:<9.2f}x {sr:<7.2f} {wr['win_rate']:<7.1f}% ${mdd:<9,.0f}")
            
            metrics_data[name] = {
                'profit': strategy_profit,
                'baseline_ratio': vs_base['ratio'],
                'sharpe': sr,
                'win_rate': wr['win_rate'],
                'max_drawdown': mdd,
                'profit_factor': profit_factor(result_df)
            }
        except Exception as e:
            logger.warning(f"Failed to calculate metrics for {name}: {e}")
            print(f"{name:<20} Error: {e}")
    
    print()
    return metrics_data


def test_metrics() -> None:
    """Test metrics on sample data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_dispatch_data
    from strategies.greedy import run_greedy_strategy
    from strategies.perfect_foresight import run_perfect_foresight
    
    data_path = Path(__file__).parent.parent / "data" / "combined_dispatch_prices.csv"
    
    try:
        df = load_dispatch_data(str(data_path), regions=['SA1'])
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        print(f"Error loading data: {e}")
        return
    
    results: Dict[str, pd.DataFrame] = {}
    
    try:
        results['greedy'], _ = run_greedy_strategy(df)
        results['perfect_foresight'] = run_perfect_foresight(df)
    except Exception as e:
        logger.error(f"Failed to run strategies: {e}")
        print(f"Error running strategies: {e}")
        return
    
    generate_metrics_report(results, df)


if __name__ == "__main__":
    test_metrics()
