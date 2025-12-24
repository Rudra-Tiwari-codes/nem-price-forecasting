"""
Evaluation metrics for trading strategy performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def calculate_baseline(df, capacity_mwh=100.0, power_mw=50.0, efficiency=0.90):
    """
    Simple daily charge/discharge baseline.
    Charge during lowest price hour, discharge during highest.
    """
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
    
    return total_profit


def profit_vs_baseline(strategy_profit, baseline_profit):
    """Compare strategy profit to baseline."""
    if baseline_profit == 0:
        return {'absolute': strategy_profit, 'ratio': np.inf, 'improvement': np.inf}
    
    return {
        'strategy_profit': strategy_profit,
        'baseline_profit': baseline_profit,
        'absolute_diff': strategy_profit - baseline_profit,
        'ratio': strategy_profit / baseline_profit,
        'improvement_pct': (strategy_profit - baseline_profit) / abs(baseline_profit) * 100
    }


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=365*288):
    """
    Calculate annualized Sharpe ratio.
    Assumes 5-minute intervals by default.
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()


def win_rate(result_df):
    """Calculate precision of buy/sell decisions."""
    charges = result_df[result_df['action'] == 'charge']
    discharges = result_df[result_df['action'] == 'discharge']
    
    total_trades = len(charges) + len(discharges)
    profitable_trades = (result_df['profit'] > 0).sum()
    
    if total_trades == 0:
        return {'win_rate': 0, 'total_trades': 0}
    
    return {
        'win_rate': profitable_trades / total_trades * 100,
        'profitable_trades': profitable_trades,
        'total_trades': total_trades,
        'charge_count': len(charges),
        'discharge_count': len(discharges)
    }


def profit_factor(result_df):
    """Ratio of gross profit to gross loss."""
    profits = result_df['profit']
    gross_profit = profits[profits > 0].sum()
    gross_loss = abs(profits[profits < 0].sum())
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0
    
    return gross_profit / gross_loss


def max_drawdown(cumulative_profits):
    """Maximum peak to trough decline."""
    cumulative = np.array(cumulative_profits)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return drawdown.max()


def generate_metrics_report(results_dict, df, capacity_mwh=100.0, power_mw=50.0, efficiency=0.90):
    """Generate comprehensive metrics for all strategies."""
    baseline = calculate_baseline(df, capacity_mwh, power_mw, efficiency)
    
    print("Strategy Performance Metrics")
    print("=" * 70)
    print(f"\nBaseline (daily min/max): ${baseline:,.2f}")
    print()
    
    header = f"{'Strategy':<20} {'Profit':<12} {'vs Base':<10} {'Sharpe':<8} {'Win%':<8} {'MaxDD':<10}"
    print(header)
    print("-" * 70)
    
    metrics_data = {}
    
    for name, result_df in results_dict.items():
        strategy_profit = result_df['cumulative_profit'].iloc[-1]
        
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
    
    print()
    return metrics_data


def test_metrics():
    """Test metrics on sample data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_dispatch_data
    from strategies.greedy import run_greedy_strategy
    from strategies.perfect_foresight import run_perfect_foresight
    
    data_path = Path(__file__).parent.parent / "data" / "combined_dispatch_prices.csv"
    df = load_dispatch_data(str(data_path), regions=['SA1'])
    
    results = {}
    results['greedy'], _ = run_greedy_strategy(df)
    results['perfect_foresight'] = run_perfect_foresight(df)
    
    generate_metrics_report(results, df)


if __name__ == "__main__":
    test_metrics()
