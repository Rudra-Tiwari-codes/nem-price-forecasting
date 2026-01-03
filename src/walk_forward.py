"""
Walk-Forward Backtesting Module.

Implements realistic out-of-sample testing where strategies are evaluated
on data they haven't seen during training/optimization.

This gives more realistic performance estimates compared to full-history backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtesting."""
    train_days: int = 7  # Days of data for training/optimization
    test_days: int = 1   # Days of data for out-of-sample testing
    step_days: int = 1   # How many days to step forward each iteration
    min_train_samples: int = 288  # Minimum training samples (288 = 1 day at 5-min intervals)


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward window."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_profit: float
    test_profit: float
    test_trades: int
    params: Optional[dict] = None


def walk_forward_split(
    df: pd.DataFrame,
    config: WalkForwardConfig
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate train/test splits for walk-forward validation.
    
    Args:
        df: DataFrame with SETTLEMENTDATE column
        config: Walk-forward configuration
        
    Returns:
        List of (train_df, test_df) tuples
    """
    if 'SETTLEMENTDATE' not in df.columns:
        raise ValueError("DataFrame must have SETTLEMENTDATE column")
    
    df = df.sort_values('SETTLEMENTDATE').reset_index(drop=True)
    
    min_date = df['SETTLEMENTDATE'].min()
    max_date = df['SETTLEMENTDATE'].max()
    
    splits = []
    current_train_start = min_date
    
    while True:
        train_end = current_train_start + timedelta(days=config.train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=config.test_days)
        
        # Check if we have enough data for this window
        if test_end > max_date:
            break
        
        # Get train and test data
        train_mask = (df['SETTLEMENTDATE'] >= current_train_start) & (df['SETTLEMENTDATE'] < train_end)
        test_mask = (df['SETTLEMENTDATE'] >= test_start) & (df['SETTLEMENTDATE'] < test_end)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        # Check minimum samples
        if len(train_df) >= config.min_train_samples and len(test_df) > 0:
            splits.append((train_df, test_df))
        
        # Step forward
        current_train_start += timedelta(days=config.step_days)
    
    return splits


def run_walk_forward_backtest(
    df: pd.DataFrame,
    strategy_fn: Callable,
    config: WalkForwardConfig = None,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90,
    optimize_fn: Callable = None
) -> Dict:
    """
    Run walk-forward backtesting on a strategy.
    
    Args:
        df: Full price data DataFrame
        strategy_fn: Strategy function that takes (df, capacity, power, efficiency, **params)
                    and returns a result DataFrame with 'cumulative_profit' column
        config: Walk-forward configuration
        capacity_mwh: Battery capacity
        power_mw: Battery power rating
        efficiency: Round-trip efficiency
        optimize_fn: Optional function to optimize parameters on training data
                    Takes (train_df, capacity, power, efficiency) and returns dict of params
    
    Returns:
        Dictionary with aggregated results and per-window details
    """
    if config is None:
        config = WalkForwardConfig()
    
    splits = walk_forward_split(df, config)
    
    if not splits:
        raise ValueError("Not enough data for walk-forward testing with current config")
    
    logger.info(f"Running walk-forward with {len(splits)} windows")
    
    results: List[WalkForwardResult] = []
    all_test_profits = []
    all_train_profits = []
    
    for i, (train_df, test_df) in enumerate(splits):
        # Optionally optimize parameters on training data
        params = {}
        if optimize_fn is not None:
            try:
                params = optimize_fn(train_df, capacity_mwh, power_mw, efficiency)
            except Exception as e:
                logger.warning(f"Optimization failed for window {i}: {e}")
        
        # Run strategy on training data (in-sample)
        try:
            train_result = strategy_fn(train_df, capacity_mwh, power_mw, efficiency, **params)
            train_profit = train_result['cumulative_profit'].iloc[-1]
        except Exception as e:
            logger.warning(f"Training run failed for window {i}: {e}")
            train_profit = 0.0
        
        # Run strategy on test data (out-of-sample)
        try:
            test_result = strategy_fn(test_df, capacity_mwh, power_mw, efficiency, **params)
            test_profit = test_result['cumulative_profit'].iloc[-1]
            test_trades = (test_result['action'] != 'hold').sum()
        except Exception as e:
            logger.warning(f"Test run failed for window {i}: {e}")
            test_profit = 0.0
            test_trades = 0
        
        result = WalkForwardResult(
            train_start=train_df['SETTLEMENTDATE'].min(),
            train_end=train_df['SETTLEMENTDATE'].max(),
            test_start=test_df['SETTLEMENTDATE'].min(),
            test_end=test_df['SETTLEMENTDATE'].max(),
            train_profit=train_profit,
            test_profit=test_profit,
            test_trades=test_trades,
            params=params if params else None
        )
        results.append(result)
        all_test_profits.append(test_profit)
        all_train_profits.append(train_profit)
    
    # Aggregate statistics
    total_test_profit = sum(all_test_profits)
    total_train_profit = sum(all_train_profits)
    
    # Calculate performance degradation (how much worse out-of-sample vs in-sample)
    if total_train_profit > 0:
        degradation = 1 - (total_test_profit / total_train_profit)
    else:
        degradation = 0.0
    
    return {
        'total_test_profit': total_test_profit,
        'total_train_profit': total_train_profit,
        'degradation_pct': degradation * 100,
        'num_windows': len(results),
        'avg_test_profit_per_window': np.mean(all_test_profits),
        'std_test_profit': np.std(all_test_profits),
        'win_rate': sum(1 for p in all_test_profits if p > 0) / len(all_test_profits) * 100,
        'max_test_profit': max(all_test_profits),
        'min_test_profit': min(all_test_profits),
        'windows': results,
        'config': config
    }


def compare_strategies_walk_forward(
    df: pd.DataFrame,
    strategies: Dict[str, Callable],
    config: WalkForwardConfig = None,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90
) -> pd.DataFrame:
    """
    Compare multiple strategies using walk-forward backtesting.
    
    Args:
        df: Price data
        strategies: Dict mapping strategy name to strategy function
        config: Walk-forward configuration
        capacity_mwh: Battery capacity
        power_mw: Battery power rating  
        efficiency: Round-trip efficiency
        
    Returns:
        DataFrame comparing strategy performance
    """
    results = []
    
    for name, strategy_fn in strategies.items():
        logger.info(f"Running walk-forward for {name}...")
        try:
            wf_result = run_walk_forward_backtest(
                df, strategy_fn, config,
                capacity_mwh, power_mw, efficiency
            )
            results.append({
                'strategy': name,
                'total_test_profit': wf_result['total_test_profit'],
                'total_train_profit': wf_result['total_train_profit'],
                'degradation_pct': wf_result['degradation_pct'],
                'win_rate_pct': wf_result['win_rate'],
                'avg_profit_per_window': wf_result['avg_test_profit_per_window'],
                'std_profit': wf_result['std_test_profit'],
                'num_windows': wf_result['num_windows']
            })
        except Exception as e:
            logger.error(f"Walk-forward failed for {name}: {e}")
            results.append({
                'strategy': name,
                'total_test_profit': 0,
                'total_train_profit': 0,
                'degradation_pct': 0,
                'win_rate_pct': 0,
                'avg_profit_per_window': 0,
                'std_profit': 0,
                'num_windows': 0
            })
    
    return pd.DataFrame(results).sort_values('total_test_profit', ascending=False)


def generate_walk_forward_report(
    df: pd.DataFrame,
    config: WalkForwardConfig = None,
    capacity_mwh: float = 100.0,
    power_mw: float = 50.0,
    efficiency: float = 0.90
) -> Dict:
    """
    Generate a comprehensive walk-forward report for all strategies.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from strategies.greedy import run_greedy_strategy
    from strategies.sliding_window import run_sliding_window_strategy
    from strategies.perfect_foresight import run_perfect_foresight
    from forecasting import run_forecast_strategy, EMAPredictor
    
    # Wrap strategies to match expected signature
    def greedy_wrapper(df, cap, pow, eff, **kwargs):
        result, _ = run_greedy_strategy(df, cap, pow, eff)
        return result
    
    def sliding_window_wrapper(df, cap, pow, eff, **kwargs):
        return run_sliding_window_strategy(df, cap, pow, eff)
    
    def perfect_foresight_wrapper(df, cap, pow, eff, **kwargs):
        return run_perfect_foresight(df, cap, pow, eff)
    
    def ema_wrapper(df, cap, pow, eff, **kwargs):
        predictor = EMAPredictor(span=12)
        return run_forecast_strategy(df, predictor, cap, pow, eff)
    
    strategies = {
        'Greedy': greedy_wrapper,
        'Sliding Window': sliding_window_wrapper,
        'EMA Forecast': ema_wrapper,
        # Note: Perfect Foresight is not realistic for walk-forward as it uses future data
    }
    
    print("\n" + "=" * 70)
    print("  WALK-FORWARD BACKTESTING REPORT")
    print("=" * 70)
    
    if config is None:
        config = WalkForwardConfig()
    
    print(f"\nConfiguration:")
    print(f"  Training window: {config.train_days} days")
    print(f"  Test window: {config.test_days} days")
    print(f"  Step size: {config.step_days} days")
    
    comparison = compare_strategies_walk_forward(
        df, strategies, config,
        capacity_mwh, power_mw, efficiency
    )
    
    print("\nResults (Out-of-Sample Performance):")
    print("-" * 70)
    print(comparison.to_string(index=False))
    
    return {
        'comparison': comparison,
        'config': config
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_dispatch_data
    
    data_path = Path(__file__).parent.parent / "data" / "combined_dispatch_prices.csv"
    
    if data_path.exists():
        df = load_dispatch_data(str(data_path), regions=['SA1'])
        print(f"Loaded {len(df)} price intervals for SA1")
        
        # Run walk-forward analysis
        generate_walk_forward_report(df)
    else:
        print(f"Data not found: {data_path}")
