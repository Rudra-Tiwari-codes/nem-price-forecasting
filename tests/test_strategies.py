"""
Unit tests for trading strategies.

Tests the core trading strategies to ensure correct behavior.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_price_df():
    """Create sample price data for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='5min')
    # Create prices with some volatility
    prices = 100 + np.cumsum(np.random.randn(n) * 10)
    # Add some spikes and dips
    prices[20] = 300  # spike
    prices[40] = -20  # negative price
    prices[60] = 500  # big spike
    prices[80] = 50   # low price

    return pd.DataFrame({
        'SETTLEMENTDATE': dates,
        'RRP': prices,
        'REGIONID': 'SA1'
    })


class TestBattery:
    """Tests for Battery class."""

    def test_battery_initialization(self):
        from src.battery import Battery
        battery = Battery(capacity_mwh=100, power_mw=50, efficiency=0.90)

        assert battery.capacity_mwh == 100
        assert battery.power_mw == 50
        assert battery.efficiency == 0.90
        assert battery.current_soc == 0.0

    def test_battery_charge(self):
        from src.battery import Battery
        battery = Battery(capacity_mwh=100, power_mw=50, efficiency=0.90)

        charged, from_grid = battery.charge(energy_mwh=10)

        assert charged > 0
        assert from_grid > charged  # Grid provides more due to losses
        assert battery.current_soc == charged
    
    def test_battery_discharge(self):
        from src.battery import Battery
        battery = Battery(capacity_mwh=100, power_mw=50, efficiency=0.90)
        
        # First charge
        battery.charge(energy_mwh=20)
        initial_soc = battery.current_soc

        # Then discharge
        discharged, to_grid = battery.discharge(energy_mwh=10)
        
        assert discharged > 0
        assert to_grid < discharged  # Grid receives less due to losses
        assert battery.current_soc < initial_soc
    
    def test_battery_cannot_overcharge(self):
        from src.battery import Battery
        battery = Battery(capacity_mwh=100, power_mw=50, efficiency=0.90)
        
        # Try to charge way more than capacity
        charged, _ = battery.charge(energy_mwh=1000)
        
        assert battery.current_soc <= battery.capacity_mwh
    
    def test_battery_cannot_overdischarge(self):
        from src.battery import Battery
        battery = Battery(capacity_mwh=100, power_mw=50, efficiency=0.90)
        
        # No charge, try to discharge
        discharged, _ = battery.discharge(energy_mwh=10)
        
        assert discharged == 0
        assert battery.current_soc >= 0


class TestGreedyStrategy:
    """Tests for Greedy strategy."""

    def test_greedy_returns_dataframe(self, sample_price_df):
        from src.strategies.greedy import run_greedy_strategy
        
        result, thresholds = run_greedy_strategy(sample_price_df)

        assert isinstance(result, pd.DataFrame)
        assert 'action' in result.columns
        assert 'cumulative_profit' in result.columns
        assert 'soc' in result.columns
    
    def test_greedy_thresholds_calculated(self, sample_price_df):
        from src.strategies.greedy import run_greedy_strategy
        
        result, thresholds = run_greedy_strategy(sample_price_df)
        
        assert 'buy_threshold' in thresholds
        assert 'sell_threshold' in thresholds
        assert thresholds['sell_threshold'] > thresholds['buy_threshold']
    
    def test_greedy_charges_at_low_prices(self, sample_price_df):
        from src.strategies.greedy import run_greedy_strategy
        
        result, _ = run_greedy_strategy(sample_price_df)
        
        # Should have some charge actions
        charge_count = (result['action'] == 'charge').sum()
        assert charge_count > 0


class TestSlidingWindowStrategy:
    """Tests for Sliding Window strategy."""

    def test_sliding_window_returns_dataframe(self, sample_price_df):
        from src.strategies.sliding_window import run_sliding_window_strategy
        
        result = run_sliding_window_strategy(sample_price_df, window_size=12)
        
        assert isinstance(result, pd.DataFrame)
        assert 'action' in result.columns
        assert 'cumulative_profit' in result.columns
    
    def test_sliding_window_finds_extrema(self, sample_price_df):
        from src.strategies.sliding_window import find_local_extrema
        
        prices = sample_price_df['RRP'].values
        is_min, is_max = find_local_extrema(prices, window_size=10)
        
        # Should find some minima and maxima
        assert is_min.sum() > 0
        assert is_max.sum() > 0


class TestPerfectForesightStrategy:
    """Tests for Perfect Foresight (DP) strategy."""

    def test_perfect_foresight_returns_dataframe(self, sample_price_df):
        from src.strategies.perfect_foresight import run_perfect_foresight
        
        result = run_perfect_foresight(sample_price_df, soc_levels=11)
        
        assert isinstance(result, pd.DataFrame)
        assert 'action' in result.columns
        assert 'cumulative_profit' in result.columns
    
    def test_perfect_foresight_beats_greedy(self, sample_price_df):
        from src.strategies.perfect_foresight import run_perfect_foresight
        from src.strategies.greedy import run_greedy_strategy
        
        pf_result = run_perfect_foresight(sample_price_df, soc_levels=11)
        greedy_result, _ = run_greedy_strategy(sample_price_df)
        
        pf_profit = pf_result['cumulative_profit'].iloc[-1]
        greedy_profit = greedy_result['cumulative_profit'].iloc[-1]

        # Perfect foresight should be >= greedy (it's optimal)
        assert pf_profit >= greedy_profit

    def test_perfect_foresight_soc_stays_valid(self, sample_price_df):
        from src.strategies.perfect_foresight import run_perfect_foresight

        result = run_perfect_foresight(sample_price_df, capacity_mwh=100, soc_levels=11)

        # SoC should always be between 0 and capacity
        assert result['soc'].min() >= 0
        assert result['soc'].max() <= 100


class TestDynamicProgrammingWrapper:
    """Tests that DP wrapper works correctly."""

    def test_dp_wrapper_calls_perfect_foresight(self, sample_price_df):
        from src.strategies.dynamic_programming import run_dp_strategy
        from src.strategies.perfect_foresight import run_perfect_foresight

        dp_result = run_dp_strategy(sample_price_df, num_soc_states=11)
        pf_result = run_perfect_foresight(sample_price_df, soc_levels=11)

        # Should produce identical results
        dp_profit = dp_result['cumulative_profit'].iloc[-1]
        pf_profit = pf_result['cumulative_profit'].iloc[-1]

        assert abs(dp_profit - pf_profit) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
