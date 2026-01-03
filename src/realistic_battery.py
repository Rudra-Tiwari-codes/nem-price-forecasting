"""
Realistic Battery Model for NEM arbitrage.

Enhanced battery simulation with:
- Separate charge/discharge efficiencies
- Cycle-based degradation
- Market participation fees (FCAS, network charges)
- Ramp rate constraints
- Depth of discharge limits
- Calendar and cycle aging
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
from datetime import datetime, timedelta
import numpy as np


@dataclass
class BatteryConfig:
    """Configuration for realistic battery model."""
    # Capacity and power
    capacity_mwh: float = 100.0
    power_mw: float = 50.0
    
    # Efficiency (separate charge/discharge)
    charge_efficiency: float = 0.95      # Grid to battery
    discharge_efficiency: float = 0.95   # Battery to grid
    # Note: Round-trip = charge_eff * discharge_eff = 0.9025
    
    # Degradation
    cycle_degradation: float = 0.0001    # Capacity loss per full cycle (0.01%)
    calendar_degradation_per_day: float = 0.00005  # Daily capacity loss (0.005%)
    
    # Operating limits
    min_soc_percent: float = 10.0        # Minimum SoC (depth of discharge limit)
    max_soc_percent: float = 90.0        # Maximum SoC (extends battery life)
    ramp_rate_mw_per_min: float = 10.0   # Max power change per minute
    
    # Costs and fees ($/MWh)
    market_fee_per_mwh: float = 0.50     # AEMO market fees
    network_charge_per_mwh: float = 5.0  # Network access charges
    variable_om_per_mwh: float = 2.0     # Variable O&M costs
    
    # Fixed costs ($/day) - for ROI calculation
    fixed_om_per_day: float = 500.0      # Fixed O&M
    

@dataclass 
class RealisticBattery:
    """
    A realistic grid-scale battery energy storage system.
    
    Models real-world constraints and costs that affect profitability.
    """
    config: BatteryConfig = field(default_factory=BatteryConfig)
    
    # State
    current_soc: float = 0.0
    nominal_capacity: float = 0.0  # Original capacity
    current_capacity: float = 0.0  # Degraded capacity
    total_cycles: float = 0.0
    total_energy_throughput: float = 0.0  # MWh charged + discharged
    last_power: float = 0.0  # For ramp rate tracking
    start_date: datetime = field(default_factory=datetime.now)
    
    # Cost tracking
    total_fees: float = 0.0
    total_degradation_cost: float = 0.0
    
    def __post_init__(self):
        """Initialize battery state."""
        self.nominal_capacity = self.config.capacity_mwh
        self.current_capacity = self.nominal_capacity
        self.current_soc = 0.0
        
    @property
    def effective_min_soc(self) -> float:
        """Minimum SoC in MWh based on DoD limit."""
        return self.current_capacity * (self.config.min_soc_percent / 100)
    
    @property
    def effective_max_soc(self) -> float:
        """Maximum SoC in MWh based on max SoC limit."""
        return self.current_capacity * (self.config.max_soc_percent / 100)
    
    @property
    def available_charge_mwh(self) -> float:
        """Available capacity to charge (respecting max SoC limit)."""
        return max(0, self.effective_max_soc - self.current_soc)
    
    @property
    def available_discharge_mwh(self) -> float:
        """Available energy to discharge (respecting min SoC limit)."""
        return max(0, self.current_soc - self.effective_min_soc)
    
    @property
    def soc_percent(self) -> float:
        """Current SoC as percentage of current capacity."""
        if self.current_capacity <= 0:
            return 0.0
        return (self.current_soc / self.current_capacity) * 100
    
    @property
    def health_percent(self) -> float:
        """Battery health as percentage of original capacity."""
        return (self.current_capacity / self.nominal_capacity) * 100
    
    def _apply_ramp_constraint(self, requested_power: float, interval_hours: float) -> float:
        """Apply ramp rate constraint to power request."""
        interval_minutes = interval_hours * 60
        max_ramp = self.config.ramp_rate_mw_per_min * interval_minutes
        
        power_change = abs(requested_power - self.last_power)
        if power_change > max_ramp:
            # Limit the power change
            if requested_power > self.last_power:
                return self.last_power + max_ramp
            else:
                return self.last_power - max_ramp
        return requested_power
    
    def _calculate_fees(self, energy_mwh: float) -> float:
        """Calculate total fees for a transaction."""
        fees = energy_mwh * (
            self.config.market_fee_per_mwh +
            self.config.network_charge_per_mwh +
            self.config.variable_om_per_mwh
        )
        return fees
    
    def _apply_cycle_degradation(self, energy_mwh: float):
        """Apply cycle-based degradation."""
        # A full cycle is charging and discharging the full capacity
        # We track partial cycles
        cycle_fraction = energy_mwh / self.nominal_capacity
        capacity_loss = self.nominal_capacity * self.config.cycle_degradation * cycle_fraction
        
        self.current_capacity = max(0, self.current_capacity - capacity_loss)
        self.total_cycles += cycle_fraction / 2  # Divide by 2 since charge+discharge = 1 cycle
        self.total_degradation_cost += capacity_loss * 50  # Assume $50/kWh replacement cost
    
    def apply_calendar_degradation(self, days: float):
        """Apply calendar-based degradation."""
        capacity_loss = self.nominal_capacity * self.config.calendar_degradation_per_day * days
        self.current_capacity = max(0, self.current_capacity - capacity_loss)
        self.total_degradation_cost += capacity_loss * 50
    
    def max_charge_this_interval(self, interval_hours: float = 5/60) -> float:
        """Calculate maximum energy that can be charged in one interval."""
        power_limit = self.config.power_mw * interval_hours
        power_limit = self._apply_ramp_constraint(self.config.power_mw, interval_hours) * interval_hours
        capacity_limit = self.available_charge_mwh
        return min(power_limit, capacity_limit)
    
    def max_discharge_this_interval(self, interval_hours: float = 5/60) -> float:
        """Calculate maximum energy that can be discharged in one interval."""
        power_limit = self.config.power_mw * interval_hours
        power_limit = self._apply_ramp_constraint(self.config.power_mw, interval_hours) * interval_hours
        soc_limit = self.available_discharge_mwh
        return min(power_limit, soc_limit)
    
    def charge(
        self, 
        energy_mwh: float, 
        price: float,
        interval_hours: float = 5/60
    ) -> Tuple[float, float, float]:
        """
        Charge the battery.
        
        Args:
            energy_mwh: Requested energy to charge (MWh)
            price: Current electricity price ($/MWh)
            interval_hours: Duration of interval
            
        Returns:
            Tuple of (actual_energy_stored, cost_from_grid, total_cost_including_fees)
        """
        max_charge = self.max_charge_this_interval(interval_hours)
        actual_charge = min(energy_mwh, max_charge)
        
        if actual_charge <= 0:
            return 0.0, 0.0, 0.0
        
        # Grid provides more than battery stores due to charging losses
        energy_from_grid = actual_charge / self.config.charge_efficiency
        
        # Costs
        energy_cost = energy_from_grid * price
        fees = self._calculate_fees(energy_from_grid)
        total_cost = energy_cost + fees
        
        # Update state
        self.current_soc += actual_charge
        self.total_energy_throughput += actual_charge
        self.total_fees += fees
        self.last_power = actual_charge / interval_hours
        
        # Apply degradation
        self._apply_cycle_degradation(actual_charge)
        
        return actual_charge, energy_cost, total_cost
    
    def discharge(
        self, 
        energy_mwh: float,
        price: float,
        interval_hours: float = 5/60
    ) -> Tuple[float, float, float]:
        """
        Discharge the battery.
        
        Args:
            energy_mwh: Requested energy to discharge (MWh)
            price: Current electricity price ($/MWh)
            interval_hours: Duration of interval
            
        Returns:
            Tuple of (actual_energy_from_battery, gross_revenue, net_revenue_after_fees)
        """
        max_discharge = self.max_discharge_this_interval(interval_hours)
        actual_discharge = min(energy_mwh, max_discharge)
        
        if actual_discharge <= 0:
            return 0.0, 0.0, 0.0
        
        # Grid receives less than battery provides due to discharge losses
        energy_to_grid = actual_discharge * self.config.discharge_efficiency
        
        # Revenue and costs
        gross_revenue = energy_to_grid * price
        fees = self._calculate_fees(energy_to_grid)
        net_revenue = gross_revenue - fees
        
        # Update state
        self.current_soc -= actual_discharge
        self.total_energy_throughput += actual_discharge
        self.total_fees += fees
        self.last_power = -actual_discharge / interval_hours  # Negative for discharge
        
        # Apply degradation
        self._apply_cycle_degradation(actual_discharge)
        
        return actual_discharge, gross_revenue, net_revenue
    
    def get_statistics(self) -> Dict:
        """Get battery statistics."""
        return {
            'nominal_capacity_mwh': self.nominal_capacity,
            'current_capacity_mwh': self.current_capacity,
            'health_percent': self.health_percent,
            'current_soc_mwh': self.current_soc,
            'soc_percent': self.soc_percent,
            'total_cycles': self.total_cycles,
            'total_energy_throughput_mwh': self.total_energy_throughput,
            'total_fees_paid': self.total_fees,
            'total_degradation_cost': self.total_degradation_cost,
            'round_trip_efficiency': self.config.charge_efficiency * self.config.discharge_efficiency
        }
    
    def reset(self, soc: float = 0.0):
        """Reset battery to specified State of Charge."""
        if not 0 <= soc <= self.current_capacity:
            raise ValueError("SoC must be between 0 and current capacity")
        self.current_soc = soc
        self.last_power = 0.0
    
    def __repr__(self) -> str:
        return (
            f"RealisticBattery(capacity={self.current_capacity:.1f}/{self.nominal_capacity:.1f}MWh, "
            f"health={self.health_percent:.1f}%, "
            f"SoC={self.current_soc:.1f}MWh/{self.soc_percent:.1f}%, "
            f"cycles={self.total_cycles:.1f})"
        )


def compare_battery_models(prices: np.ndarray, simple_efficiency: float = 0.90) -> Dict:
    """
    Compare simple vs realistic battery model on same price data.
    
    Shows the impact of realistic constraints on profitability.
    """
    from battery import Battery
    
    n = len(prices)
    interval_hours = 5/60
    
    # Simple battery
    simple = Battery(capacity_mwh=100, power_mw=50, efficiency=simple_efficiency)
    simple_profit = 0.0
    
    # Realistic battery  
    config = BatteryConfig(capacity_mwh=100, power_mw=50)
    realistic = RealisticBattery(config=config)
    realistic_profit = 0.0
    
    # Simple threshold strategy
    buy_threshold = np.percentile(prices, 25)
    sell_threshold = np.percentile(prices, 75)
    
    for i in range(n):
        price = prices[i]
        
        # Simple battery trades
        if price <= buy_threshold and simple.available_charge_mwh > 0:
            charged, from_grid = simple.charge(energy_mwh=10)
            simple_profit -= from_grid * price
        elif price >= sell_threshold and simple.available_discharge_mwh > 0:
            discharged, to_grid = simple.discharge(energy_mwh=10)
            simple_profit += to_grid * price
        
        # Realistic battery trades
        if price <= buy_threshold and realistic.available_charge_mwh > 0:
            _, _, total_cost = realistic.charge(energy_mwh=10, price=price)
            realistic_profit -= total_cost
        elif price >= sell_threshold and realistic.available_discharge_mwh > 0:
            _, _, net_revenue = realistic.discharge(energy_mwh=10, price=price)
            realistic_profit += net_revenue
    
    stats = realistic.get_statistics()
    
    return {
        'simple_profit': simple_profit,
        'realistic_profit': realistic_profit,
        'profit_reduction_pct': (1 - realistic_profit / simple_profit) * 100 if simple_profit > 0 else 0,
        'fees_paid': stats['total_fees_paid'],
        'degradation_cost': stats['total_degradation_cost'],
        'battery_health_pct': stats['health_percent'],
        'total_cycles': stats['total_cycles']
    }


if __name__ == "__main__":
    # Demonstration
    config = BatteryConfig(
        capacity_mwh=100,
        power_mw=50,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        min_soc_percent=10,
        max_soc_percent=90
    )
    
    battery = RealisticBattery(config=config)
    print(f"Initial: {battery}")
    print(f"  Round-trip efficiency: {config.charge_efficiency * config.discharge_efficiency:.1%}")
    print(f"  Usable capacity: {battery.effective_max_soc - battery.effective_min_soc:.1f} MWh")
    
    # Simulate some trades
    print("\nSimulating trades...")
    
    # Charge at low price
    stored, energy_cost, total_cost = battery.charge(50, price=20)
    print(f"  Charged {stored:.1f} MWh at $20/MWh")
    print(f"    Energy cost: ${energy_cost:.2f}, Total cost (incl fees): ${total_cost:.2f}")
    
    # Discharge at high price
    discharged, gross_rev, net_rev = battery.discharge(30, price=200)
    print(f"  Discharged {discharged:.1f} MWh at $200/MWh")
    print(f"    Gross revenue: ${gross_rev:.2f}, Net revenue (after fees): ${net_rev:.2f}")
    
    print(f"\nFinal state: {battery}")
    stats = battery.get_statistics()
    print(f"  Total fees: ${stats['total_fees_paid']:.2f}")
    print(f"  Degradation cost: ${stats['total_degradation_cost']:.2f}")
