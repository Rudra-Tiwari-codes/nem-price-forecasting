"""
Battery simulation module for NEM arbitrage.

Models a grid-scale battery with realistic constraints:
- Capacity limits (MWh)
- Power rating limits (MW)
- Round-trip efficiency losses
- State of Charge tracking
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Battery:
    """
    A grid-scale battery energy storage system.
    
    Attributes:
        capacity_mwh: Total energy storage capacity in MWh
        power_mw: Maximum charge/discharge power in MW
        efficiency: Round-trip efficiency (0.0 to 1.0)
        current_soc: Current State of Charge in MWh
    """
    capacity_mwh: float = 100.0
    power_mw: float = 50.0
    efficiency: float = 0.90
    current_soc: float = 0.0
    
    def __post_init__(self):
        """Validate battery parameters."""
        if self.capacity_mwh <= 0:
            raise ValueError("Capacity must be positive")
        if self.power_mw <= 0:
            raise ValueError("Power rating must be positive")
        if not 0 < self.efficiency <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        if not 0 <= self.current_soc <= self.capacity_mwh:
            raise ValueError("SoC must be between 0 and capacity")
    
    @property
    def soc_percent(self) -> float:
        """Return State of Charge as a percentage (0-100)."""
        return (self.current_soc / self.capacity_mwh) * 100
    
    @property
    def available_charge_mwh(self) -> float:
        """Return available capacity to charge (MWh)."""
        return self.capacity_mwh - self.current_soc
    
    @property
    def available_discharge_mwh(self) -> float:
        """Return available energy to discharge (MWh)."""
        return self.current_soc
    
    def max_charge_this_interval(self, interval_hours: float = 5/60) -> float:
        """
        Calculate maximum energy that can be charged in one interval.
        
        Args:
            interval_hours: Duration of interval (default: 5 minutes = 5/60 hours)
            
        Returns:
            Maximum MWh that can be charged, considering power and capacity limits
        """
        power_limit = self.power_mw * interval_hours
        capacity_limit = self.available_charge_mwh
        return min(power_limit, capacity_limit)
    
    def max_discharge_this_interval(self, interval_hours: float = 5/60) -> float:
        """
        Calculate maximum energy that can be discharged in one interval.
        
        Args:
            interval_hours: Duration of interval (default: 5 minutes = 5/60 hours)
            
        Returns:
            Maximum MWh that can be discharged, considering power and SoC limits
        """
        power_limit = self.power_mw * interval_hours
        soc_limit = self.available_discharge_mwh
        return min(power_limit, soc_limit)
    
    def charge(self, energy_mwh: float, interval_hours: float = 5/60) -> Tuple[float, float]:
        """
        Charge the battery.
        
        Args:
            energy_mwh: Requested energy to charge (MWh)
            interval_hours: Duration of interval
            
        Returns:
            Tuple of (actual_energy_charged, energy_from_grid)
            Energy from grid is higher due to charging losses.
        """
        max_charge = self.max_charge_this_interval(interval_hours)
        actual_charge = min(energy_mwh, max_charge)
        
        # Account for efficiency: grid provides more than battery receives
        # For 90% efficiency, charging 10 MWh requires 10/sqrt(0.9) from grid
        efficiency_factor = self.efficiency ** 0.5
        energy_from_grid = actual_charge / efficiency_factor
        
        self.current_soc += actual_charge
        return actual_charge, energy_from_grid
    
    def discharge(self, energy_mwh: float, interval_hours: float = 5/60) -> Tuple[float, float]:
        """
        Discharge the battery.
        
        Args:
            energy_mwh: Requested energy to discharge (MWh)
            interval_hours: Duration of interval
            
        Returns:
            Tuple of (actual_energy_discharged, energy_to_grid)
            Energy to grid is lower due to discharging losses.
        """
        max_discharge = self.max_discharge_this_interval(interval_hours)
        actual_discharge = min(energy_mwh, max_discharge)
        
        # Account for efficiency: grid receives less than battery provides
        # For 90% efficiency, discharging 10 MWh delivers 10*sqrt(0.9) to grid
        efficiency_factor = self.efficiency ** 0.5
        energy_to_grid = actual_discharge * efficiency_factor
        
        self.current_soc -= actual_discharge
        return actual_discharge, energy_to_grid
    
    def reset(self, soc: float = 0.0):
        """Reset battery to specified State of Charge."""
        if not 0 <= soc <= self.capacity_mwh:
            raise ValueError("SoC must be between 0 and capacity")
        self.current_soc = soc
    
    def __repr__(self) -> str:
        return (
            f"Battery(capacity={self.capacity_mwh}MWh, "
            f"power={self.power_mw}MW, "
            f"efficiency={self.efficiency:.0%}, "
            f"SoC={self.current_soc:.1f}MWh/{self.soc_percent:.1f}%)"
        )


if __name__ == "__main__":
    # Quick demonstration
    battery = Battery(capacity_mwh=100, power_mw=50, efficiency=0.90)
    print(f"Initial: {battery}")
    
    # Charge for one 5-min interval at full power
    charged, from_grid = battery.charge(energy_mwh=10)
    print(f"Charged {charged:.2f} MWh (drew {from_grid:.2f} MWh from grid)")
    print(f"After charge: {battery}")
    
    # Discharge
    discharged, to_grid = battery.discharge(energy_mwh=5)
    print(f"Discharged {discharged:.2f} MWh (sent {to_grid:.2f} MWh to grid)")
    print(f"After discharge: {battery}")
