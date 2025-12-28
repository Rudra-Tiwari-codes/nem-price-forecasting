"""
Constants for NEM Price Forecasting project.

Centralizes magic numbers and configuration values.
"""

# Time intervals
INTERVAL_MINUTES = 5  # AEMO dispatch interval
INTERVALS_PER_HOUR = 12  # 60 / 5
INTERVALS_PER_DAY = 288  # 24 * 12

# Default battery configuration
DEFAULT_CAPACITY_MWH = 100.0
DEFAULT_POWER_MW = 50.0
DEFAULT_EFFICIENCY = 0.90

# Price thresholds
SPIKE_THRESHOLD = 300.0  # $/MWh - prices above this are considered spikes
NEGATIVE_THRESHOLD = 0.0  # $/MWh

# NEM Regions
REGIONS = ['SA1', 'NSW1', 'VIC1', 'QLD1', 'TAS1']

# Peak hours (7-10am and 5-9pm)
PEAK_HOURS = [7, 8, 9, 10, 17, 18, 19, 20]
