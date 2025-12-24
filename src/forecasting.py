"""
Price forecasting module with simple predictive models.
"""

import numpy as np
import pandas as pd
from pathlib import Path


class RollingMeanPredictor:
    """Predicts next price as rolling average of recent prices."""
    
    def __init__(self, window=12):
        self.window = window
    
    def predict(self, prices):
        predictions = np.full(len(prices), np.nan)
        for i in range(self.window, len(prices)):
            predictions[i] = prices[i-self.window:i].mean()
        return predictions


class EMAPredictor:
    """Exponential moving average predictor."""
    
    def __init__(self, span=12):
        self.span = span
    
    def predict(self, prices):
        series = pd.Series(prices)
        ema = series.ewm(span=self.span, adjust=False).mean()
        predictions = np.full(len(prices), np.nan)
        predictions[1:] = ema.values[:-1]
        return predictions


class LinearTrendPredictor:
    """Simple linear regression on recent window."""
    
    def __init__(self, window=24):
        self.window = window
    
    def predict(self, prices):
        predictions = np.full(len(prices), np.nan)
        for i in range(self.window, len(prices)):
            y = prices[i-self.window:i]
            x = np.arange(self.window)
            slope, intercept = np.polyfit(x, y, 1)
            predictions[i] = slope * self.window + intercept
        return predictions


def evaluate_forecast(actual, predicted):
    """Calculate forecast error metrics."""
    mask = ~np.isnan(predicted) & ~np.isnan(actual)
    actual = np.array(actual)[mask]
    predicted = np.array(predicted)[mask]
    
    if len(actual) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}
    
    errors = actual - predicted
    abs_errors = np.abs(errors)
    
    mae = abs_errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    
    nonzero = actual != 0
    if nonzero.sum() > 0:
        mape = (abs_errors[nonzero] / np.abs(actual[nonzero])).mean() * 100
    else:
        mape = np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'n_samples': len(actual)
    }


def run_forecast_strategy(df, predictor, capacity_mwh=100.0, power_mw=50.0, efficiency=0.90):
    """
    Trading strategy based on forecast vs current price.
    Buy when forecast > current (price expected to rise).
    Sell when forecast < current (price expected to fall).
    """
    prices = df['RRP'].values
    predictions = predictor.predict(prices)
    
    n = len(prices)
    interval_hours = 5/60
    max_energy = power_mw * interval_hours
    efficiency_factor = np.sqrt(efficiency)
    
    actions = np.full(n, 'hold', dtype=object)
    soc = np.zeros(n)
    profit = np.zeros(n)
    cumulative_profit = np.zeros(n)
    
    current_soc = 0.0
    total_profit = 0.0
    
    for i in range(1, n):
        price = prices[i]
        forecast = predictions[i]
        
        if np.isnan(forecast):
            soc[i] = current_soc
            cumulative_profit[i] = total_profit
            continue
        
        if forecast > price * 1.05 and current_soc < capacity_mwh:
            charge_amount = min(max_energy, capacity_mwh - current_soc)
            grid_energy = charge_amount / efficiency_factor
            cost = grid_energy * price
            
            actions[i] = 'charge'
            current_soc += charge_amount
            profit[i] = -cost
            
        elif forecast < price * 0.95 and current_soc > 0:
            discharge_amount = min(max_energy, current_soc)
            grid_energy = discharge_amount * efficiency_factor
            revenue = grid_energy * price
            
            actions[i] = 'discharge'
            current_soc -= discharge_amount
            profit[i] = revenue
        
        soc[i] = current_soc
        total_profit += profit[i]
        cumulative_profit[i] = total_profit
    
    result = df.copy()
    result['action'] = actions
    result['soc'] = soc
    result['profit'] = profit
    result['cumulative_profit'] = cumulative_profit
    result['forecast'] = predictions
    
    return result


def test_forecasters():
    """Test all predictors on sample data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_dispatch_data
    
    data_path = Path(__file__).parent.parent / "data" / "combined_dispatch_prices.csv"
    df = load_dispatch_data(str(data_path))
    prices = df['RRP'].values
    
    print("Forecast Model Evaluation")
    print("-" * 50)
    
    predictors = [
        ('Rolling Mean (12)', RollingMeanPredictor(12)),
        ('Rolling Mean (24)', RollingMeanPredictor(24)),
        ('EMA (12)', EMAPredictor(12)),
        ('EMA (24)', EMAPredictor(24)),
        ('Linear Trend (24)', LinearTrendPredictor(24)),
    ]
    
    print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-" * 50)
    
    for name, predictor in predictors:
        predictions = predictor.predict(prices)
        metrics = evaluate_forecast(prices, predictions)
        print(f"{name:<20} ${metrics['mae']:<9.2f} ${metrics['rmse']:<9.2f} {metrics['mape']:<9.1f}%")
    
    print("\nForecast-Based Strategy Test")
    print("-" * 50)
    
    best_predictor = EMAPredictor(12)
    result = run_forecast_strategy(df, best_predictor)
    final_profit = result['cumulative_profit'].iloc[-1]
    n_charges = (result['action'] == 'charge').sum()
    n_discharges = (result['action'] == 'discharge').sum()
    
    print(f"EMA(12) Strategy Results:")
    print(f"  Total Profit: ${final_profit:,.2f}")
    print(f"  Trades: {n_charges} charges, {n_discharges} discharges")


if __name__ == "__main__":
    test_forecasters()
