"""
Price forecasting module with simple predictive models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RollingMeanPredictor:
    """Predicts next price as rolling average of recent prices."""
    
    def __init__(self, window: int = 12) -> None:
        if window < 1:
            raise ValueError("Window must be at least 1")
        self.window = window
    
    def predict(self, prices: np.ndarray) -> np.ndarray:
        if len(prices) == 0:
            return np.array([])
        
        predictions = np.full(len(prices), np.nan)
        for i in range(self.window, len(prices)):
            predictions[i] = prices[i-self.window:i].mean()
        return predictions


class EMAPredictor:
    """Exponential moving average predictor."""
    
    def __init__(self, span: int = 12) -> None:
        if span < 1:
            raise ValueError("Span must be at least 1")
        self.span = span
    
    def predict(self, prices: np.ndarray) -> np.ndarray:
        if len(prices) == 0:
            return np.array([])
        
        series = pd.Series(prices)
        ema = series.ewm(span=self.span, adjust=False).mean()
        predictions = np.full(len(prices), np.nan)
        predictions[1:] = ema.values[:-1]
        return predictions


class LinearTrendPredictor:
    """Simple linear regression on recent window."""
    
    def __init__(self, window: int = 24) -> None:
        if window < 2:
            raise ValueError("Window must be at least 2 for linear regression")
        self.window = window
    
    def predict(self, prices: np.ndarray) -> np.ndarray:
        if len(prices) < self.window:
            return np.full(len(prices), np.nan)
        
        predictions = np.full(len(prices), np.nan)
        x = np.arange(self.window)
        
        for i in range(self.window, len(prices)):
            y = prices[i-self.window:i]
            try:
                slope, intercept = np.polyfit(x, y, 1)
                predictions[i] = slope * self.window + intercept
            except (np.linalg.LinAlgError, ValueError):
                predictions[i] = np.nan
        
        return predictions


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate forecast error metrics."""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have same length")
    
    mask = ~np.isnan(predicted) & ~np.isnan(actual)
    actual_clean = np.array(actual)[mask]
    predicted_clean = np.array(predicted)[mask]
    
    if len(actual_clean) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'n_samples': 0}
    
    errors = actual_clean - predicted_clean
    abs_errors = np.abs(errors)
    
    mae = float(abs_errors.mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    
    nonzero = actual_clean != 0
    if nonzero.sum() > 0:
        mape = float((abs_errors[nonzero] / np.abs(actual_clean[nonzero])).mean() * 100)
    else:
        mape = np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'n_samples': len(actual_clean)
    }


def run_forecast_strategy(
    df: pd.DataFrame, 
    predictor: object,
    capacity_mwh: float = 100.0, 
    power_mw: float = 50.0, 
    efficiency: float = 0.90
) -> pd.DataFrame:
    """
    Trading strategy based on forecast vs current price.
    Buy when forecast > current (price expected to rise).
    Sell when forecast < current (price expected to fall).
    """
    if 'RRP' not in df.columns:
        raise ValueError("DataFrame must contain 'RRP' column")
    
    prices = df['RRP'].values
    
    if not hasattr(predictor, 'predict'):
        raise TypeError("Predictor must have a predict method")
    
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


def test_forecasters() -> None:
    """Test all predictors on sample data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_dispatch_data
    
    data_path = Path(__file__).parent.parent / "data" / "combined_dispatch_prices.csv"
    
    try:
        df = load_dispatch_data(str(data_path))
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        print(f"Error loading data: {e}")
        return
    
    prices = df['RRP'].values
    
    print("Forecast Model Evaluation")
    print("-" * 50)
    
    predictors: List[Tuple[str, object]] = [
        ('Rolling Mean (12)', RollingMeanPredictor(12)),
        ('Rolling Mean (24)', RollingMeanPredictor(24)),
        ('EMA (12)', EMAPredictor(12)),
        ('EMA (24)', EMAPredictor(24)),
        ('Linear Trend (24)', LinearTrendPredictor(24)),
    ]
    
    print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-" * 50)
    
    for name, predictor in predictors:
        try:
            predictions = predictor.predict(prices)
            metrics = evaluate_forecast(prices, predictions)
            print(f"{name:<20} ${metrics['mae']:<9.2f} ${metrics['rmse']:<9.2f} {metrics['mape']:<9.1f}%")
        except Exception as e:
            logger.warning(f"Failed to evaluate {name}: {e}")
            print(f"{name:<20} Error: {e}")
    
    print("\nForecast-Based Strategy Test")
    print("-" * 50)
    
    try:
        best_predictor = EMAPredictor(12)
        result = run_forecast_strategy(df, best_predictor)
        final_profit = result['cumulative_profit'].iloc[-1]
        n_charges = (result['action'] == 'charge').sum()
        n_discharges = (result['action'] == 'discharge').sum()
        
        print(f"EMA(12) Strategy Results:")
        print(f"  Total Profit: ${final_profit:,.2f}")
        print(f"  Trades: {n_charges} charges, {n_discharges} discharges")
    except Exception as e:
        logger.error(f"Strategy test failed: {e}")
        print(f"Strategy test failed: {e}")


if __name__ == "__main__":
    test_forecasters()
