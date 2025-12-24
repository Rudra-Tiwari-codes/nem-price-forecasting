"""
Machine Learning forecasting models using scikit-learn and XGBoost.
"""

import logging
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def create_lag_features(prices: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create lagged features for time series prediction."""
    if len(prices) <= lookback:
        raise ValueError(f"Price series too short for lookback {lookback}")
    
    X, y = [], []
    for i in range(lookback, len(prices)):
        X.append(prices[i-lookback:i])
        y.append(prices[i])
    
    return np.array(X), np.array(y)


def create_features(prices: np.ndarray, lookback: int) -> np.ndarray:
    """Create feature matrix with lag, rolling stats, and time features."""
    if len(prices) <= lookback:
        raise ValueError(f"Price series too short for lookback {lookback}")
    
    features = []
    for i in range(lookback, len(prices)):
        window = prices[i-lookback:i]
        feat = [
            *window,
            window.mean(),
            window.std(),
            window.min(),
            window.max(),
            window[-1] - window[0],
            (window[-1] - window.mean()) / (window.std() + 1e-8),
        ]
        features.append(feat)
    
    return np.array(features)


class XGBoostPredictor:
    """XGBoost-based price predictor."""
    
    def __init__(self, lookback: int = 12, n_estimators: int = 100) -> None:
        if lookback < 1:
            raise ValueError("Lookback must be at least 1")
        self.lookback = lookback
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = StandardScaler()
    
    def fit_predict(self, prices: np.ndarray) -> np.ndarray:
        """Fit model and return predictions for all prices."""
        if len(prices) <= self.lookback:
            return np.full(len(prices), np.nan)
        
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        except ImportError:
            logger.warning("GradientBoosting not available, using Ridge")
            self.model = Ridge(alpha=1.0)
        
        X = create_features(prices, self.lookback)
        y = prices[self.lookback:]
        
        X_scaled = self.scaler.fit_transform(X)
        
        predictions = np.full(len(prices), np.nan)
        
        train_size = max(50, len(X) // 2)
        
        if len(X) > train_size:
            X_train, y_train = X_scaled[:train_size], y[:train_size]
            self.model.fit(X_train, y_train)
            
            for i in range(train_size, len(X)):
                predictions[self.lookback + i] = self.model.predict(X_scaled[i:i+1])[0]
                
                if i % 50 == 0 and i > train_size:
                    self.model.fit(X_scaled[:i], y[:i])
        
        return predictions
    
    def predict_next(self, recent_prices: np.ndarray) -> float:
        """Predict next price given recent history."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        
        if len(recent_prices) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} prices")
        
        window = recent_prices[-self.lookback:]
        feat = [
            *window,
            window.mean(),
            window.std(),
            window.min(),
            window.max(),
            window[-1] - window[0],
            (window[-1] - window.mean()) / (window.std() + 1e-8),
        ]
        X = np.array([feat])
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])


class RandomForestPredictor:
    """Random Forest-based price predictor."""
    
    def __init__(self, lookback: int = 12, n_estimators: int = 50) -> None:
        if lookback < 1:
            raise ValueError("Lookback must be at least 1")
        self.lookback = lookback
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    def fit_predict(self, prices: np.ndarray) -> np.ndarray:
        """Fit model and return predictions for all prices."""
        if len(prices) <= self.lookback:
            return np.full(len(prices), np.nan)
        
        X = create_features(prices, self.lookback)
        y = prices[self.lookback:]
        
        X_scaled = self.scaler.fit_transform(X)
        
        predictions = np.full(len(prices), np.nan)
        
        train_size = max(50, len(X) // 2)
        
        if len(X) > train_size:
            X_train, y_train = X_scaled[:train_size], y[:train_size]
            self.model.fit(X_train, y_train)
            
            for i in range(train_size, len(X)):
                predictions[self.lookback + i] = self.model.predict(X_scaled[i:i+1])[0]
        
        return predictions


class RidgePredictor:
    """Ridge regression-based price predictor (lightweight baseline)."""
    
    def __init__(self, lookback: int = 12, alpha: float = 1.0) -> None:
        if lookback < 1:
            raise ValueError("Lookback must be at least 1")
        self.lookback = lookback
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
    
    def fit_predict(self, prices: np.ndarray) -> np.ndarray:
        """Fit model and return predictions for all prices."""
        if len(prices) <= self.lookback:
            return np.full(len(prices), np.nan)
        
        X = create_features(prices, self.lookback)
        y = prices[self.lookback:]
        
        X_scaled = self.scaler.fit_transform(X)
        
        predictions = np.full(len(prices), np.nan)
        
        train_size = max(30, len(X) // 2)
        
        if len(X) > train_size:
            self.model.fit(X_scaled[:train_size], y[:train_size])
            predictions[self.lookback + train_size:] = self.model.predict(X_scaled[train_size:])
        
        return predictions


def evaluate_ml_models(prices: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Evaluate all ML models on price data."""
    from src.forecasting import evaluate_forecast
    
    models = {
        'XGBoost': XGBoostPredictor(lookback=12),
        'RandomForest': RandomForestPredictor(lookback=12),
        'Ridge': RidgePredictor(lookback=12),
    }
    
    results = {}
    for name, model in models.items():
        try:
            predictions = model.fit_predict(prices)
            metrics = evaluate_forecast(prices, predictions)
            results[name] = metrics
            logger.info(f"{name}: MAE=${metrics['mae']:.2f}, RMSE=${metrics['rmse']:.2f}")
        except Exception as e:
            logger.warning(f"Failed to evaluate {name}: {e}")
            results[name] = {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}
    
    return results


def test_ml_forecasters() -> None:
    """Test ML forecasters on sample data."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_dispatch_data
    from forecasting import evaluate_forecast
    
    data_path = Path(__file__).parent.parent / "data" / "combined_dispatch_prices.csv"
    
    try:
        df = load_dispatch_data(str(data_path))
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    prices = df['RRP'].values
    
    print("ML Model Evaluation")
    print("=" * 60)
    
    models = [
        ('XGBoost', XGBoostPredictor(lookback=12)),
        ('Random Forest', RandomForestPredictor(lookback=12)),
        ('Ridge Regression', RidgePredictor(lookback=12)),
    ]
    
    print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'Samples':<10}")
    print("-" * 60)
    
    for name, model in models:
        try:
            predictions = model.fit_predict(prices)
            metrics = evaluate_forecast(prices, predictions)
            print(f"{name:<20} ${metrics['mae']:<11.2f} ${metrics['rmse']:<11.2f} {metrics['n_samples']:<10}")
        except Exception as e:
            print(f"{name:<20} Error: {e}")


if __name__ == "__main__":
    test_ml_forecasters()
