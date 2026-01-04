"""
Historical Accuracy Tracking Module.

Tracks prediction accuracy over time to monitor forecast quality:
- Logs predictions vs actuals
- Calculates rolling accuracy metrics
- Detects accuracy degradation
- Provides trend analysis for forecast MAPE
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Melbourne timezone
MELBOURNE_TZ = ZoneInfo('Australia/Sydney')


@dataclass
class PredictionRecord:
    """Single prediction record."""
    timestamp: str
    prediction_horizon: str  # e.g., '1h', '4h', '24h'
    model_name: str
    region: str
    predicted_value: float
    actual_value: Optional[float]
    error: Optional[float]
    absolute_error: Optional[float]
    percentage_error: Optional[float]


@dataclass
class AccuracySnapshot:
    """Accuracy metrics for a time period."""
    period_start: str
    period_end: str
    model_name: str
    region: str
    prediction_count: int
    mape: float
    mae: float
    rmse: float
    median_ape: float
    bias: float  # Positive = over-predicting, negative = under-predicting


class AccuracyTracker:
    """
    Tracks and analyzes prediction accuracy over time.
    
    Stores predictions and actuals, calculates rolling metrics,
    and detects accuracy degradation.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize tracker.
        
        Args:
            storage_path: Path to store prediction logs
        """
        if storage_path is None:
            storage_path = Path(__file__).parent.parent / "data" / "accuracy_logs"
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.predictions_file = self.storage_path / "predictions.json"
        self.snapshots_file = self.storage_path / "accuracy_snapshots.json"
        
        self.predictions = self._load_predictions()
        self.snapshots = self._load_snapshots()
    
    def _load_predictions(self) -> List[Dict]:
        """Load predictions from storage."""
        if self.predictions_file.exists():
            try:
                with open(self.predictions_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load predictions: {e}")
        return []
    
    def _load_snapshots(self) -> List[Dict]:
        """Load accuracy snapshots from storage."""
        if self.snapshots_file.exists():
            try:
                with open(self.snapshots_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load snapshots: {e}")
        return []
    
    def _save_predictions(self):
        """Save predictions to storage."""
        # Keep only last 30 days of predictions
        cutoff = (datetime.now(MELBOURNE_TZ) - timedelta(days=30)).isoformat()
        self.predictions = [p for p in self.predictions if p['timestamp'] >= cutoff]
        
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions, f, indent=2)
    
    def _save_snapshots(self):
        """Save accuracy snapshots to storage."""
        # Keep only last 90 days of snapshots
        cutoff = (datetime.now(MELBOURNE_TZ) - timedelta(days=90)).isoformat()
        self.snapshots = [s for s in self.snapshots if s['period_end'] >= cutoff]
        
        with open(self.snapshots_file, 'w') as f:
            json.dump(self.snapshots, f, indent=2)
    
    def log_prediction(
        self,
        predicted_value: float,
        model_name: str,
        region: str = 'NSW1',
        prediction_horizon: str = '1h',
        actual_value: Optional[float] = None
    ) -> PredictionRecord:
        """
        Log a prediction.
        
        Args:
            predicted_value: The predicted price
            model_name: Name of the forecasting model
            region: NEM region
            prediction_horizon: Time horizon of prediction
            actual_value: Actual value if known
            
        Returns:
            PredictionRecord object
        """
        timestamp = datetime.now(MELBOURNE_TZ).isoformat()
        
        error = None
        abs_error = None
        pct_error = None
        
        if actual_value is not None:
            error = predicted_value - actual_value
            abs_error = abs(error)
            if actual_value != 0:
                pct_error = abs(error / actual_value) * 100
        
        record = PredictionRecord(
            timestamp=timestamp,
            prediction_horizon=prediction_horizon,
            model_name=model_name,
            region=region,
            predicted_value=predicted_value,
            actual_value=actual_value,
            error=error,
            absolute_error=abs_error,
            percentage_error=pct_error
        )
        
        self.predictions.append(asdict(record))
        self._save_predictions()
        
        return record
    
    def update_actuals(
        self,
        actuals: List[Tuple[str, float]],  # List of (timestamp, actual_value)
        tolerance_minutes: int = 5
    ) -> int:
        """
        Update predictions with actual values.
        
        Args:
            actuals: List of (timestamp, actual_value) tuples
            tolerance_minutes: Time tolerance for matching
            
        Returns:
            Number of predictions updated
        """
        updated = 0
        tolerance = timedelta(minutes=tolerance_minutes)
        
        for pred in self.predictions:
            if pred['actual_value'] is not None:
                continue
            
            pred_time = datetime.fromisoformat(pred['timestamp'])
            
            for actual_timestamp, actual_value in actuals:
                actual_time = datetime.fromisoformat(actual_timestamp)
                
                if abs((pred_time - actual_time).total_seconds()) < tolerance.total_seconds():
                    pred['actual_value'] = actual_value
                    pred['error'] = pred['predicted_value'] - actual_value
                    pred['absolute_error'] = abs(pred['error'])
                    if actual_value != 0:
                        pred['percentage_error'] = abs(pred['error'] / actual_value) * 100
                    else:
                        pred['percentage_error'] = None
                    updated += 1
                    break
        
        if updated > 0:
            self._save_predictions()
        
        return updated
    
    def calculate_accuracy_metrics(
        self,
        model_name: Optional[str] = None,
        region: Optional[str] = None,
        hours: int = 24
    ) -> Optional[AccuracySnapshot]:
        """
        Calculate accuracy metrics for recent predictions.
        
        Args:
            model_name: Filter by model (optional)
            region: Filter by region (optional)
            hours: Look-back period in hours
            
        Returns:
            AccuracySnapshot or None if insufficient data
        """
        cutoff = (datetime.now(MELBOURNE_TZ) - timedelta(hours=hours)).isoformat()
        
        # Filter predictions
        filtered = [
            p for p in self.predictions
            if p['timestamp'] >= cutoff
            and p['actual_value'] is not None
            and p['percentage_error'] is not None
            and (model_name is None or p['model_name'] == model_name)
            and (region is None or p['region'] == region)
        ]
        
        if len(filtered) < 3:
            return None
        
        errors = [p['error'] for p in filtered]
        abs_errors = [p['absolute_error'] for p in filtered]
        pct_errors = [p['percentage_error'] for p in filtered]
        
        mape = np.mean(pct_errors)
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        median_ape = np.median(pct_errors)
        bias = np.mean(errors)
        
        period_start = min(p['timestamp'] for p in filtered)
        period_end = max(p['timestamp'] for p in filtered)
        
        return AccuracySnapshot(
            period_start=period_start,
            period_end=period_end,
            model_name=model_name or 'ALL',
            region=region or 'ALL',
            prediction_count=len(filtered),
            mape=round(mape, 2),
            mae=round(mae, 2),
            rmse=round(rmse, 2),
            median_ape=round(median_ape, 2),
            bias=round(bias, 2)
        )
    
    def create_daily_snapshot(
        self,
        model_name: Optional[str] = None,
        region: Optional[str] = None
    ) -> Optional[AccuracySnapshot]:
        """
        Create and store a daily accuracy snapshot.
        
        Args:
            model_name: Filter by model
            region: Filter by region
            
        Returns:
            AccuracySnapshot or None
        """
        snapshot = self.calculate_accuracy_metrics(model_name, region, hours=24)
        
        if snapshot:
            self.snapshots.append(asdict(snapshot))
            self._save_snapshots()
            logger.info(f"Created accuracy snapshot: MAPE={snapshot.mape}%")
        
        return snapshot
    
    def get_accuracy_trend(
        self,
        model_name: Optional[str] = None,
        region: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get accuracy trend over time.
        
        Args:
            model_name: Filter by model
            region: Filter by region
            days: Number of days to look back
            
        Returns:
            DataFrame with accuracy trend
        """
        cutoff = (datetime.now(MELBOURNE_TZ) - timedelta(days=days)).isoformat()
        
        filtered = [
            s for s in self.snapshots
            if s['period_end'] >= cutoff
            and (model_name is None or s['model_name'] == model_name)
            and (region is None or s['region'] == region)
        ]
        
        if not filtered:
            return pd.DataFrame()
        
        df = pd.DataFrame(filtered)
        df['date'] = pd.to_datetime(df['period_end']).dt.date
        
        return df.sort_values('date')
    
    def detect_accuracy_degradation(
        self,
        threshold_pct: float = 20.0,
        window_days: int = 7
    ) -> Dict:
        """
        Detect if forecast accuracy is degrading.
        
        Compares recent accuracy to historical baseline.
        
        Args:
            threshold_pct: Percentage increase in MAPE to trigger alert
            window_days: Days for recent window
            
        Returns:
            Dictionary with degradation analysis
        """
        trend = self.get_accuracy_trend(days=30)
        
        if len(trend) < 10:
            return {'status': 'insufficient_data', 'message': 'Need at least 10 days of data'}
        
        # Split into recent and historical
        recent = trend.tail(window_days)
        historical = trend.head(len(trend) - window_days)
        
        if len(recent) < 3 or len(historical) < 3:
            return {'status': 'insufficient_data', 'message': 'Not enough data in windows'}
        
        recent_mape = recent['mape'].mean()
        historical_mape = historical['mape'].mean()
        
        if historical_mape == 0:
            return {'status': 'error', 'message': 'Historical MAPE is zero'}
        
        pct_change = ((recent_mape - historical_mape) / historical_mape) * 100
        
        is_degrading = pct_change > threshold_pct
        
        return {
            'status': 'degrading' if is_degrading else 'stable',
            'recent_mape': round(recent_mape, 2),
            'historical_mape': round(historical_mape, 2),
            'pct_change': round(pct_change, 2),
            'threshold': threshold_pct,
            'is_degrading': is_degrading,
            'recommendation': 'Consider retraining models' if is_degrading else 'Models performing normally'
        }
    
    def get_model_comparison(self, days: int = 7) -> pd.DataFrame:
        """
        Compare accuracy across models.
        
        Args:
            days: Days to analyze
            
        Returns:
            DataFrame comparing model performance
        """
        cutoff = (datetime.now(MELBOURNE_TZ) - timedelta(days=days)).isoformat()
        
        filtered = [
            p for p in self.predictions
            if p['timestamp'] >= cutoff
            and p['actual_value'] is not None
            and p['percentage_error'] is not None
        ]
        
        if not filtered:
            return pd.DataFrame()
        
        df = pd.DataFrame(filtered)
        
        comparison = df.groupby('model_name').agg({
            'percentage_error': ['mean', 'median', 'std', 'count'],
            'error': 'mean'  # bias
        }).round(2)
        
        comparison.columns = ['mape', 'median_ape', 'std', 'count', 'bias']
        
        return comparison.sort_values('mape')
    
    def generate_dashboard_data(self, output_path: Optional[Path] = None) -> Dict:
        """
        Generate accuracy data for dashboard.
        
        Args:
            output_path: Path to save JSON
            
        Returns:
            Dictionary with dashboard data
        """
        # Current accuracy
        current = self.calculate_accuracy_metrics(hours=24)
        
        # 7-day trend
        trend = self.get_accuracy_trend(days=7)
        trend_data = []
        if not trend.empty:
            trend_data = trend[['date', 'mape', 'mae', 'prediction_count']].to_dict('records')
            # Convert dates to strings
            for item in trend_data:
                item['date'] = str(item['date'])
        
        # Degradation check
        degradation = self.detect_accuracy_degradation()
        
        # Model comparison
        comparison = self.get_model_comparison()
        comparison_data = comparison.to_dict('index') if not comparison.empty else {}
        
        dashboard_data = {
            'generated_at': datetime.now(MELBOURNE_TZ).isoformat(),
            'current_accuracy': {
                'mape': current.mape if current else None,
                'mae': current.mae if current else None,
                'prediction_count': current.prediction_count if current else 0,
                'period': f"Last 24 hours"
            },
            'trend': trend_data,
            'degradation': degradation,
            'model_comparison': comparison_data,
            'summary': {
                'status': 'good' if (current and current.mape < 15) else 
                          ('warning' if (current and current.mape < 30) else 'poor'),
                'total_predictions': len(self.predictions),
                'tracked_models': list(set(p['model_name'] for p in self.predictions))
            }
        }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            logger.info(f"Accuracy dashboard data saved to {output_path}")
        
        return dashboard_data


def integrate_with_forecaster(tracker: AccuracyTracker, forecaster_func, price_data: pd.DataFrame):
    """
    Example integration with forecasting module.
    
    Shows how to log predictions from the forecasting system.
    
    Args:
        tracker: AccuracyTracker instance
        forecaster_func: Function that returns predictions
        price_data: Historical price data
    """
    # Get predictions from forecaster
    predictions = forecaster_func(price_data)
    
    # Log each prediction
    for pred in predictions:
        tracker.log_prediction(
            predicted_value=pred['value'],
            model_name=pred.get('model', 'ensemble'),
            region=pred.get('region', 'NSW1'),
            prediction_horizon=pred.get('horizon', '1h')
        )


def print_accuracy_report(tracker: AccuracyTracker):
    """Print formatted accuracy report."""
    print("\n" + "=" * 60)
    print("  FORECAST ACCURACY REPORT")
    print("=" * 60)
    
    current = tracker.calculate_accuracy_metrics(hours=24)
    
    if current:
        print(f"\n  24-Hour Accuracy:")
        print(f"    MAPE: {current.mape}%")
        print(f"    MAE: ${current.mae:.2f}")
        print(f"    RMSE: ${current.rmse:.2f}")
        print(f"    Median APE: {current.median_ape}%")
        print(f"    Bias: ${current.bias:.2f}")
        print(f"    Predictions: {current.prediction_count}")
    else:
        print("\n  Insufficient data for 24-hour accuracy")
    
    # Degradation check
    degradation = tracker.detect_accuracy_degradation()
    print(f"\n  Accuracy Status: {degradation['status'].upper()}")
    if degradation.get('pct_change') is not None:
        print(f"    Change from baseline: {degradation['pct_change']:+.1f}%")
    print(f"    {degradation.get('recommendation', '')}")
    
    # Model comparison
    comparison = tracker.get_model_comparison()
    if not comparison.empty:
        print(f"\n  Model Comparison (7 days):")
        for model, metrics in comparison.iterrows():
            print(f"    {model}: MAPE={metrics['mape']:.1f}%, n={int(metrics['count'])}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Demo usage
    tracker = AccuracyTracker()
    
    # Log some sample predictions
    models = ['persistence', 'arima', 'ensemble']
    
    print("Logging sample predictions...")
    for i in range(20):
        predicted = np.random.uniform(50, 150)
        actual = predicted + np.random.normal(0, 20)
        
        model = np.random.choice(models)
        
        record = tracker.log_prediction(
            predicted_value=predicted,
            actual_value=actual,
            model_name=model,
            region='NSW1',
            prediction_horizon='1h'
        )
    
    # Create snapshot
    snapshot = tracker.create_daily_snapshot()
    
    # Print report
    print_accuracy_report(tracker)
    
    # Generate dashboard data
    output_path = Path(__file__).parent.parent / "dashboard" / "public" / "accuracy_tracking.json"
    tracker.generate_dashboard_data(output_path)
    print(f"\nDashboard data saved to {output_path}")
