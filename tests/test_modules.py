"""
Unit tests for the data science modules.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestEDA:
    """Tests for EDA module."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample price data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        np.random.seed(42)
        prices = np.random.normal(100, 50, 100)
        prices[10] = 500
        prices[20] = -50
        return pd.DataFrame({
            'SETTLEMENTDATE': dates,
            'RRP': prices,
            'REGIONID': 'SA1'
        })
    
    def test_price_distribution_analysis(self, sample_df):
        from src.eda import price_distribution_analysis
        result = price_distribution_analysis(sample_df)
        
        assert 'mean' in result
        assert 'median' in result
        assert 'std' in result
        assert 'skewness' in result
        assert isinstance(result['mean'], float)
    
    def test_price_distribution_missing_column(self):
        from src.eda import price_distribution_analysis
        df = pd.DataFrame({'OTHER': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="must contain 'RRP'"):
            price_distribution_analysis(df)
    
    def test_volatility_analysis(self, sample_df):
        from src.eda import volatility_analysis
        result = volatility_analysis(sample_df, window=10)
        
        assert 'rolling_std' in result.columns
        assert 'rolling_mean' in result.columns
        assert len(result) == len(sample_df)
    
    def test_temporal_patterns(self, sample_df):
        from src.eda import temporal_patterns
        result = temporal_patterns(sample_df)
        
        assert 'hourly_stats' in result
        assert 'peak_avg' in result
        assert 'offpeak_avg' in result
    
    def test_outlier_analysis(self, sample_df):
        from src.eda import outlier_analysis
        result = outlier_analysis(sample_df, spike_threshold=300)
        
        assert result['spike_count'] >= 1
        assert result['negative_count'] >= 1
        assert 'spike_events' in result


class TestForecasting:
    """Tests for forecasting module."""
    
    @pytest.fixture
    def sample_prices(self):
        np.random.seed(42)
        return np.random.normal(100, 20, 100)
    
    def test_rolling_mean_predictor(self, sample_prices):
        from src.forecasting import RollingMeanPredictor
        predictor = RollingMeanPredictor(window=5)
        predictions = predictor.predict(sample_prices)
        
        assert len(predictions) == len(sample_prices)
        assert np.isnan(predictions[0])
        assert not np.isnan(predictions[10])
    
    def test_rolling_mean_invalid_window(self):
        from src.forecasting import RollingMeanPredictor
        with pytest.raises(ValueError):
            RollingMeanPredictor(window=0)
    
    def test_ema_predictor(self, sample_prices):
        from src.forecasting import EMAPredictor
        predictor = EMAPredictor(span=10)
        predictions = predictor.predict(sample_prices)
        
        assert len(predictions) == len(sample_prices)
    
    def test_linear_trend_predictor(self, sample_prices):
        from src.forecasting import LinearTrendPredictor
        predictor = LinearTrendPredictor(window=10)
        predictions = predictor.predict(sample_prices)
        
        assert len(predictions) == len(sample_prices)
    
    def test_evaluate_forecast(self, sample_prices):
        from src.forecasting import evaluate_forecast, RollingMeanPredictor
        predictor = RollingMeanPredictor(window=5)
        predictions = predictor.predict(sample_prices)
        
        metrics = evaluate_forecast(sample_prices, predictions)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert metrics['mae'] >= 0


class TestMetrics:
    """Tests for metrics module."""
    
    @pytest.fixture
    def sample_result_df(self):
        n = 100
        return pd.DataFrame({
            'action': ['charge']*30 + ['discharge']*30 + ['hold']*40,
            'profit': list(np.random.uniform(-10, 20, 60)) + [0]*40,
            'cumulative_profit': np.cumsum(list(np.random.uniform(-10, 20, 60)) + [0]*40)
        })
    
    def test_sharpe_ratio(self):
        from src.metrics import sharpe_ratio
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        sr = sharpe_ratio(returns)
        
        assert isinstance(sr, float)
    
    def test_sharpe_ratio_zero_std(self):
        from src.metrics import sharpe_ratio
        returns = np.array([1.0, 1.0, 1.0])
        sr = sharpe_ratio(returns)
        
        assert sr == 0.0
    
    def test_win_rate(self, sample_result_df):
        from src.metrics import win_rate
        result = win_rate(sample_result_df)
        
        assert 'win_rate' in result
        assert 'total_trades' in result
        assert result['total_trades'] == 60
    
    def test_profit_factor(self, sample_result_df):
        from src.metrics import profit_factor
        pf = profit_factor(sample_result_df)
        
        assert pf >= 0
    
    def test_max_drawdown(self):
        from src.metrics import max_drawdown
        cumulative = np.array([0, 10, 20, 15, 25, 20, 30])
        mdd = max_drawdown(cumulative)
        
        assert mdd == 5.0


class TestMLForecasters:
    """Tests for ML-based forecasters."""
    
    @pytest.fixture
    def sample_prices(self):
        np.random.seed(42)
        return np.random.normal(100, 20, 200)
    
    def test_xgboost_predictor(self, sample_prices):
        from src.ml_forecasting import XGBoostPredictor
        predictor = XGBoostPredictor(lookback=12)
        predictions = predictor.fit_predict(sample_prices)
        
        assert len(predictions) == len(sample_prices)
    
    def test_random_forest_predictor(self, sample_prices):
        from src.ml_forecasting import RandomForestPredictor
        predictor = RandomForestPredictor(lookback=12)
        predictions = predictor.fit_predict(sample_prices)
        
        assert len(predictions) == len(sample_prices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
