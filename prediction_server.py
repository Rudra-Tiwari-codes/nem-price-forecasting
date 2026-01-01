"""
FastAPI Prediction Server for NEM Price Forecasting.

Provides real-time ML predictions with error tracking.
Run: python prediction_server.py
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.ml_forecasting import XGBoostPredictor

# Configuration
DATA_FILE = "data/combined_dispatch_prices.csv"
PREDICTIONS_FILE = "data/predictions_history.json"
LOOKBACK = 12  # Number of past intervals for features
PREDICTION_HORIZON = 3  # 3 intervals = 15 minutes

app = FastAPI(
    title="NEM Price Prediction API",
    description="ML-powered electricity price predictions with error tracking",
    version="1.0.0"
)

# Enable CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
models: Dict[str, XGBoostPredictor] = {}
last_train_time: Optional[datetime] = None
predictions_history: Dict[str, List[dict]] = {}

REGIONS = ["SA1", "NSW1", "VIC1", "QLD1", "TAS1"]


def load_data() -> pd.DataFrame:
    """Load price data from CSV."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    
    df = pd.read_csv(DATA_FILE)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    return df


def get_region_prices(df: pd.DataFrame, region: str) -> np.ndarray:
    """Extract prices for a specific region."""
    region_df = df[df["REGIONID"] == region].copy()
    region_df = region_df.sort_values("SETTLEMENTDATE")
    return region_df["RRP"].values


def train_models():
    """Train XGBoost models for all regions."""
    global models, last_train_time
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training models...")
    
    df = load_data()
    
    for region in REGIONS:
        prices = get_region_prices(df, region)
        if len(prices) < LOOKBACK + 10:
            print(f"  Skipping {region}: insufficient data ({len(prices)} points)")
            continue
        
        model = XGBoostPredictor(lookback=LOOKBACK, n_estimators=100)
        model.fit_predict(prices)
        models[region] = model
        print(f"  Trained {region} model on {len(prices)} data points")
    
    last_train_time = datetime.now()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training complete")


def load_predictions_history():
    """Load prediction history from file."""
    global predictions_history
    
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r") as f:
            predictions_history = json.load(f)
    else:
        predictions_history = {region: [] for region in REGIONS}


def save_predictions_history():
    """Save prediction history to file."""
    Path(PREDICTIONS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions_history, f, indent=2, default=str)


def update_errors(region: str, df: pd.DataFrame, current_time: datetime):
    """Update error calculations for past predictions using actual prices at target times."""
    if region not in predictions_history:
        return
    
    # Get the region's data with timestamps
    region_df = df[df["REGIONID"] == region].copy()
    region_df = region_df.sort_values("SETTLEMENTDATE")
    
    updated = False
    for pred in predictions_history[region]:
        if pred.get("actual") is not None:
            continue  # Already has actual price
        
        pred_time = datetime.fromisoformat(pred["target_time"])
        
        # Check if we now have the actual price for this prediction
        if pred_time <= current_time:
            # Find the price at or closest to the target time
            # Look for prices within 5 minutes of target time
            mask = (region_df["SETTLEMENTDATE"] >= pred_time - timedelta(minutes=5)) & \
                   (region_df["SETTLEMENTDATE"] <= pred_time + timedelta(minutes=5))
            matching_prices = region_df[mask]
            
            if len(matching_prices) > 0:
                # Get the price closest to the target time
                closest_idx = (matching_prices["SETTLEMENTDATE"] - pred_time).abs().idxmin()
                actual_price = float(matching_prices.loc[closest_idx, "RRP"])
                pred["actual"] = actual_price
                pred["actual_time"] = str(matching_prices.loc[closest_idx, "SETTLEMENTDATE"])
                
                if actual_price != 0:
                    pred["error_percent"] = abs(pred["predicted"] - actual_price) / abs(actual_price) * 100
                else:
                    pred["error_percent"] = 0.0
                pred["updated_at"] = datetime.now().isoformat()
                updated = True
    
    if updated:
        # Keep only last 100 predictions per region
        predictions_history[region] = predictions_history[region][-100:]
        save_predictions_history()


@app.on_event("startup")
async def startup_event():
    """Initialize on server startup."""
    load_predictions_history()
    train_models()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "models_loaded": list(models.keys()),
        "last_trained": last_train_time.isoformat() if last_train_time else None
    }


@app.get("/predictions/{region}")
async def get_predictions(region: str):
    """Get current prediction for a region."""
    region = region.upper()
    
    if region not in REGIONS:
        raise HTTPException(status_code=400, detail=f"Invalid region. Use one of: {REGIONS}")
    
    if region not in models:
        raise HTTPException(status_code=503, detail=f"Model for {region} not loaded. Try again later.")
    
    # Load latest data
    df = load_data()
    prices = get_region_prices(df, region)
    
    if len(prices) < LOOKBACK:
        raise HTTPException(status_code=503, detail=f"Insufficient data for {region}")
    
    # Get current time and price
    region_df = df[df["REGIONID"] == region].sort_values("SETTLEMENTDATE")
    current_time = region_df["SETTLEMENTDATE"].iloc[-1]
    current_price = float(prices[-1])
    
    # Make prediction for T+15min
    model = models[region]
    recent_prices = prices[-LOOKBACK:]
    predicted_price = model.predict_next(recent_prices)
    
    target_time = current_time + timedelta(minutes=15)
    
    # Update any pending predictions with actual values
    update_errors(region, df, current_time)
    
    # Store this prediction (with deduplication check)
    prediction_record = {
        "prediction_time": datetime.now().isoformat(),
        "target_time": target_time.isoformat(),
        "predicted": float(predicted_price),
        "current_price": current_price,
        "actual": None,
        "error_percent": None
    }
    
    if region not in predictions_history:
        predictions_history[region] = []
    
    # Check if we already have a prediction for this target time (avoid duplicates)
    existing_target_times = {p["target_time"] for p in predictions_history[region]}
    if target_time.isoformat() not in existing_target_times:
        predictions_history[region].append(prediction_record)
        save_predictions_history()
    
    # Get pending predictions (waiting for actual price)
    pending = [p for p in predictions_history[region] if p.get("actual") is None]
    
    # Get recent completed predictions with errors
    completed = [p for p in predictions_history[region] if p.get("actual") is not None][-20:]
    
    return {
        "region": region,
        "current_price": current_price,
        "current_time": current_time.isoformat(),
        "prediction": {
            "target_time": target_time.isoformat(),
            "predicted_price": float(predicted_price),
            "horizon_minutes": 15
        },
        "pending_predictions": pending[-5:],  # Last 5 pending
        "completed_predictions": completed,
        "model": "XGBoost",
        "last_trained": last_train_time.isoformat() if last_train_time else None
    }


@app.get("/accuracy/{region}")
async def get_accuracy(region: str):
    """Get accuracy metrics for a region."""
    region = region.upper()
    
    if region not in REGIONS:
        raise HTTPException(status_code=400, detail=f"Invalid region. Use one of: {REGIONS}")
    
    completed = [p for p in predictions_history.get(region, []) if p.get("error_percent") is not None]
    
    if not completed:
        return {
            "region": region,
            "total_predictions": 0,
            "message": "No completed predictions yet. Wait for actual prices to arrive."
        }
    
    errors = [p["error_percent"] for p in completed]
    
    return {
        "region": region,
        "total_predictions": len(completed),
        "mape": float(np.mean(errors)),  # Mean Absolute Percentage Error
        "median_error": float(np.median(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "std_error": float(np.std(errors)),
        "recent_predictions": completed[-10:]
    }


@app.post("/retrain")
async def retrain_models():
    """Manually trigger model retraining."""
    train_models()
    return {
        "status": "success",
        "models_trained": list(models.keys()),
        "trained_at": last_train_time.isoformat()
    }


if __name__ == "__main__":
    print("Starting NEM Price Prediction Server...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
