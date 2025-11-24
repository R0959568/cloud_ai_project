"""
UK Electricity Demand Prediction API - Backend
===============================================
FastAPI backend that loads the Gradient Boosting model and serves predictions
with 39 enhanced features including lag and rolling statistics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="UK Electricity Demand Prediction API",
    description="Predict UK electricity demand using Gradient Boosting with 39 enhanced features",
    version="1.0.0"
)

# CORS - allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and data variables
model = None
historical_data = None

# Request schema
class PredictionRequest(BaseModel):
    prediction_datetime: str  # ISO format: "2024-01-15T14:30:00"
    
    class Config:
        schema_extra = {
            "example": {
                "prediction_datetime": "2024-06-15T14:30:00"
            }
        }

# Response schema
class PredictionResponse(BaseModel):
    predicted_demand_mw: float
    prediction_datetime: str
    features_used: dict
    message: str

def compute_enhanced_features(prediction_datetime: datetime, historical_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute all 39 enhanced features needed for the model
    """
    # Extract basic temporal features
    year = prediction_datetime.year
    month = prediction_datetime.month
    day = prediction_datetime.day
    hour = prediction_datetime.hour
    day_of_week = prediction_datetime.weekday()
    quarter = (month - 1) // 3 + 1
    week_of_year = prediction_datetime.isocalendar()[1]
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Enhanced temporal features
    is_business_hours = 1 if (hour >= 8 and hour <= 18 and is_weekend == 0) else 0
    is_night = 1 if (hour >= 23 or hour <= 5) else 0
    is_peak_morning = 1 if (hour >= 7 and hour <= 9) else 0
    is_peak_evening = 1 if (hour >= 17 and hour <= 20) else 0
    
    # Season
    season = {12: 0, 1: 0, 2: 0,  # Winter
              3: 1, 4: 1, 5: 1,   # Spring
              6: 2, 7: 2, 8: 2,   # Summer
              9: 3, 10: 3, 11: 3  # Autumn
    }[month]
    
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # UK Bank Holidays
    uk_holidays = [
        '2023-01-01', '2023-01-02', '2023-04-07', '2023-04-10', '2023-05-01', '2023-05-29', 
        '2023-08-28', '2023-12-25', '2023-12-26',
        '2024-01-01', '2024-03-29', '2024-04-01', '2024-05-06', '2024-05-27', 
        '2024-08-26', '2024-12-25', '2024-12-26',
        '2025-01-01', '2025-04-18', '2025-04-21', '2025-05-05', '2025-05-26', 
        '2025-08-25', '2025-12-25', '2025-12-26',
        '2026-01-01', '2026-04-03', '2026-04-06', '2026-05-04', '2026-05-25', 
        '2026-08-31', '2026-12-25', '2026-12-28',
    ]
    
    date_str = prediction_datetime.strftime('%Y-%m-%d')
    is_holiday = 1 if date_str in uk_holidays else 0
    
    day_before = (prediction_datetime - timedelta(days=1)).strftime('%Y-%m-%d')
    day_after = (prediction_datetime + timedelta(days=1)).strftime('%Y-%m-%d')
    is_day_before_holiday = 1 if day_before in uk_holidays else 0
    is_day_after_holiday = 1 if day_after in uk_holidays else 0
    
    # Interaction features
    weekend_hour = is_weekend * hour
    holiday_hour = is_holiday * hour
    month_hour = month * hour
    
    # Lag features - use realistic typical demand based on hour and season
    hour_demand_profile = {
        0: 23000, 1: 21000, 2: 20000, 3: 19500, 4: 19000, 5: 20000,
        6: 24000, 7: 30000, 8: 35000, 9: 37000, 10: 38000, 11: 38500,
        12: 38000, 13: 37500, 14: 37000, 15: 36500, 16: 37000, 17: 39000,
        18: 41000, 19: 42000, 20: 40000, 21: 37000, 22: 32000, 23: 27000
    }
    
    typical_demand = hour_demand_profile.get(hour, 35000)
    
    # Seasonal adjustment
    if season == 0:  # Winter
        typical_demand *= 1.15
    elif season == 2:  # Summer
        typical_demand *= 0.85
    elif season == 3:  # Autumn
        typical_demand *= 1.05
    
    if is_weekend:
        typical_demand *= 0.90
    
    # Calculate lag features
    demand_lag_1 = typical_demand * 0.98
    demand_lag_1d = typical_demand * 1.0
    demand_lag_3h = hour_demand_profile.get((hour - 3) % 24, 35000) * (1.15 if season == 0 else 0.85 if season == 2 else 1.0)
    demand_lag_7d = typical_demand * 1.0
    demand_rolling_mean_24h = typical_demand * 0.98
    demand_rolling_std_24h = 2500
    demand_rolling_mean_7d = typical_demand * 0.99
    demand_diff_from_24h_avg = typical_demand * 0.02
    
    # If historical data available, use actual values
    if historical_df is not None:
        hist_before = historical_df[historical_df['datetime'] < prediction_datetime]
        
        if len(hist_before) > 0:
            if len(hist_before) >= 1:
                demand_lag_1 = hist_before.iloc[-1]['demand_value']
            
            lag_1d_time = prediction_datetime - timedelta(hours=24)
            lag_1d_record = hist_before[hist_before['datetime'] <= lag_1d_time]
            if len(lag_1d_record) > 0:
                demand_lag_1d = lag_1d_record.iloc[-1]['demand_value']
            
            lag_3h_time = prediction_datetime - timedelta(hours=3)
            lag_3h_record = hist_before[hist_before['datetime'] <= lag_3h_time]
            if len(lag_3h_record) > 0:
                demand_lag_3h = lag_3h_record.iloc[-1]['demand_value']
            
            lag_7d_time = prediction_datetime - timedelta(days=7)
            lag_7d_record = hist_before[hist_before['datetime'] <= lag_7d_time]
            if len(lag_7d_record) > 0:
                demand_lag_7d = lag_7d_record.iloc[-1]['demand_value']
            
            rolling_24h_time = prediction_datetime - timedelta(hours=24)
            rolling_24h_data = hist_before[hist_before['datetime'] >= rolling_24h_time]
            if len(rolling_24h_data) > 0:
                demand_rolling_mean_24h = rolling_24h_data['demand_value'].mean()
                demand_rolling_std_24h = rolling_24h_data['demand_value'].std()
                if pd.isna(demand_rolling_std_24h):
                    demand_rolling_std_24h = 0
                demand_diff_from_24h_avg = demand_lag_1 - demand_rolling_mean_24h
            
            rolling_7d_time = prediction_datetime - timedelta(days=7)
            rolling_7d_data = hist_before[hist_before['datetime'] >= rolling_7d_time]
            if len(rolling_7d_data) > 0:
                demand_rolling_mean_7d = rolling_7d_data['demand_value'].mean()
    
    # Feature names in exact order
    feature_names = [
        'year', 'month', 'day', 'hour', 'day_of_week', 'quarter', 'week_of_year',
        'is_weekend', 'is_business_hours', 'is_night', 'is_peak_morning', 'is_peak_evening',
        'season',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
        'demand_lag_1', 'demand_lag_1d', 'demand_lag_3h', 'demand_lag_7d',
        'demand_rolling_mean_24h', 'demand_rolling_std_24h', 'demand_rolling_mean_7d',
        'demand_diff_from_24h_avg',
        'is_holiday', 'is_day_before_holiday', 'is_day_after_holiday',
        'weekend_hour', 'holiday_hour', 'month_hour'
    ]
    
    feature_values = [
        year, month, day, hour, day_of_week, quarter, week_of_year,
        is_weekend, is_business_hours, is_night, is_peak_morning, is_peak_evening,
        season,
        hour_sin, hour_cos, month_sin, month_cos, day_of_week_sin, day_of_week_cos,
        demand_lag_1, demand_lag_1d, demand_lag_3h, demand_lag_7d,
        demand_rolling_mean_24h, demand_rolling_std_24h, demand_rolling_mean_7d,
        demand_diff_from_24h_avg,
        is_holiday, is_day_before_holiday, is_day_after_holiday,
        weekend_hour, holiday_hour, month_hour
    ]
    
    return pd.DataFrame([feature_values], columns=feature_names)

@app.on_event("startup")
async def load_model_and_data():
    """Load the trained model and historical data on startup"""
    global model, historical_data
    try:
        # Try multiple paths for model
        model_paths = [
            Path("gradient_boosting_enhanced.pkl"),     # Same directory (Railway)
            Path("../../dataset_2_electricity_app/data/final/models/gradient_boosting_enhanced.pkl"),
            Path("/app/dataset_2_electricity_app/data/final/models/gradient_boosting_enhanced.pkl"),
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if model_path.exists():
                model = joblib.load(model_path)
                logger.info(f"✅ Model loaded from {model_path}")
                model_loaded = True
                break
        
        if not model_loaded:
            logger.error(f"❌ Model not found in paths: {model_paths}")
            raise FileNotFoundError("Model file not found")
        
        # Try multiple paths for historical data
        data_paths = [
            Path("elec_cleaned_full_sample.csv"),  # Same directory
            Path("../../dataset_2_electricity_app/data/interim/elec_cleaned_full.parquet"),
            Path("/app/dataset_2_electricity_app/data/interim/elec_cleaned_full.parquet"),
            Path("../../dataset_2_electricity_app/data/interim/elec_cleaned_full_sample.csv"),
            Path("/app/dataset_2_electricity_app/data/interim/elec_cleaned_full_sample.csv"),
        ]
        
        data_loaded = False
        for data_path in data_paths:
            if data_path.exists():
                if data_path.suffix == '.parquet':
                    historical_data = pd.read_parquet(data_path)
                else:
                    historical_data = pd.read_csv(data_path)
                historical_data['datetime'] = pd.to_datetime(historical_data['settlement_date'])
                historical_data = historical_data.dropna(subset=['datetime', 'demand_value'])
                historical_data = historical_data.sort_values('datetime').reset_index(drop=True)
                logger.info(f"✅ Historical data loaded from {data_path}: {len(historical_data):,} records")
                data_loaded = True
                break
        
        if not data_loaded:
            logger.warning("⚠️ Historical data not found. Using typical demand values.")
            
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading model/data: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "UK Electricity Demand Prediction API",
        "model_loaded": model is not None,
        "historical_data_loaded": historical_data is not None,
        "historical_records": len(historical_data) if historical_data is not None else 0
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not loaded",
        "data_status": "loaded" if historical_data is not None else "not loaded",
        "version": "1.0.0",
        "features": 39
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest):
    """
    Predict electricity demand for a specific datetime
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Parse datetime
        prediction_datetime = datetime.fromisoformat(request.prediction_datetime)
        logger.info(f"Processing prediction for {prediction_datetime}")
        
        # Compute features
        logger.info("Computing enhanced features...")
        features = compute_enhanced_features(prediction_datetime, historical_data)
        logger.info(f"Features computed: {features.shape}")
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = float(model.predict(features)[0])
        
        logger.info(f"✅ Prediction for {prediction_datetime}: {prediction:,.0f} MW")
        
        # Return key features for transparency
        key_features = {
            "year": int(features['year'].values[0]),
            "month": int(features['month'].values[0]),
            "hour": int(features['hour'].values[0]),
            "is_weekend": bool(features['is_weekend'].values[0]),
            "season": int(features['season'].values[0]),
            "demand_lag_1d": float(features['demand_lag_1d'].values[0]),
            "rolling_mean_24h": float(features['demand_rolling_mean_24h'].values[0])
        }
        
        return PredictionResponse(
            predicted_demand_mw=prediction,
            prediction_datetime=prediction_datetime.isoformat(),
            features_used=key_features,
            message="Prediction successful"
        )
        
    except ValueError as e:
        logger.error(f"❌ Invalid datetime format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Prediction error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(e).__name__}: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    date_range = {}
    if historical_data is not None:
        date_range = {
            "min_date": historical_data['datetime'].min().isoformat(),
            "max_date": historical_data['datetime'].max().isoformat(),
            "total_records": len(historical_data)
        }
    
    return {
        "model_type": "Gradient Boosting Regressor",
        "features_count": 39,
        "feature_categories": {
            "temporal": ["year", "month", "day", "hour", "day_of_week", "quarter", "week_of_year"],
            "binary_indicators": ["is_weekend", "is_business_hours", "is_night", "is_peak_morning", "is_peak_evening"],
            "cyclical": ["hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos"],
            "lag_features": ["demand_lag_1", "demand_lag_1d", "demand_lag_3h", "demand_lag_7d"],
            "rolling_stats": ["demand_rolling_mean_24h", "demand_rolling_std_24h", "demand_rolling_mean_7d", "demand_diff_from_24h_avg"],
            "holidays": ["is_holiday", "is_day_before_holiday", "is_day_after_holiday"],
            "interactions": ["weekend_hour", "holiday_hour", "month_hour"]
        },
        "training_data": "UK electricity demand 2001-2025",
        "performance": {
            "r2_score": 0.70,
            "mae_mw": 2353.23,
            "rmse_mw": 3107.24
        },
        "historical_data": date_range
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
