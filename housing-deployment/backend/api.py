"""
Housing Price Prediction API - Backend
=======================================
FastAPI backend that loads the LightGBM model and serves predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="UK Housing Price Prediction API",
    description="Predict UK housing prices using LightGBM",
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

# Global model variable
model = None

# Request schema
class PredictionRequest(BaseModel):
    property_type_label: str
    is_new_build: bool
    tenure_label: str
    county: str
    district: str
    town_city: str
    year: int
    month: int
    quarter: int

    class Config:
        schema_extra = {
            "example": {
                "property_type_label": "Detached",
                "is_new_build": False,
                "tenure_label": "Freehold",
                "county": "GREATER LONDON",
                "district": "CITY OF WESTMINSTER",
                "town_city": "LONDON",
                "year": 2017,
                "month": 6,
                "quarter": 2
            }
        }

# Response schema
class PredictionResponse(BaseModel):
    predicted_price: float
    lower_bound: float
    upper_bound: float
    message: str

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model
    try:
        model_path = Path("../../models/lightgbm_housing.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("✅ Model loaded successfully!")
    except FileNotFoundError:
        logger.error("❌ Model file not found!")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "UK Housing Price Prediction API",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not loaded",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Predict housing price based on property features
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'property_type_label': [request.property_type_label],
            'is_new_build': [int(request.is_new_build)],
            'tenure_label': [request.tenure_label],
            'county': [request.county],
            'district': [request.district],
            'town_city': [request.town_city],
            'year': [request.year],
            'month': [request.month],
            'quarter': [request.quarter]
        })
        
        # Convert categoricals to category dtype (LightGBM requirement)
        cat_columns = ['property_type_label', 'tenure_label', 'county', 'district', 'town_city']
        for col in cat_columns:
            input_data[col] = input_data[col].astype('category')
        
        # Make prediction
        prediction = float(model.predict(input_data)[0])
        
        # Calculate bounds (±10%)
        lower_bound = prediction * 0.9
        upper_bound = prediction * 1.1
        
        logger.info(f"Prediction: £{prediction:,.0f}")
        
        return PredictionResponse(
            predicted_price=prediction,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            message="Prediction successful"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "LightGBM",
        "features": [
            "property_type_label",
            "is_new_build",
            "tenure_label",
            "county",
            "district",
            "town_city",
            "year",
            "month",
            "quarter"
        ],
        "training_data": "5.9M transactions (1995-2017)",
        "r2_score": "~67%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)