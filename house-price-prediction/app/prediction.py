import joblib
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
import uuid
from .config import settings
from .models import HouseFeatures, PredictionResponse

logger = logging.getLogger(__name__)

class HousePricePredictor:
    """House price prediction service"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model and scaler"""
        try:
            if not settings.MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {settings.MODEL_PATH}")
            
            # Load model (assuming it's a dict containing model and scaler)
            model_data = joblib.load(settings.MODEL_PATH)
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            
            logger.info(f"Model loaded successfully from {settings.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    async def predict(self, features: HouseFeatures) -> PredictionResponse:
        """Make prediction based on input features"""
        try:
            # Convert features to numpy array
            feature_array = np.array([
                features.median_income,
                features.house_age,
                features.total_rooms,
                features.total_bedrooms,
                features.population,
                features.households,
                features.latitude,
                features.longitude
            ]).reshape(1, -1)
            
            # Apply scaler if available
            if self.scaler:
                feature_array = self.scaler.transform(feature_array)
            
            # Make prediction
            prediction = self.model.predict(feature_array)[0]
            
            # Generate prediction ID
            prediction_id = str(uuid.uuid4())
            
            return PredictionResponse(
                predicted_price=float(prediction),
                prediction_id=prediction_id,
                status="success",
                confidence_score=0.85  # Optional confidence metric
            )
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

# Initialize predictor
predictor = HousePricePredictor()
