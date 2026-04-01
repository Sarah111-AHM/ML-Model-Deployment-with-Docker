from pydantic import BaseModel, Field, validator
from typing import Optional
import numpy as np

class HouseFeatures(BaseModel):
    """Input features for house price prediction"""
    median_income: float = Field(..., ge=0, le=15, description="Median income in neighborhood")
    house_age: float = Field(..., ge=0, le=100, description="Age of house in years")
    total_rooms: float = Field(..., ge=1, description="Total rooms in house")
    total_bedrooms: float = Field(..., ge=0, description="Total bedrooms")
    population: float = Field(..., ge=0, description="Population in area")
    households: float = Field(..., ge=0, description="Number of households")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    
    class Config:
        schema_extra = {
            "example": {
                "median_income": 3.5,
                "house_age": 25,
                "total_rooms": 2000,
                "total_bedrooms": 400,
                "population": 1200,
                "households": 450,
                "latitude": 34.05,
                "longitude": -118.24
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response model"""
    predicted_price: float
    prediction_id: str
    status: str
    confidence_score: Optional[float] = None
