from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
from .models import HouseFeatures, PredictionResponse
from .prediction import predictor
from .config import settings

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/data/app.log')
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    version=settings.API_VERSION,
    description="ML-powered house price prediction service",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "house-price-prediction",
        "version": settings.API_VERSION
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check model is loaded
        if predictor.model is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "unhealthy", "error": "Model not loaded"}
            )
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_path": str(settings.MODEL_PATH)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HouseFeatures):
    """Predict house price based on features"""
    try:
        logger.info(f"Prediction request received: {features.dict()}")
        result = await predictor.predict(features)
        logger.info(f"Prediction successful: {result.prediction_id}")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(features_list: list[HouseFeatures]):
    """Batch prediction endpoint"""
    try:
        results = []
        for features in features_list:
            result = await predictor.predict(features)
            results.append(result)
        
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )
