import os
from pathlib import Path

class Settings:
    """Application settings"""
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", "/app/models/house_price_model.pkl"))
    DATA_VOLUME_PATH: Path = Path(os.getenv("DATA_VOLUME_PATH", "/data"))
    API_VERSION: str = "v1"
    MAX_REQUEST_SIZE: int = 1024 * 1024  # 1MB
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
settings = Settings()
