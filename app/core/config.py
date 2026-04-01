from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Model paths
    MODEL_PATH: str = "models/siamese_signature_model.h5"
    ENCODER_PATH: str = "models/signature_encoder.h5"

    # Inference
    MATCH_THRESHOLD: float = 0.5
    IMG_SIZE: int = 128
    MAX_FILE_SIZE_MB: int = 2

    # CORS — comma-separated list of allowed origins
    ALLOWED_ORIGINS: str = (
        "http://localhost:3000,"
        "http://127.0.0.1:5500,"
        "http://localhost:5500"
    )

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
