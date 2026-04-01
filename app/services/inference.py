import numpy as np
import logging
from typing import Optional, Tuple

import tensorflow as tf

from app.core.config import settings
from app.models.loader import load_model
from app.utils.preprocessing import preprocess_bytes

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Singleton service that holds the loaded model and runs predictions.
    Model is loaded once at application startup via FastAPI lifespan.
    """

    def __init__(self) -> None:
        self._model: Optional[tf.keras.Model] = None
        self._loaded: bool = False

    def load(self) -> None:
        """Load the Siamese model from disk. Called once at startup."""
        try:
            self._model = load_model(settings.MODEL_PATH)
            self._loaded = True
            logger.info("Inference service ready.")
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(
        self,
        img1_bytes: bytes,
        img2_bytes: bytes,
    ) -> Tuple[float, bool, float]:
        """
        Run signature verification on two images.

        Returns:
            similarity_score (float): Raw model output [0, 1]
            is_match (bool):          True if score >= MATCH_THRESHOLD
            confidence (float):       Human-readable confidence % [0, 100]
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model is not loaded. Cannot run inference.")

        img_size = settings.IMG_SIZE
        img1 = preprocess_bytes(img1_bytes, img_size)
        img2 = preprocess_bytes(img2_bytes, img_size)

        # Add batch dimension: (1, H, W, 1)
        batch1 = np.expand_dims(img1, axis=0)
        batch2 = np.expand_dims(img2, axis=0)

        raw_score: float = float(self._model.predict([batch1, batch2], verbose=0)[0][0])

        is_match = raw_score >= settings.MATCH_THRESHOLD
        # Confidence: distance from threshold, scaled to [0, 100]
        if is_match:
            confidence = ((raw_score - settings.MATCH_THRESHOLD) / (1.0 - settings.MATCH_THRESHOLD)) * 100
        else:
            confidence = ((settings.MATCH_THRESHOLD - raw_score) / settings.MATCH_THRESHOLD) * 100

        return round(raw_score, 4), is_match, round(confidence, 2)


# Module-level singleton
inference_service = InferenceService()
