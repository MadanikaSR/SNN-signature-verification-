import cv2
import numpy as np


def preprocess_bytes(raw_bytes: bytes, img_size: int = 128) -> np.ndarray:
    """
    Convert raw image bytes → preprocessed numpy array ready for model input.

    Pipeline:
      1. Decode bytes → OpenCV grayscale image
      2. Resize to (img_size, img_size)
      3. Otsu binarisation (inverted) to isolate ink strokes
      4. Normalise to [0, 1] float32
      5. Add channel dimension → shape (img_size, img_size, 1)
    """
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image. Ensure it is a valid PNG/JPG.")

    img = cv2.resize(img, (img_size, img_size))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    return img
