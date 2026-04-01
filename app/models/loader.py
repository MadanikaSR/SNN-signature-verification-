import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


# ─── Custom Layers ─────────────────────────────────────────────────────────────
# Must match the definitions in ml/train.py so .h5 files deserialise correctly.
class L2NormalizeLayer(tf.keras.layers.Layer):
    """L2-normalizes input along axis=1."""
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


class AbsDiffLayer(tf.keras.layers.Layer):
    """Element-wise absolute difference between two tensors."""
    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)


# ─── Loader ────────────────────────────────────────────────────────────────────
def load_model(path: str) -> tf.keras.Model:
    """
    Load a saved Keras model that was built with custom layers.
    Passes custom_objects so Keras can reconstruct the architecture.
    """
    logger.info(f"Loading model from: {path}")
    model = tf.keras.models.load_model(
        path,
        custom_objects={
            "L2NormalizeLayer": L2NormalizeLayer,
            "AbsDiffLayer": AbsDiffLayer,
        },
        compile=False,
    )
    logger.info(f"Model loaded successfully: {path}")
    return model
