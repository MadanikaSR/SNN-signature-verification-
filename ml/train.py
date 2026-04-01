"""
ml/train.py — Clean Siamese Network Training Script
====================================================
Usage:
    python -m ml.train

Environment Variables (via .env or shell):
    CEDAR_ROOT              Path to CEDAR dataset root directory
    MODEL_PATH              Where to save the siamese model (.h5)
    ENCODER_PATH            Where to save the encoder model (.h5)
    TRAIN_EPOCHS            Number of training epochs (default: 12)
    TRAIN_BATCH_SIZE        Batch size (default: 32)
    TRAIN_STEPS_PER_EPOCH   Steps per epoch (default: 300)
    TRAIN_VALIDATION_STEPS  Validation steps (default: 50)
    IMG_SIZE                Image dimension (default: 128)
"""

import os
import random
import logging
import sys

import cv2
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Config from environment ─────────────────────────────────────────────────
CEDAR_ROOT: str = os.getenv("CEDAR_ROOT", "./CEDAR")
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/siamese_signature_model.h5")
ENCODER_PATH: str = os.getenv("ENCODER_PATH", "models/signature_encoder.h5")
IMG_SIZE: int = int(os.getenv("IMG_SIZE", "128"))
EPOCHS: int = int(os.getenv("TRAIN_EPOCHS", "12"))
BATCH_SIZE: int = int(os.getenv("TRAIN_BATCH_SIZE", "32"))
STEPS_PER_EPOCH: int = int(os.getenv("TRAIN_STEPS_PER_EPOCH", "300"))
VALIDATION_STEPS: int = int(os.getenv("TRAIN_VALIDATION_STEPS", "50"))

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)


# ─── Preprocessing ───────────────────────────────────────────────────────────
def preprocess_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)


# ─── Dataset Loading ──────────────────────────────────────────────────────────
def load_dataset_paths(root_dir: str) -> dict:
    dataset = {}
    for subject in range(1, 56):
        subject_path = os.path.join(root_dir, str(subject))
        if not os.path.isdir(subject_path):
            continue
        originals, forgeries = [], []
        for fname in sorted(os.listdir(subject_path)):
            fpath = os.path.join(subject_path, fname)
            if not os.path.isfile(fpath):
                continue
            name = fname.lower()
            if "original" in name:
                originals.append(fpath)
            elif "forgeries" in name or "forg" in name:
                forgeries.append(fpath)
        if originals and forgeries:
            dataset[subject] = {"original": originals, "forgery": forgeries}
            logger.info(f"Subject {subject}: {len(originals)} originals, {len(forgeries)} forgeries")
    return dataset


def pair_generator_fn(dataset: dict):
    subjects = list(dataset.keys())
    while True:
        subj = random.choice(subjects)
        originals = dataset[subj]["original"]
        forgeries = dataset[subj]["forgery"]
        if random.random() < 0.5:
            if len(originals) < 2:
                continue
            a, b = random.sample(originals, 2)
            yield (preprocess_image(a), preprocess_image(b)), np.int32(1)
        else:
            if not originals or not forgeries:
                continue
            a = random.choice(originals)
            b = random.choice(forgeries)
            yield (preprocess_image(a), preprocess_image(b)), np.int32(0)


def make_dataset(dataset: dict, batch_size: int = 32, buffer_size: int = 512):
    output_signature = (
        (
            tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: pair_generator_fn(dataset), output_signature=output_signature
    )
    return ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ─── Custom Layers (replaces Lambda — fixes TF2.16+/Keras3 compatibility) ─────
class L2NormalizeLayer(tf.keras.layers.Layer):
    """L2-normalizes input along axis=1. Replaces Lambda for Keras 3 compat."""
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


class AbsDiffLayer(tf.keras.layers.Layer):
    """Computes element-wise absolute difference between two tensors."""
    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)


# ─── Model Architecture ───────────────────────────────────────────────────────
def build_cnn_encoder(input_shape=IMG_SHAPE, embedding_dim: int = 128) -> tf.keras.Model:
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(embedding_dim)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = L2NormalizeLayer()(x)
    return tf.keras.Model(inputs=inp, outputs=x, name="cnn_encoder")


def build_siamese_network(input_shape=IMG_SHAPE, embedding_dim: int = 128):
    encoder = build_cnn_encoder(input_shape, embedding_dim)
    i1 = tf.keras.Input(shape=input_shape)
    i2 = tf.keras.Input(shape=input_shape)
    e1 = encoder(i1)
    e2 = encoder(i2)
    diff = AbsDiffLayer()([e1, e2])
    out = tf.keras.layers.Dense(1, activation="sigmoid")(diff)
    model = tf.keras.Model(inputs=[i1, i2], outputs=out, name="siamese")
    return model, encoder


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"CEDAR dataset root: {CEDAR_ROOT}")
    dataset = load_dataset_paths(CEDAR_ROOT)
    if not dataset:
        raise RuntimeError(
            f"No subjects found in '{CEDAR_ROOT}'. "
            "Set CEDAR_ROOT env variable to the correct path."
        )

    logger.info(f"Loaded {len(dataset)} subjects. Building model...")
    train_ds = make_dataset(dataset, batch_size=BATCH_SIZE)
    val_ds = make_dataset(dataset, batch_size=BATCH_SIZE)

    siamese, encoder = build_siamese_network()
    siamese.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    siamese.summary()

    # ─── Callbacks ───────────────────────────────────────────────────────────
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    logger.info(f"Training for {EPOCHS} epochs, {STEPS_PER_EPOCH} steps/epoch...")
    siamese.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_ds,
        validation_steps=VALIDATION_STEPS,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    # Save final versions including encoder
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    siamese.save(MODEL_PATH)
    encoder.save(ENCODER_PATH)
    logger.info(f"Siamese model saved → {MODEL_PATH}")
    logger.info(f"Encoder saved       → {ENCODER_PATH}")

    # Quick sanity check
    test_batch, test_labels = next(iter(make_dataset(dataset, batch_size=10)))
    preds = siamese.predict(test_batch)
    logger.info("Quick test on 10 random pairs:")
    for i, (pred, label) in enumerate(zip(preds, test_labels.numpy())):
        logger.info(f"  Pair {i+1:02d}: score={float(pred[0]):.4f}  label={int(label)}")
