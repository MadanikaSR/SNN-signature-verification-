import os
import cv2
import numpy as np
import random
import tensorflow as tf
from keras import layers, Model, Input

IMG_SIZE = (128, 128)
DATASET_ROOT = r"folder"
UPLOADED_FILE_PATH = "path"

def preprocess_image(img_path, size=IMG_SIZE):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, size)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def load_dataset_paths(root_dir):
    dataset = {}
    for subject in range(1, 56):
        subject_path = os.path.join(root_dir, str(subject))
        if not os.path.isdir(subject_path):
            continue
        originals = []
        forgeries = []
        for fname in sorted(os.listdir(subject_path)):
            fpath = os.path.join(subject_path, fname)
            if not os.path.isfile(fpath):
                continue
            name = fname.lower()
            if "original" in name:
                originals.append(fpath)
            elif "forgeries" in name or "forg" in name:
                forgeries.append(fpath)
        if len(originals) > 0 and len(forgeries) > 0:
            dataset[subject] = {"original": originals, "forgery": forgeries}
            print(f"Loaded Subject {subject}: {len(originals)} originals, {len(forgeries)} forgeries")
    return dataset

def pair_generator_fn(dataset):
    subjects = list(dataset.keys())
    while True:
        subj = random.choice(subjects)
        originals = dataset[subj]["original"]
        forgeries = dataset[subj]["forgery"]
        if random.random() < 0.5:
            if len(originals) < 2:
                continue
            a, b = random.sample(originals, 2)
            img1 = preprocess_image(a)
            img2 = preprocess_image(b)
            label = np.int32(1)
            yield (img1, img2), label
        else:
            if len(originals) < 1 or len(forgeries) < 1:
                continue
            a = random.choice(originals)
            b = random.choice(forgeries)
            img1 = preprocess_image(a)
            img2 = preprocess_image(b)
            label = np.int32(0)
            yield (img1, img2), label

def make_dataset(dataset, batch_size=32, buffer_size=512):
    output_signature = (
        (
            tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(lambda: pair_generator_fn(dataset), output_signature=output_signature)
    ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def build_cnn_encoder(input_shape=(128,128,1), embedding_dim=128):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    model = Model(inputs=inp, outputs=x, name="cnn_encoder")
    return model

def build_siamese_network(input_shape=(128,128,1), embedding_dim=128):
    encoder = build_cnn_encoder(input_shape, embedding_dim)
    i1 = Input(shape=input_shape)
    i2 = Input(shape=input_shape)
    e1 = encoder(i1)
    e2 = encoder(i2)
    diff = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([e1, e2])
    out = layers.Dense(1, activation="sigmoid")(diff)
    model = Model(inputs=[i1, i2], outputs=out, name="siamese")
    return model, encoder

if __name__ == "__main__":
    dataset = load_dataset_paths(DATASET_ROOT)
    if len(dataset) == 0:
        raise RuntimeError("No subjects found in dataset path. Check DATASET_ROOT.")

    batch_size = 32
    train_ds = make_dataset(dataset, batch_size=batch_size)
    val_ds = make_dataset(dataset, batch_size=batch_size)

    siamese, encoder = build_siamese_network()
    siamese.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    siamese.summary()

    steps_per_epoch = 300
    validation_steps = 50
    epochs = 12

    siamese.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps, epochs=epochs)

    model_path = "siamese_signature_model.h5"
    encoder_path = "signature_encoder.h5"
    siamese.save(model_path)
    encoder.save(encoder_path)
    print(f"Saved siamese model to: {model_path}")
    print(f"Saved encoder to: {encoder_path}")

    test_batch, test_labels = next(iter(make_dataset(dataset, batch_size=10)))
    preds = siamese.predict(test_batch)
    print("\nTEST RESULTS ON 10 RANDOM PAIRS")
    for i in range(len(preds)):
        print(f"Pair {i+1}: pred={float(preds[i][0]):.4f} label={int(test_labels[i])}")

    print("\nUploaded file path (use this as file URL if needed):")
    print(UPLOADED_FILE_PATH)
