"""
Agri Vision — Training Script
==============================
Trains a MobileNetV2-based classifier for rice plant disease detection.
Classes: Healthy, Brown_Spot, Leaf_Blast, Tungro

Capstone Project by:
  Stefanus Westin G.M.  (71241073)
  Giovanni Albert Harjanto (71241075)
  Sinatrio Happy Triadji (71241093)

Usage:
    python train.py
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 4
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "agri_vision_model.h5")
SEED = 42


def build_datasets():
    """Load training and validation datasets from the dataset/ directory."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    print(f"[INFO] Detected classes: {class_names}")

    # Performance: cache, shuffle, prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model():
    """
    Build a transfer-learning model with MobileNetV2 as the feature extractor
    and a custom classification head for 4-class rice disease detection.
    """
    # Data augmentation layer
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    # MobileNetV2 base — freeze weights for transfer learning
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False

    # Full model
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="fc_hidden")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="AgriVision_MobileNetV2")
    return model


def train():
    """Main training loop."""
    print("=" * 60)
    print("  🌾 Agri Vision — Model Training")
    print("=" * 60)

    # 1. Data
    train_ds, val_ds, class_names = build_datasets()

    # 2. Model
    model = build_model()
    model.summary()

    # 3. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. Callbacks
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 5. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb_list,
    )

    # 6. Final evaluation
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\n[RESULT] Validation Loss : {val_loss:.4f}")
    print(f"[RESULT] Validation Accuracy: {val_acc:.4f}")
    print(f"[INFO]   Model saved to: {MODEL_SAVE_PATH}")
    print(f"[INFO]   Class order   : {class_names}")

    # Save class order to a text file for the Flask app
    class_file = os.path.join(os.path.dirname(__file__), "class_names.txt")
    with open(class_file, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))
    print(f"[INFO]   Class names saved to: {class_file}")

    return history


if __name__ == "__main__":
    train()
