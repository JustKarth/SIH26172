from pathlib import Path
import random

import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "features"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 26172

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3

INPUT_SHAPE = (
    98,
    40,
    1
)

NUM_CLASSES = 2


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(split):

    X = np.load(
        FEATURE_DIR / f"{split}_X.npy"
    )

    y = np.load(
        FEATURE_DIR / f"{split}_y.npy"
    )

    # Add CNN channel dimension.
    X = X[..., np.newaxis]

    return X.astype(
        np.float32
    ), y.astype(
        np.int64
    )


print("=" * 70)
print("KWS BASELINE TRAINING")
print("=" * 70)

print()
print("Loading feature data...")

X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

print()
print(
    f"Train : {X_train.shape}"
)
print(
    f"Val   : {X_val.shape}"
)
print(
    f"Test  : {X_test.shape}"
)

print()
print(
    f"Train keyword : "
    f"{np.sum(y_train == 1)}"
)

print(
    f"Train negative : "
    f"{np.sum(y_train == 0)}"
)

print(
    f"Val keyword : "
    f"{np.sum(y_val == 1)}"
)

print(
    f"Val negative : "
    f"{np.sum(y_val == 0)}"
)

print(
    f"Test keyword : "
    f"{np.sum(y_test == 1)}"
)

print(
    f"Test negative : "
    f"{np.sum(y_test == 0)}"
)


# ============================================================
# NORMALIZATION
# ============================================================

# IMPORTANT:
# Calculate normalization statistics ONLY from training data.
#
# This prevents validation/test information leaking into
# training.

train_mean = X_train.mean()
train_std = X_train.std()

print()
print(
    "TRAIN FEATURE NORMALIZATION"
)

print(
    f"Mean : {train_mean:.6f}"
)

print(
    f"Std  : {train_std:.6f}"
)

X_train = (
    X_train - train_mean
) / train_std

X_val = (
    X_val - train_mean
) / train_std

X_test = (
    X_test - train_mean
) / train_std


# ============================================================
# TF.DATA PIPELINES
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        X_train,
        y_train
    )
)

train_ds = (
    train_ds
    .shuffle(
        buffer_size=len(X_train),
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        X_val,
        y_val
    )
)

val_ds = (
    val_ds
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# MODEL
# ============================================================

def build_model():

    inputs = tf.keras.Input(
        shape=INPUT_SHAPE,
        name="log_mel_input"
    )

    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="same",
        use_bias=False
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.25
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="output"
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="kws_tiny_cnn_baseline"
    )


model = build_model()


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 70)
print("MODEL")
print("=" * 70)

model.summary()


trainable_params = (
    model.count_params()
)

print()
print(
    f"Total parameters : "
    f"{trainable_params:,}"
)

print(
    f"Approx FP32 weights : "
    f"{trainable_params * 4 / 1024:.2f} KB"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

keyword_count = np.sum(
    y_train == 1
)

negative_count = np.sum(
    y_train == 0
)

total = (
    keyword_count
    + negative_count
)

class_weight = {
    0: total / (
        2.0 * negative_count
    ),
    1: total / (
        2.0 * keyword_count
    ),
}

print()
print(
    "CLASS WEIGHTS"
)

print(
    f"Negative : "
    f"{class_weight[0]:.4f}"
)

print(
    f"Keyword  : "
    f"{class_weight[1]:.4f}"
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=(
        "sparse_categorical_crossentropy"
    ),
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = (
    MODEL_DIR
    / "kws_baseline_best.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(
            best_model_path
        ),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        mode="min",
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        mode="min",
        verbose=1
    ),
]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

val_loss, val_accuracy = (
    model.evaluate(
        val_ds,
        verbose=0
    )
)

print(
    f"Validation loss     : "
    f"{val_loss:.6f}"
)

print(
    f"Validation accuracy : "
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("TEST EVALUATION")
print("=" * 70)

test_probabilities = (
    model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )
)

test_predictions = (
    np.argmax(
        test_probabilities,
        axis=1
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

true_negative = np.sum(
    (y_test == 0)
    & (test_predictions == 0)
)

false_positive = np.sum(
    (y_test == 0)
    & (test_predictions == 1)
)

false_negative = np.sum(
    (y_test == 1)
    & (test_predictions == 0)
)

true_positive = np.sum(
    (y_test == 1)
    & (test_predictions == 1)
)

print()
print(
    "CONFUSION MATRIX"
)

print()
print(
    "                 Predicted"
)

print(
    "                 Negative  Keyword"
)

print(
    f"Actual Negative  "
    f"{true_negative:8d}  "
    f"{false_positive:7d}"
)

print(
    f"Actual Keyword   "
    f"{false_negative:8d}  "
    f"{true_positive:7d}"
)


# ============================================================
# METRICS
# ============================================================

test_accuracy = (
    (
        test_predictions
        == y_test
    ).mean()
)

if (
    true_positive
    + false_negative
) > 0:

    keyword_recall = (
        true_positive
        / (
            true_positive
            + false_negative
        )
    )

else:

    keyword_recall = 0.0


if (
    true_negative
    + false_positive
) > 0:

    negative_recall = (
        true_negative
        / (
            true_negative
            + false_positive
        )
    )

else:

    negative_recall = 0.0


if (
    true_positive
    + false_positive
) > 0:

    keyword_precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
    )

else:

    keyword_precision = 0.0


print()
print(
    "TEST METRICS"
)

print(
    f"Accuracy             : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Keyword recall       : "
    f"{keyword_recall * 100:.2f}%"
)

print(
    f"Keyword precision    : "
    f"{keyword_precision * 100:.2f}%"
)

print(
    f"Negative recall      : "
    f"{negative_recall * 100:.2f}%"
)

print(
    f"False positives      : "
    f"{false_positive}"
)

print(
    f"False negatives      : "
    f"{false_negative}"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR
    / "kws_baseline_final.keras"
)

model.save(
    final_model_path
)

print()
print(
    f"Final model saved : "
    f"{final_model_path}"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)