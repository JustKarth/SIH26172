from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "features"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "kws_baseline_best.keras"
)

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.98,
    0.99,
]


# ============================================================
# LOAD
# ============================================================

def load_split(split):

    X = np.load(
        FEATURE_DIR / f"{split}_X.npy"
    )

    y = np.load(
        FEATURE_DIR / f"{split}_y.npy"
    )

    X = X[..., np.newaxis]

    return (
        X.astype(np.float32),
        y.astype(np.int32),
    )


print("=" * 70)
print("KWS THRESHOLD EVALUATION")
print("=" * 70)

print()
print(
    f"Loading model: {MODEL_PATH}"
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

# Same normalization used during training.
X_train = np.load(
    FEATURE_DIR / "train_X.npy"
)

train_mean = X_train.mean()
train_std = X_train.std()

del X_train

print(
    f"Training mean : {train_mean:.6f}"
)

print(
    f"Training std  : {train_std:.6f}"
)


# ============================================================
# METRICS
# ============================================================

def evaluate_threshold(
    probabilities,
    y,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    tp = int(
        np.sum(
            (y == 1)
            & (predictions == 1)
        )
    )

    tn = int(
        np.sum(
            (y == 0)
            & (predictions == 0)
        )
    )

    fp = int(
        np.sum(
            (y == 0)
            & (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (y == 1)
            & (predictions == 0)
        )
    )

    total = len(y)

    accuracy = (
        (tp + tn) / total
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "fpr": false_positive_rate,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# EVALUATE SPLIT
# ============================================================

def evaluate_split(
    split
):

    print()
    print("=" * 70)
    print(
        f"{split.upper()} THRESHOLD ANALYSIS"
    )
    print("=" * 70)

    X, y = load_split(
        split
    )

    X = (
        X - train_mean
    ) / train_std

    probabilities = model.predict(
        X,
        batch_size=32,
        verbose=0
    )[:, 1]

    auc = roc_auc_score(
        y,
        probabilities
    )

    ap = average_precision_score(
        y,
        probabilities
    )

    print()
    print(
        f"ROC-AUC : {auc:.6f}"
    )

    print(
        f"PR-AUC  : {ap:.6f}"
    )

    print()

    print(
        "Threshold | Recall | Precision | "
        "FPR     | Accuracy | TP | FP | TN | FN"
    )

    print(
        "-" * 85
    )

    results = []

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            probabilities,
            y,
            threshold
        )

        results.append(
            result
        )

        print(
            f"{threshold:9.2f} | "
            f"{result['recall'] * 100:6.2f}% | "
            f"{result['precision'] * 100:9.2f}% | "
            f"{result['fpr'] * 100:6.2f}% | "
            f"{result['accuracy'] * 100:8.2f}% | "
            f"{result['tp']:2d} | "
            f"{result['fp']:2d} | "
            f"{result['tn']:3d} | "
            f"{result['fn']:2d}"
        )

    return probabilities, y, results


# ============================================================
# MAIN
# ============================================================

val_prob, val_y, val_results = (
    evaluate_split("val")
)

test_prob, test_y, test_results = (
    evaluate_split("test")
)


# ============================================================
# SCORE DISTRIBUTIONS
# ============================================================

print()
print("=" * 70)
print("PROBABILITY DISTRIBUTIONS")
print("=" * 70)

for name, probabilities, y in [
    ("VAL", val_prob, val_y),
    ("TEST", test_prob, test_y),
]:

    keyword_scores = probabilities[
        y == 1
    ]

    negative_scores = probabilities[
        y == 0
    ]

    print()
    print(name)

    print(
        "Keyword probability:"
    )

    print(
        f"  min    : {keyword_scores.min():.6f}"
    )

    print(
        f"  p01    : {np.percentile(keyword_scores, 1):.6f}"
    )

    print(
        f"  p05    : {np.percentile(keyword_scores, 5):.6f}"
    )

    print(
        f"  median : {np.median(keyword_scores):.6f}"
    )

    print(
        f"  p95    : {np.percentile(keyword_scores, 95):.6f}"
    )

    print(
        f"  max    : {keyword_scores.max():.6f}"
    )

    print(
        "Negative probability:"
    )

    print(
        f"  min    : {negative_scores.min():.6f}"
    )

    print(
        f"  p95    : {np.percentile(negative_scores, 95):.6f}"
    )

    print(
        f"  p99    : {np.percentile(negative_scores, 99):.6f}"
    )

    print(
        f"  median : {np.median(negative_scores):.6f}"
    )

    print(
        f"  max    : {negative_scores.max():.6f}"
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)