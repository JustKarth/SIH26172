from pathlib import Path

import numpy as np


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

SPLITS = (
    "train",
    "val",
    "test",
)


# ============================================================
# HELPERS
# ============================================================

def describe_class(
    X,
    y,
    label,
    name
):
    mask = y == label

    class_X = X[mask]

    if len(class_X) == 0:
        print(
            f"{name}: no samples"
        )
        return

    sample_means = (
        class_X.mean(axis=(1, 2))
    )

    sample_stds = (
        class_X.std(axis=(1, 2))
    )

    sample_mins = (
        class_X.min(axis=(1, 2))
    )

    sample_maxs = (
        class_X.max(axis=(1, 2))
    )

    print()
    print(
        f"{name.upper()}"
    )

    print(
        f"Samples              : "
        f"{len(class_X)}"
    )

    print(
        f"Feature mean         : "
        f"{class_X.mean():.6f}"
    )

    print(
        f"Feature std          : "
        f"{class_X.std():.6f}"
    )

    print(
        f"Feature min          : "
        f"{class_X.min():.6f}"
    )

    print(
        f"Feature max          : "
        f"{class_X.max():.6f}"
    )

    print(
        f"Per-sample mean      : "
        f"min={sample_means.min():.6f} "
        f"median={np.median(sample_means):.6f} "
        f"max={sample_means.max():.6f}"
    )

    print(
        f"Per-sample std       : "
        f"min={sample_stds.min():.6f} "
        f"median={np.median(sample_stds):.6f} "
        f"max={sample_stds.max():.6f}"
    )

    print(
        f"Per-sample min       : "
        f"min={sample_mins.min():.6f} "
        f"median={np.median(sample_mins):.6f} "
        f"max={sample_mins.max():.6f}"
    )

    print(
        f"Per-sample max       : "
        f"min={sample_maxs.min():.6f} "
        f"median={np.median(sample_maxs):.6f} "
        f"max={sample_maxs.max():.6f}"
    )


def nearest_class_margin(
    X,
    y
):
    """
    Very simple baseline separation test.

    For each sample, compare its Euclidean distance to the
    class centroid for keyword vs negative.

    This is NOT a classifier we will deploy. It is only a
    sanity check to see whether the feature representation
    contains obvious class information.
    """

    keyword = X[y == 1]
    negative = X[y == 0]

    keyword_centroid = (
        keyword.mean(axis=0)
    )

    negative_centroid = (
        negative.mean(axis=0)
    )

    keyword_distance = np.linalg.norm(
        X - keyword_centroid,
        axis=(1, 2)
    )

    negative_distance = np.linalg.norm(
        X - negative_centroid,
        axis=(1, 2)
    )

    predicted = (
        keyword_distance
        < negative_distance
    ).astype(np.int64)

    accuracy = (
        predicted == y
    ).mean()

    keyword_recall = (
        predicted[y == 1] == 1
    ).mean()

    negative_recall = (
        predicted[y == 0] == 0
    ).mean()

    return (
        accuracy,
        keyword_recall,
        negative_recall,
    )


# ============================================================
# AUGMENTATION CHECK
# ============================================================

def inspect_keyword_variation(
    X,
    y
):
    """
    Check whether keyword samples have meaningful variation.

    The preprocessing stage creates 3 windows per training
    keyword recording. We cannot reconstruct the original
    grouping from the feature filenames alone, so this check
    operates at the distribution level rather than claiming
    exact triplets.
    """

    keyword = X[y == 1]

    sample_means = (
        keyword.mean(axis=(1, 2))
    )

    sample_stds = (
        keyword.std(axis=(1, 2))
    )

    print()
    print(
        "KEYWORD VARIATION CHECK"
    )

    print(
        f"Samples              : "
        f"{len(keyword)}"
    )

    print(
        f"Sample-mean std      : "
        f"{sample_means.std():.6f}"
    )

    print(
        f"Sample-std std       : "
        f"{sample_stds.std():.6f}"
    )

    print(
        f"Mean range           : "
        f"{sample_means.min():.6f} "
        f"to "
        f"{sample_means.max():.6f}"
    )

    print(
        f"Std range            : "
        f"{sample_stds.min():.6f} "
        f"to "
        f"{sample_stds.max():.6f}"
    )


# ============================================================
# SPLIT CHECK
# ============================================================

def inspect_split(
    split_name
):
    x_path = (
        FEATURE_DIR
        / f"{split_name}_X.npy"
    )

    y_path = (
        FEATURE_DIR
        / f"{split_name}_y.npy"
    )

    if not x_path.exists():
        raise FileNotFoundError(
            x_path
        )

    if not y_path.exists():
        raise FileNotFoundError(
            y_path
        )

    X = np.load(
        x_path,
        mmap_mode="r"
    )

    y = np.load(
        y_path
    )

    print()
    print("=" * 70)
    print(
        f"{split_name.upper()} FEATURE CHECK"
    )
    print("=" * 70)

    print(
        f"X shape              : "
        f"{X.shape}"
    )

    print(
        f"X dtype              : "
        f"{X.dtype}"
    )

    print(
        f"y shape              : "
        f"{y.shape}"
    )

    print(
        f"Labels               : "
        f"{np.unique(y).tolist()}"
    )

    print(
        f"Keyword count        : "
        f"{np.sum(y == 1)}"
    )

    print(
        f"Negative count       : "
        f"{np.sum(y == 0)}"
    )

    # --------------------------------------------------------
    # Structural checks
    # --------------------------------------------------------

    assert X.ndim == 3
    assert X.shape[1] == 98
    assert X.shape[2] == 40

    assert len(X) == len(y)

    assert X.dtype == np.float32

    # --------------------------------------------------------
    # NaN / Inf checks
    # --------------------------------------------------------

    nan_count = int(
        np.isnan(X).sum()
    )

    inf_count = int(
        np.isinf(X).sum()
    )

    print()
    print(
        "NUMERICAL VALIDATION"
    )

    print(
        f"NaN values          : "
        f"{nan_count}"
    )

    print(
        f"Inf values          : "
        f"{inf_count}"
    )

    print(
        f"Global min          : "
        f"{X.min():.6f}"
    )

    print(
        f"Global max          : "
        f"{X.max():.6f}"
    )

    print(
        f"Global mean         : "
        f"{X.mean():.6f}"
    )

    print(
        f"Global std          : "
        f"{X.std():.6f}"
    )

    if nan_count != 0:
        raise RuntimeError(
            "NaN values detected."
        )

    if inf_count != 0:
        raise RuntimeError(
            "Inf values detected."
        )

    # --------------------------------------------------------
    # Class statistics
    # --------------------------------------------------------

    describe_class(
        X,
        y,
        1,
        "Keyword"
    )

    describe_class(
        X,
        y,
        0,
        "Negative"
    )

    # --------------------------------------------------------
    # Simple centroid separation
    # --------------------------------------------------------

    (
        centroid_accuracy,
        keyword_recall,
        negative_recall,
    ) = nearest_class_margin(
        X,
        y
    )

    print()
    print(
        "CENTROID SEPARATION CHECK"
    )

    print(
        f"Centroid accuracy    : "
        f"{centroid_accuracy * 100:.2f}%"
    )

    print(
        f"Keyword recall       : "
        f"{keyword_recall * 100:.2f}%"
    )

    print(
        f"Negative recall      : "
        f"{negative_recall * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Training augmentation variation
    # --------------------------------------------------------

    if split_name == "train":
        inspect_keyword_variation(
            X,
            y
        )

    return X, y


# ============================================================
# CROSS-SPLIT CHECK
# ============================================================

def compare_split_statistics(
    datasets
):
    print()
    print("=" * 70)
    print(
        "CROSS-SPLIT DISTRIBUTION CHECK"
    )
    print("=" * 70)

    for split_name, (X, y) in datasets.items():

        keyword = X[y == 1]
        negative = X[y == 0]

        print()
        print(
            f"{split_name.upper()}"
        )

        print(
            f"Keyword mean         : "
            f"{keyword.mean():.6f}"
        )

        print(
            f"Keyword std          : "
            f"{keyword.std():.6f}"
        )

        print(
            f"Negative mean        : "
            f"{negative.mean():.6f}"
        )

        print(
            f"Negative std         : "
            f"{negative.std():.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "FEATURE SANITY VALIDATION"
    )
    print("=" * 70)

    datasets = {}

    for split_name in SPLITS:

        X, y = inspect_split(
            split_name
        )

        datasets[
            split_name
        ] = (X, y)

    compare_split_statistics(
        datasets
    )

    print()
    print("=" * 70)
    print(
        "FINAL VALIDATION"
    )
    print("=" * 70)

    failures = []

    for split_name, (X, y) in datasets.items():

        if X.shape[1:] != (
            98,
            40
        ):
            failures.append(
                f"{split_name}: wrong feature shape"
            )

        if len(X) != len(y):
            failures.append(
                f"{split_name}: X/y length mismatch"
            )

        if not np.isfinite(X).all():
            failures.append(
                f"{split_name}: NaN/Inf detected"
            )

        if set(
            np.unique(y).tolist()
        ) != {0, 1}:
            failures.append(
                f"{split_name}: unexpected labels"
            )

    if failures:

        print(
            "FAIL"
        )

        for failure in failures:
            print(
                f"  - {failure}"
            )

        raise SystemExit(1)

    print(
        "PASS: feature corpus is "
        "numerically and structurally valid."
    )


if __name__ == "__main__":
    main()