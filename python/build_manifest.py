from pathlib import Path
import csv
import hashlib
import random

ROOT = Path("dataset")
RAW = ROOT / "raw"
SPLITS = ROOT / "splits"

SEED = 26172

SPLITS.mkdir(parents=True, exist_ok=True)


def stable_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def write_csv(path: Path, rows):
    fields = [
        "path",
        "label",
        "dataset",
        "speaker_id",
        "split",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_common_voice():
    tsv = RAW / "negative" / "common_voice" / "ss-corpus-en.tsv"

    rows = []

    with tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for r in reader:
            split = r["split"].strip()

            # Only use the clean, explicitly assigned baseline splits.
            if split not in {"train", "dev", "test"}:
                continue

            audio = RAW / "negative" / "common_voice" / r["audio_file"]

            if not audio.exists():
                continue

            rows.append({
                "path": str(audio),
                "label": "negative",
                "dataset": "common_voice",
                "speaker_id": r["client_id"].strip(),
                "split": {
                    "train": "train",
                    "dev": "val",
                    "test": "test",
                }[split],
            })

    return rows


def load_picovoice():
    directory = RAW / "keyword" / "pico"

    rows = []

    for path in sorted(directory.glob("*.wav")):
        rows.append({
            "path": str(path),
            "label": "keyword",
            "dataset": "picovoice",
            "speaker_id": "",
            "split": "",
        })

    return rows


def load_ovos_keyword():
    directory = RAW / "keyword" / "ovos"

    rows = []

    for path in sorted(directory.glob("*.wav")):
        rows.append({
            "path": str(path),
            "label": "keyword",
            "dataset": "ovos",
            "speaker_id": "",
            "split": "",
        })

    return rows


def load_ovos_negative():
    directory = RAW / "negative" / "ovos"

    rows = []

    for path in sorted(directory.glob("*.wav")):
        rows.append({
            "path": str(path),
            "label": "negative",
            "dataset": "ovos",
            "speaker_id": "",
            "split": "",
        })

    return rows


def main():
    random.seed(SEED)

    common_voice = load_common_voice()
    pico = load_picovoice()
    ovos_keyword = load_ovos_keyword()
    ovos_negative = load_ovos_negative()

    print("=" * 70)
    print("DATASET MANIFEST BUILDER")
    print("=" * 70)

    print(f"Common Voice : {len(common_voice)}")
    print(f"Picovoice    : {len(pico)}")
    print(f"OVOS keyword : {len(ovos_keyword)}")
    print(f"OVOS negative: {len(ovos_negative)}")

    # ---------------------------------------------------------------
    # Common Voice already has a speaker-disjoint train/dev/test split.
    # Keep it exactly as provided.
    # ---------------------------------------------------------------

    train = [
        r for r in common_voice
        if r["split"] == "train"
    ]

    val = [
        r for r in common_voice
        if r["split"] == "val"
    ]

    test = [
        r for r in common_voice
        if r["split"] == "test"
    ]

    # ---------------------------------------------------------------
    # Positive datasets do not yet have verified speaker metadata.
    #
    # Therefore we do NOT pretend they are speaker-disjoint.
    #
    # Randomly assign individual recordings for the first baseline.
    # We will improve this once provenance metadata is established.
    # ---------------------------------------------------------------

    positives = pico + ovos_keyword
    random.shuffle(positives)

    n = len(positives)

    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    positive_train = positives[:n_train]
    positive_val = positives[n_train:n_train + n_val]
    positive_test = positives[n_train + n_val:]

    for row in positive_train:
        row["split"] = "train"

    for row in positive_val:
        row["split"] = "val"

    for row in positive_test:
        row["split"] = "test"

    # ---------------------------------------------------------------
    # OVOS negative recordings.
    #
    # Randomly distribute them. They have no verified speaker metadata
    # in this manifest yet.
    # ---------------------------------------------------------------

    ovos_negative = ovos_negative.copy()
    random.shuffle(ovos_negative)

    n = len(ovos_negative)

    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    negative_train = ovos_negative[:n_train]
    negative_val = ovos_negative[n_train:n_train + n_val]
    negative_test = ovos_negative[n_train + n_val:]

    for row in negative_train:
        row["split"] = "train"

    for row in negative_val:
        row["split"] = "val"

    for row in negative_test:
        row["split"] = "test"

    # ---------------------------------------------------------------
    # Combine
    # ---------------------------------------------------------------

    train.extend(positive_train)
    train.extend(negative_train)

    val.extend(positive_val)
    val.extend(negative_val)

    test.extend(positive_test)
    test.extend(negative_test)

    # Shuffle each split deterministically.
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    write_csv(SPLITS / "train.csv", train)
    write_csv(SPLITS / "val.csv", val)
    write_csv(SPLITS / "test.csv", test)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print()
    print("--- FINAL MANIFEST ---")

    for name, rows in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        keyword = sum(r["label"] == "keyword" for r in rows)
        negative = sum(r["label"] == "negative" for r in rows)

        print(
            f"{name:5s}: "
            f"{len(rows):5d} total | "
            f"{keyword:5d} keyword | "
            f"{negative:5d} negative"
        )

    print()
    print("Created:")
    print(f"  {SPLITS / 'train.csv'}")
    print(f"  {SPLITS / 'val.csv'}")
    print(f"  {SPLITS / 'test.csv'}")

    print()
    print("NOTE:")
    print("Common Voice uses its verified speaker-disjoint splits.")
    print("Positive/OVOS recordings currently use deterministic random")
    print("recording-level splits because speaker metadata is not yet verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()