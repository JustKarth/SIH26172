from pathlib import Path
import csv
import numpy as np
import soundfile as sf

FILES = [
    "dataset/splits/train.csv",
    "dataset/splits/val.csv",
    "dataset/splits/test.csv",
]

FRAME_MS = 25
HOP_MS = 10


def activity(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    frame = int(sr * FRAME_MS / 1000)
    hop = int(sr * HOP_MS / 1000)

    if len(audio) < frame:
        return 0.0, 0.0, len(audio) / sr

    energies = []

    for start in range(0, len(audio) - frame + 1, hop):
        x = audio[start:start + frame]
        rms = np.sqrt(np.mean(x * x) + 1e-12)
        energies.append(rms)

    energies = np.asarray(energies)

    # Relative threshold.
    threshold = max(
        np.percentile(energies, 20) * 2.0,
        0.01,
    )

    active = np.where(energies >= threshold)[0]

    if len(active) == 0:
        return 0.0, 0.0, len(audio) / sr

    first = active[0] * hop / sr
    last = (active[-1] * hop + frame) / sr

    return first, min(last, len(audio) / sr), len(audio) / sr


def main():
    rows = []

    for csv_path in FILES:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))

    rows = [r for r in rows if r["label"] == "keyword"]

    print("=" * 80)
    print("KEYWORD ACTIVITY INSPECTION")
    print("=" * 80)

    results = []

    for r in rows:
        start, end, duration = activity(r["path"])

        results.append({
            "path": r["path"],
            "dataset": r["dataset"],
            "duration": duration,
            "active_start": start,
            "active_end": end,
            "active_duration": end - start,
            "leading_silence": start,
            "trailing_silence": duration - end,
        })

    for dataset in ["picovoice", "ovos"]:
        subset = [x for x in results if x["dataset"] == dataset]

        print()
        print(f"--- {dataset.upper()} ---")
        print(f"Files: {len(subset)}")

        for field in [
            "duration",
            "active_duration",
            "leading_silence",
            "trailing_silence",
        ]:
            values = np.array([x[field] for x in subset])

            print(
                f"{field:18s}: "
                f"min={values.min():.3f} "
                f"median={np.median(values):.3f} "
                f"mean={values.mean():.3f} "
                f"max={values.max():.3f}"
            )

    # Show representative files.
    print()
    print("--- REPRESENTATIVE RECORDINGS ---")

    for dataset in ["picovoice", "ovos"]:
        subset = [x for x in results if x["dataset"] == dataset]

        subset.sort(key=lambda x: x["active_duration"])

        print()
        print(dataset.upper())

        for x in (
            subset[:3]
            + subset[len(subset) // 2:len(subset) // 2 + 3]
            + subset[-3:]
        ):
            print(
                f"{Path(x['path']).name:55s} "
                f"duration={x['duration']:.2f}s "
                f"active={x['active_start']:.2f}-{x['active_end']:.2f}s "
                f"lead={x['leading_silence']:.2f}s "
                f"trail={x['trailing_silence']:.2f}s"
            )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()