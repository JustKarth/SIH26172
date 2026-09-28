from pathlib import Path
import numpy as np
import soundfile as sf


ROOT = Path("dataset/processed")

EXPECTED_SR = 16000
EXPECTED_SAMPLES = 16000


def inspect_directory(path):
    files = sorted(path.glob("*.wav"))

    durations = []
    rms_values = []
    peaks = []
    sample_rates = set()
    channels = set()
    sample_counts = set()

    bad = []

    for f in files:
        try:
            info = sf.info(f)

            sample_rates.add(info.samplerate)
            channels.add(info.channels)
            sample_counts.add(info.frames)

            if (
                info.samplerate != EXPECTED_SR
                or info.channels != 1
                or info.frames != EXPECTED_SAMPLES
            ):
                bad.append(f)

            audio, sr = sf.read(
                f,
                dtype="float32",
                always_2d=True,
            )

            audio = audio[:, 0]

            durations.append(len(audio) / sr)
            rms_values.append(float(np.sqrt(np.mean(audio * audio))))
            peaks.append(float(np.max(np.abs(audio))))

        except Exception as e:
            bad.append((f, str(e)))

    return {
        "count": len(files),
        "durations": np.asarray(durations),
        "rms": np.asarray(rms_values),
        "peaks": np.asarray(peaks),
        "sample_rates": sample_rates,
        "channels": channels,
        "sample_counts": sample_counts,
        "bad": bad,
    }


def print_stats(name, stats):
    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(f"Files          : {stats['count']}")
    print(f"Sample rates   : {sorted(stats['sample_rates'])}")
    print(f"Channels       : {sorted(stats['channels'])}")
    print(f"Sample counts  : {sorted(stats['sample_counts'])}")

    if len(stats["durations"]):
        print(
            f"Duration       : "
            f"min={stats['durations'].min():.3f} "
            f"mean={stats['durations'].mean():.3f} "
            f"max={stats['durations'].max():.3f}"
        )

        print(
            f"RMS            : "
            f"min={stats['rms'].min():.6f} "
            f"median={np.median(stats['rms']):.6f} "
            f"mean={stats['rms'].mean():.6f} "
            f"max={stats['rms'].max():.6f}"
        )

        print(
            f"Peak           : "
            f"min={stats['peaks'].min():.6f} "
            f"median={np.median(stats['peaks']):.6f} "
            f"max={stats['peaks'].max():.6f}"
        )

    print(f"Invalid files  : {len(stats['bad'])}")

    if stats["bad"]:
        print("\nFirst invalid files:")

        for item in stats["bad"][:10]:
            print(f"  {item}")


def main():
    print("=" * 70)
    print("PROCESSED AUDIO VALIDATION")
    print("=" * 70)

    total_files = 0
    total_bad = 0

    for split in ["train", "val", "test"]:
        for label in ["keyword", "negative"]:

            directory = ROOT / split / label

            stats = inspect_directory(directory)

            print_stats(
                f"{split.upper()} / {label.upper()}",
                stats,
            )

            total_files += stats["count"]
            total_bad += len(stats["bad"])

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total processed WAV files : {total_files}")
    print(f"Total invalid files       : {total_bad}")

    expected = 2113 + 3969

    print(f"Expected files            : {expected}")

    if total_files == expected and total_bad == 0:
        print("\nPASS: processed corpus is structurally valid.")
    else:
        print("\nFAIL: investigate the processed corpus.")


if __name__ == "__main__":
    main()