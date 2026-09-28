from pathlib import Path
from collections import Counter
import contextlib
import wave

# Project root:
# D:\Projects\SIH26172
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset"

DATASETS = {
    "Picovoice positives": DATASET_ROOT / "raw" / "keyword" / "pico",
    "OVOS positives": DATASET_ROOT / "raw" / "keyword" / "ovos",
    "Common Voice negatives": DATASET_ROOT / "raw" / "negative" / "common_voice",
    "OVOS negatives": DATASET_ROOT / "raw" / "negative" / "ovos",
}

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
}


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.2f} s"

    minutes = seconds / 60

    if minutes < 60:
        return f"{minutes:.2f} min"

    return f"{minutes / 60:.2f} h"


def inspect_wav(path):
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()

            duration = frames / rate if rate else 0

            return {
                "format": "wav",
                "sample_rate": rate,
                "channels": channels,
                "sample_width": f"{sample_width * 8}-bit",
                "duration": duration,
                "error": None,
            }

    except Exception as exc:
        return {
            "format": "wav",
            "sample_rate": None,
            "channels": None,
            "sample_width": None,
            "duration": 0,
            "error": str(exc),
        }


def inspect_compressed(path):
    """
    Uses soundfile for formats such as MP3.
    """

    try:
        import soundfile as sf

        info = sf.info(str(path))

        return {
            "format": path.suffix.lower().lstrip("."),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "sample_width": info.subtype,
            "duration": info.duration,
            "error": None,
        }

    except Exception as exc:
        return {
            "format": path.suffix.lower().lstrip("."),
            "sample_rate": None,
            "channels": None,
            "sample_width": None,
            "duration": 0,
            "error": str(exc),
        }


def inspect_file(path):
    if path.suffix.lower() == ".wav":
        return inspect_wav(path)

    return inspect_compressed(path)


def inspect_dataset(name, directory):

    print()
    print("=" * 75)
    print(name)
    print("=" * 75)

    print(f"Directory: {directory}")

    if not directory.exists():
        print("STATUS: DIRECTORY NOT FOUND")
        return

    files = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
    ]

    print(f"Audio files: {len(files)}")

    if not files:
        return

    formats = Counter()
    sample_rates = Counter()
    channels = Counter()
    sample_widths = Counter()

    durations = []

    errors = []

    for index, path in enumerate(files, start=1):

        info = inspect_file(path)

        formats[info["format"]] += 1

        if info["sample_rate"] is not None:
            sample_rates[info["sample_rate"]] += 1

        if info["channels"] is not None:
            channels[info["channels"]] += 1

        if info["sample_width"] is not None:
            sample_widths[str(info["sample_width"])] += 1

        if info["error"] is not None:
            errors.append((path, info["error"]))
        else:
            durations.append(info["duration"])

        if index % 500 == 0:
            print(f"  inspected {index}/{len(files)}...")

    print()
    print(f"Formats:          {dict(formats)}")
    print(f"Sample rates:     {dict(sample_rates)}")
    print(f"Channels:         {dict(channels)}")
    print(f"Sample widths:    {dict(sample_widths)}")

    if durations:

        total_duration = sum(durations)

        print(f"Total duration:   {format_duration(total_duration)}")
        print(f"Shortest:         {min(durations):.3f} s")
        print(f"Longest:          {max(durations):.3f} s")
        print(f"Average:          {sum(durations) / len(durations):.3f} s")

    print(f"Unreadable files: {len(errors)}")

    if errors:

        print()
        print("First 10 errors:")

        for path, error in errors[:10]:

            print(f"  {path}")
            print(f"    {error}")


def main():

    print("=" * 75)
    print("SIH 26172 — DATASET INSPECTION")
    print("=" * 75)

    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Dataset root:  {DATASET_ROOT}")

    print()

    # soundfile is needed for MP3 inspection.
    try:
        import soundfile  # noqa: F401
    except ImportError:

        print("ERROR: Python package 'soundfile' is not installed.")
        print()
        print("Install it with:")
        print("  py -3.14 -m pip install soundfile")
        return

    for name, directory in DATASETS.items():
        inspect_dataset(name, directory)

    print()
    print("=" * 75)
    print("Inspection complete.")
    print("No dataset files were modified.")
    print("=" * 75)


if __name__ == "__main__":
    main()