from pathlib import Path
import csv
import shutil

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000       # 1 second
FRAME_SIZE = 400             # 25 ms
FRAME_HOP = 160              # 10 ms

# Number of positive training windows generated per recording.
TRAIN_POSITIVE_WINDOWS = 3

SEED = 26172

MANIFESTS = {
    "train": Path("dataset/splits/train.csv"),
    "val": Path("dataset/splits/val.csv"),
    "test": Path("dataset/splits/test.csv"),
}

OUTPUT_ROOT = Path("dataset/processed")

# Keep the original raw files completely untouched.


# ============================================================
# AUDIO LOADING / NORMALIZATION
# ============================================================

def load_audio(path: str):
    """
    Load arbitrary source audio and convert to mono float32.
    """
    audio, sr = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    # Stereo -> mono
    audio = audio.mean(axis=1)

    # Resample if necessary.
    if sr != SAMPLE_RATE:
        gcd = np.gcd(sr, SAMPLE_RATE)

        up = SAMPLE_RATE // gcd
        down = sr // gcd

        audio = resample_poly(audio, up, down).astype(np.float32)

    return audio


def float_to_pcm16(audio):
    """
    Convert float32 [-1, 1] audio to PCM16.
    """
    audio = np.asarray(audio, dtype=np.float32)

    audio = np.clip(audio, -1.0, 1.0)

    return np.round(audio * 32767.0).astype(np.int16)


# ============================================================
# POSITIVE KEYWORD LOCALIZATION
# ============================================================

def locate_keyword(audio):
    """
    Estimate the center of the keyword using relative short-time
    energy.

    This deliberately uses relative energy rather than an absolute
    amplitude threshold because OVOS contains very quiet recordings.
    """

    if len(audio) <= FRAME_SIZE:
        return len(audio) // 2

    rms = []

    for start in range(0, len(audio) - FRAME_SIZE + 1, FRAME_HOP):
        frame = audio[start:start + FRAME_SIZE]
        energy = np.sqrt(np.mean(frame * frame) + 1e-12)
        rms.append(energy)

    rms = np.asarray(rms)

    # Robust baseline.
    baseline = np.percentile(rms, 20)

    peak = np.max(rms)

    # Relative threshold.
    #
    # This adapts to recordings with wildly different microphone
    # levels, including the very quiet OVOS tng-* recordings.
    threshold = max(
        baseline * 1.5,
        peak * 0.20,
    )

    active = rms >= threshold

    # If the threshold produces no useful region, fall back to the
    # strongest frame.
    if not np.any(active):
        best = int(np.argmax(rms))
        return best * FRAME_HOP + FRAME_SIZE // 2

    # Find contiguous active regions.
    indices = np.where(active)[0]

    regions = []

    start = indices[0]
    previous = indices[0]

    for index in indices[1:]:
        if index != previous + 1:
            regions.append((start, previous))
            start = index

        previous = index

    regions.append((start, previous))

    # Choose the region containing the greatest total energy.
    best_region = None
    best_energy = -1.0

    for start_frame, end_frame in regions:
        energy = float(np.sum(rms[start_frame:end_frame + 1]))

        if energy > best_energy:
            best_energy = energy
            best_region = (start_frame, end_frame)

    start_frame, end_frame = best_region

    start_sample = start_frame * FRAME_HOP
    end_sample = end_frame * FRAME_HOP + FRAME_SIZE

    # Energy-weighted center gives us a more stable estimate than
    # simply taking the midpoint of the recording.
    region_rms = rms[start_frame:end_frame + 1]
    frame_centers = (
        np.arange(start_frame, end_frame + 1) * FRAME_HOP
        + FRAME_SIZE / 2
    )

    weights = np.maximum(region_rms - threshold, 0.0)

    if np.sum(weights) > 0:
        center = np.sum(frame_centers * weights) / np.sum(weights)
    else:
        center = (start_sample + end_sample) / 2

    return int(center)


# ============================================================
# FIXED WINDOW EXTRACTION
# ============================================================

def extract_centered_window(audio, center):
    """
    Extract exactly one second centered around `center`.

    Zero-pad when the requested window extends beyond the recording.
    """

    half = WINDOW_SAMPLES // 2

    start = center - half
    end = start + WINDOW_SAMPLES

    output = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    src_start = max(start, 0)
    src_end = min(end, len(audio))

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    if src_end > src_start:
        output[dst_start:dst_end] = audio[src_start:src_end]

    return output


def extract_random_shifted_window(audio, center, rng):
    """
    Extract a positive window with a small random temporal shift.

    The shift is deliberately limited so that the keyword remains
    comfortably inside the one-second context.
    """

    # ±150 ms temporal variation.
    max_shift = int(0.150 * SAMPLE_RATE)

    shift = int(rng.integers(-max_shift, max_shift + 1))

    return extract_centered_window(
        audio,
        center + shift,
    )


# ============================================================
# NEGATIVE WINDOW EXTRACTION
# ============================================================

def extract_negative_window(audio, rng):
    """
    Extract one deterministic random 1-second region from a negative
    recording.

    For short recordings, zero-pad.
    """

    if len(audio) <= WINDOW_SAMPLES:
        return extract_centered_window(
            audio,
            len(audio) // 2,
        )

    max_start = len(audio) - WINDOW_SAMPLES

    start = int(rng.integers(0, max_start + 1))

    return audio[start:start + WINDOW_SAMPLES].copy()


# ============================================================
# WAV WRITING
# ============================================================

def save_pcm16(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)

    pcm = float_to_pcm16(audio)

    sf.write(
        path,
        pcm,
        SAMPLE_RATE,
        subtype="PCM_16",
    )


# ============================================================
# MANIFEST
# ============================================================

def load_manifest(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_split(split, rows, rng):
    output_dir = OUTPUT_ROOT / split

    keyword_dir = output_dir / "keyword"
    negative_dir = output_dir / "negative"

    keyword_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    keyword_count = 0
    negative_count = 0

    for row_index, row in enumerate(rows):

        source = Path(row["path"])
        label = row["label"]

        audio = load_audio(str(source))

        if label == "keyword":

            center = locate_keyword(audio)

            if split == "train":
                windows = [
                    extract_centered_window(audio, center)
                ]

                for _ in range(TRAIN_POSITIVE_WINDOWS - 1):
                    windows.append(
                        extract_random_shifted_window(
                            audio,
                            center,
                            rng,
                        )
                    )
            else:
                # Validation and test get exactly one deterministic
                # window per original recording.
                windows = [
                    extract_centered_window(audio, center)
                ]

            for window_index, window in enumerate(windows):

                output = (
                    keyword_dir
                    / f"{source.stem}_w{window_index}.wav"
                )

                save_pcm16(output, window)

                keyword_count += 1

        elif label == "negative":

            # One baseline negative window per recording.
            window = extract_negative_window(audio, rng)

            output = (
                negative_dir
                / f"{source.stem}_w0.wav"
            )

            save_pcm16(output, window)

            negative_count += 1

        else:
            raise ValueError(
                f"Unknown label: {label}"
            )

    return keyword_count, negative_count


def main():

    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("AUDIO PREPROCESSING")
    print("=" * 70)

    # Clean only the generated processed directory.
    if OUTPUT_ROOT.exists():
        print(f"Removing previous processed data: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True)

    total_keyword = 0
    total_negative = 0

    for split, manifest_path in MANIFESTS.items():

        print()
        print(f"Processing {split}...")
        print(f"Manifest: {manifest_path}")

        rows = load_manifest(manifest_path)

        keyword, negative = process_split(
            split,
            rows,
            rng,
        )

        total_keyword += keyword
        total_negative += negative

        print(f"  keyword windows : {keyword}")
        print(f"  negative windows: {negative}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Total keyword windows : {total_keyword}")
    print(f"Total negative windows: {total_negative}")

    print()
    print("Output:")
    print(f"  {OUTPUT_ROOT}")

    print()
    print("Every generated WAV:")
    print("  sample rate : 16000 Hz")
    print("  channels    : 1")
    print("  format      : PCM16")
    print("  duration    : 1.000 s")

    print()
    print("Raw dataset was not modified.")


if __name__ == "__main__":
    main()