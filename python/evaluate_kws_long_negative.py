from __future__ import annotations

import argparse
import csv
import heapq
import math
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "kws_baseline_best.keras"

COMMON_VOICE_DIR = (
    ROOT
    / "dataset"
    / "raw"
    / "negative"
    / "common_voice"
)

METADATA_PATH = COMMON_VOICE_DIR / "ss-corpus-en.tsv"

DEBUG_DIR = (
    ROOT
    / "dataset"
    / "debug"
    / "false_positives"
)

TOP_N = 20


# ============================================================
# AUDIO / FEATURE CONFIGURATION
# Must remain identical to training.
# ============================================================

SAMPLE_RATE = 16000

WINDOW_SAMPLES = 16000
HOP_SAMPLES = 160

FRAME_SIZE = 400
FRAME_HOP = 160

FFT_SIZE = 512
NUM_MEL = 40

LOW_FREQ = 20.0
HIGH_FREQ = 8000.0

PRE_EMPHASIS = 0.97

EPSILON = 1e-10


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS = [
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]


# ============================================================
# FEATURE EXTRACTOR
# Exact Python reference pipeline used for training.
# ============================================================

def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank() -> np.ndarray:
    low_mel = hz_to_mel(LOW_FREQ)
    high_mel = hz_to_mel(HIGH_FREQ)

    mel_points = np.linspace(
        low_mel,
        high_mel,
        NUM_MEL + 2,
    )

    hz_points = mel_to_hz(mel_points)

    bins = np.floor(
        (FFT_SIZE + 1) * hz_points / SAMPLE_RATE
    ).astype(int)

    filters = np.zeros(
        (NUM_MEL, FFT_SIZE // 2 + 1),
        dtype=np.float32,
    )

    for m in range(1, NUM_MEL + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center > left:
            for k in range(left, center):
                if 0 <= k < filters.shape[1]:
                    filters[m - 1, k] = (
                        (k - left) / float(center - left)
                    )

        if right > center:
            for k in range(center, right):
                if 0 <= k < filters.shape[1]:
                    filters[m - 1, k] = (
                        (right - k) / float(right - center)
                    )

    return filters


MEL_FILTERS = build_mel_filterbank()

HAMMING_WINDOW = np.hamming(FRAME_SIZE).astype(np.float32)


def extract_window_features(
    pcm16: np.ndarray,
) -> np.ndarray:
    """
    Input:
        exactly 1 second of int16 PCM at 16 kHz.

    Output:
        [98, 40] float32 feature matrix.
    """

    if len(pcm16) != WINDOW_SAMPLES:
        raise ValueError(
            f"Expected {WINDOW_SAMPLES} samples, "
            f"got {len(pcm16)}"
        )

    audio = (
        pcm16.astype(np.float32)
        / 32768.0
    )

    # Pre-emphasis.
    emphasized = np.empty_like(audio)

    emphasized[0] = audio[0]

    emphasized[1:] = (
        audio[1:]
        - PRE_EMPHASIS * audio[:-1]
    )

    num_frames = (
        1
        + (WINDOW_SAMPLES - FRAME_SIZE)
        // FRAME_HOP
    )

    features = np.empty(
        (num_frames, NUM_MEL),
        dtype=np.float32,
    )

    for frame_index in range(num_frames):

        start = frame_index * FRAME_HOP
        end = start + FRAME_SIZE

        frame = emphasized[start:end]

        frame = frame * HAMMING_WINDOW

        fft_input = np.zeros(
            FFT_SIZE,
            dtype=np.float32,
        )

        fft_input[:FRAME_SIZE] = frame

        spectrum = np.fft.rfft(
            fft_input,
            n=FFT_SIZE,
        )

        power = (
            np.abs(spectrum) ** 2
        ).astype(np.float32)

        mel_energy = (
            MEL_FILTERS @ power
        ).astype(np.float32)

        mel_energy = np.maximum(
            mel_energy,
            EPSILON,
        )

        features[frame_index] = np.log(
            mel_energy
        )

    return features


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path: Path) -> np.ndarray:
    """
    Load any supported audio file and convert to:
        mono, 16 kHz, int16 PCM.
    """

    audio, sr = librosa.load(
        str(path),
        sr=SAMPLE_RATE,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    pcm16 = np.round(
        audio * 32767.0
    ).astype(np.int16)

    return pcm16


# ============================================================
# 1-SECOND WINDOW GENERATION
# ============================================================

def make_windows(
    pcm16: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 1-second windows every 10 ms.

    Returns:
        windows:
            [N, 16000] int16

        starts:
            [N] start sample indices
    """

    if len(pcm16) < WINDOW_SAMPLES:
        padded = np.zeros(
            WINDOW_SAMPLES,
            dtype=np.int16,
        )

        padded[:len(pcm16)] = pcm16

        return (
            padded[np.newaxis, :],
            np.array([0], dtype=np.int64),
        )

    starts = np.arange(
        0,
        len(pcm16) - WINDOW_SAMPLES + 1,
        HOP_SAMPLES,
        dtype=np.int64,
    )

    windows = np.stack(
        [
            pcm16[
                start:
                start + WINDOW_SAMPLES
            ]
            for start in starts
        ],
        axis=0,
    )

    return windows, starts


# ============================================================
# MODEL INPUT
# ============================================================

def prepare_model_features(
    windows: np.ndarray,
    train_mean: float,
    train_std: float,
) -> np.ndarray:
    """
    Convert [N, 16000] audio windows into
    normalized [N, 98, 40, 1] features.
    """

    feature_list = []

    for window in windows:
        feature_list.append(
            extract_window_features(window)
        )

    features = np.stack(
        feature_list,
        axis=0,
    )

    features = (
        features - train_mean
    ) / train_std

    features = features[..., np.newaxis]

    return features.astype(
        np.float32
    )


# ============================================================
# EVENT COUNTING
# ============================================================

def count_events(
    probabilities: np.ndarray,
    threshold: float,
) -> int:
    """
    Count contiguous above-threshold regions.

    Each contiguous region is one candidate
    activation event.

    This operates on ONE source file only.
    """

    active = probabilities >= threshold

    if len(active) == 0:
        return 0

    starts = (
        active
        & ~np.concatenate(
            [
                np.array([False]),
                active[:-1],
            ]
        )
    )

    return int(
        np.sum(starts)
    )


# ============================================================
# TOP-N TRACKING
# ============================================================

# Heap entries:
#
# (
#     probability,
#     unique_counter,
#     path_string,
#     start_sample
# )
#
# Python's heap is a min-heap, so the smallest
# probability stays at the root and gets removed
# when a better candidate arrives.

def update_top_windows(
    heap: List[Tuple[float, int, str, int]],
    probabilities: np.ndarray,
    starts: np.ndarray,
    path: Path,
    counter_start: int,
) -> int:

    counter = counter_start

    for probability, start in zip(
        probabilities,
        starts,
    ):

        probability = float(probability)
        start = int(start)

        entry = (
            probability,
            counter,
            str(path),
            start,
        )

        counter += 1

        if len(heap) < TOP_N:
            heapq.heappush(
                heap,
                entry,
            )

        elif probability > heap[0][0]:
            heapq.heapreplace(
                heap,
                entry,
            )

    return counter


# ============================================================
# DEBUG CLIP WRITER
# ============================================================

def save_debug_clip(
    pcm16: np.ndarray,
    start_sample: int,
    output_path: Path,
) -> None:

    start_sample = max(
        0,
        start_sample,
    )

    end_sample = min(
        len(pcm16),
        start_sample + WINDOW_SAMPLES,
    )

    clip = pcm16[
        start_sample:end_sample
    ]

    if len(clip) < WINDOW_SAMPLES:
        padded = np.zeros(
            WINDOW_SAMPLES,
            dtype=np.int16,
        )

        padded[:len(clip)] = clip
        clip = padded

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(output_path),
        clip,
        SAMPLE_RATE,
        subtype="PCM_16",
    )


# ============================================================
# METADATA
# ============================================================

def iter_audio_files() -> List[Path]:
    """
    Read Common Voice metadata and return only
    metadata-listed audio files.
    """

    paths = []

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            filename = row.get(
                "audio_file",
                "",
            ).strip()

            if not filename:
                continue

            path = (
                COMMON_VOICE_DIR
                / filename
            )

            if path.exists():
                paths.append(path)

    return paths


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help=(
            "Maximum amount of audio to scan. "
            "If omitted, scan all metadata-listed audio."
        ),
    )

    parser.add_argument(
        "--batch-files",
        type=int,
        default=1,
        help=(
            "Number of files processed before "
            "printing progress."
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LONG-FORM NEGATIVE KWS EVALUATION")
    print("=" * 70)

    print()
    print(f"Model    : {MODEL_PATH}")
    print(f"Audio    : {COMMON_VOICE_DIR}")
    print(f"Metadata : {METADATA_PATH}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # Training normalization
    # --------------------------------------------------------

    train_X = np.load(
        ROOT
        / "dataset"
        / "processed"
        / "features"
        / "train_X.npy"
    )

    train_mean = float(
        train_X.mean()
    )

    train_std = float(
        train_X.std()
    )

    del train_X

    print()
    print(
        f"Training mean : {train_mean:.6f}"
    )

    print(
        f"Training std  : {train_std:.6f}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    paths = iter_audio_files()

    print()
    print(
        f"Metadata-listed audio files : {len(paths)}"
    )

    # --------------------------------------------------------
    # Scan state
    # --------------------------------------------------------

    total_files = 0
    skipped_files = 0

    total_samples = 0
    total_windows = 0

    total_duration_seconds = 0.0

    # Events are accumulated per threshold.
    total_events: Dict[float, int] = {
        threshold: 0
        for threshold in THRESHOLDS
    }

    total_positive_windows: Dict[float, int] = {
        threshold: 0
        for threshold in THRESHOLDS
    }

    # Top-N heap.
    top_heap: List[
        Tuple[float, int, str, int]
    ] = []

    unique_counter = 0

    # --------------------------------------------------------
    # Maximum scan duration
    # --------------------------------------------------------

    if args.max_hours is not None:

        max_seconds = (
            args.max_hours * 3600.0
        )

    else:

        max_seconds = None

    scanned_seconds = 0.0

    # --------------------------------------------------------
    # Scan
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SCANNING NEGATIVE AUDIO")
    print("=" * 70)

    for path in paths:

        if (
            max_seconds is not None
            and scanned_seconds >= max_seconds
        ):
            break

        try:

            pcm16 = load_audio(path)

        except Exception as exc:

            skipped_files += 1

            print(
                f"SKIP {path.name}: {exc}"
            )

            continue

        duration = (
            len(pcm16)
            / float(SAMPLE_RATE)
        )

        # Do not exceed requested max duration.
        if (
            max_seconds is not None
            and scanned_seconds + duration
            > max_seconds
        ):

            remaining = (
                max_seconds
                - scanned_seconds
            )

            max_samples = int(
                remaining
                * SAMPLE_RATE
            )

            if max_samples <= 0:
                break

            pcm16 = pcm16[
                :max_samples
            ]

            duration = (
                len(pcm16)
                / float(SAMPLE_RATE)
            )

        if len(pcm16) == 0:
            continue

        # ----------------------------------------------------
        # Windows
        # ----------------------------------------------------

        windows, starts = make_windows(
            pcm16
        )

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        features = prepare_model_features(
            windows,
            train_mean,
            train_std,
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = model.predict(
            features,
            verbose=0,
        )

        probabilities = (
            predictions[:, 1]
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Positive window counts
        # ----------------------------------------------------

        for threshold in THRESHOLDS:

            positive_count = int(
                np.sum(
                    probabilities
                    >= threshold
                )
            )

            total_positive_windows[
                threshold
            ] += positive_count

        # ----------------------------------------------------
        # Event counting
        #
        # IMPORTANT:
        # Count events independently inside
        # each source file.
        # ----------------------------------------------------

        for threshold in THRESHOLDS:

            events = count_events(
                probabilities,
                threshold,
            )

            total_events[
                threshold
            ] += events

        # ----------------------------------------------------
        # Top-N windows
        # ----------------------------------------------------

        unique_counter = (
            update_top_windows(
                top_heap,
                probabilities,
                starts,
                path,
                unique_counter,
            )
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_files += 1

        total_samples += len(pcm16)

        total_windows += len(
            probabilities
        )

        total_duration_seconds += (
            duration
        )

        scanned_seconds += duration

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            total_files % args.batch_files
            == 0
        ):

            print(
                f"files={total_files:5d} "
                f"audio="
                f"{total_duration_seconds / 3600.0:8.3f} h "
                f"windows="
                f"{total_windows:9d}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Processed files       : "
        f"{total_files}"
    )

    print(
        f"Skipped files         : "
        f"{skipped_files}"
    )

    print(
        f"Audio duration        : "
        f"{total_duration_seconds:.2f} s "
        f"("
        f"{total_duration_seconds / 3600.0:.4f}"
        f" h)"
    )

    print(
        f"Total samples         : "
        f"{total_samples}"
    )

    print(
        f"Total 1-sec windows   : "
        f"{total_windows}"
    )

    print(
        f"Scan hop              : "
        f"{HOP_SAMPLES / SAMPLE_RATE * 1000.0:.1f} ms"
    )

    # --------------------------------------------------------
    # Probability distribution
    #
    # We don't retain every probability, so calculate
    # distribution statistics from the top-N only is NOT
    # acceptable.
    #
    # Therefore this version intentionally reports the
    # operational threshold/event statistics and top-N.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RAW THRESHOLD RESULTS")
    print("=" * 70)

    print()
    print(
        "Threshold | Positive windows | "
        "Events | Events/hour"
    )

    print("-" * 70)

    hours = (
        total_duration_seconds / 3600.0
    )

    for threshold in THRESHOLDS:

        positive = (
            total_positive_windows[
                threshold
            ]
        )

        events = (
            total_events[
                threshold
            ]
        )

        if hours > 0:
            events_per_hour = (
                events / hours
            )
        else:
            events_per_hour = 0.0

        print(
            f"{threshold:9.2f} | "
            f"{positive:16d} | "
            f"{events:6d} | "
            f"{events_per_hour:11.3f}"
        )

    # --------------------------------------------------------
    # Sort top-N
    # --------------------------------------------------------

    top_records = sorted(
        top_heap,
        key=lambda item: item[0],
        reverse=True,
    )

    # --------------------------------------------------------
    # Save debug clips
    #
    # Reload each source file only for the top-N clips.
    # --------------------------------------------------------

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Clear old debug WAVs.
    for old_file in DEBUG_DIR.glob(
        "*.wav"
    ):
        try:
            old_file.unlink()
        except OSError:
            pass

    print()
    print("=" * 70)
    print("TOP NEGATIVE PROBABILITY WINDOWS")
    print("=" * 70)

    print()
    print(
        "Rank | Probability | Time (s) | Source"
    )

    print("-" * 70)

    audio_cache: Dict[
        str,
        np.ndarray,
    ] = {}

    for rank, (
        probability,
        _counter,
        path_string,
        start_sample,
    ) in enumerate(
        top_records,
        start=1,
    ):

        path = Path(
            path_string
        )

        time_seconds = (
            start_sample
            / float(SAMPLE_RATE)
        )

        safe_probability = (
            f"{probability:.6f}"
        )

        source_stem = (
            path.stem
        )

        clip_name = (
            f"{rank:02d}_"
            f"p{safe_probability}_"
            f"{source_stem}_"
            f"{time_seconds:.3f}s.wav"
        )

        clip_path = (
            DEBUG_DIR
            / clip_name
        )

        cache_key = str(path)

        if cache_key not in audio_cache:

            try:
                audio_cache[
                    cache_key
                ] = load_audio(path)

            except Exception as exc:

                print(
                    f"Could not reload "
                    f"{path}: {exc}"
                )

                continue

        save_debug_clip(
            audio_cache[cache_key],
            start_sample,
            clip_path,
        )

        print(
            f"{rank:4d} | "
            f"{probability:11.8f} | "
            f"{time_seconds:8.3f} | "
            f"{path.name}"
        )

        print(
            f"     clip: {clip_path}"
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()