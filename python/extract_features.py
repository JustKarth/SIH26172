from pathlib import Path
import wave

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000

FRAME_SIZE = 400
FRAME_HOP = 160
FFT_SIZE = 512

NUM_MEL = 40
LOW_FREQ = 20.0
HIGH_FREQ = 8000.0

PRE_EMPHASIS = 0.97
LOG_FLOOR = 1e-10

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
)

FEATURE_DIR = (
    PROCESSED_DIR
    / "features"
)


# ============================================================
# WAV LOADING
# ============================================================

def load_pcm16_wav(path):
    with wave.open(str(path), "rb") as wav:

        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        sample_count = wav.getnframes()

        if channels != 1:
            raise ValueError(
                f"{path}: expected mono, got {channels}"
            )

        if sample_width != 2:
            raise ValueError(
                f"{path}: expected 16-bit PCM, "
                f"got {sample_width * 8}-bit"
            )

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"{path}: expected {SAMPLE_RATE} Hz, "
                f"got {sample_rate} Hz"
            )

        if sample_count != WINDOW_SAMPLES:
            raise ValueError(
                f"{path}: expected {WINDOW_SAMPLES} samples, "
                f"got {sample_count}"
            )

        raw = wav.readframes(sample_count)

    return np.frombuffer(
        raw,
        dtype="<i2"
    )


# ============================================================
# MEL FILTER BANK
# ============================================================

def hz_to_mel(hz):
    return 2595.0 * np.log10(
        1.0 + hz / 700.0
    )


def mel_to_hz(mel):
    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def build_mel_filterbank():

    num_bins = (
        FFT_SIZE // 2 + 1
    )

    low_mel = hz_to_mel(
        LOW_FREQ
    )

    high_mel = hz_to_mel(
        HIGH_FREQ
    )

    mel_points = np.linspace(
        low_mel,
        high_mel,
        NUM_MEL + 2
    )

    hz_points = mel_to_hz(
        mel_points
    )

    bins = np.floor(
        (FFT_SIZE + 1)
        * hz_points
        / SAMPLE_RATE
    ).astype(int)

    filters = np.zeros(
        (NUM_MEL, num_bins),
        dtype=np.float32
    )

    for m in range(
        1,
        NUM_MEL + 1
    ):

        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center > left:

            for k in range(
                left,
                center
            ):

                if 0 <= k < num_bins:

                    filters[
                        m - 1,
                        k
                    ] = (
                        (k - left)
                        / float(
                            center - left
                        )
                    )

        if right > center:

            for k in range(
                center,
                right
            ):

                if 0 <= k < num_bins:

                    filters[
                        m - 1,
                        k
                    ] = (
                        (right - k)
                        / float(
                            right - center
                        )
                    )

    return filters


# ============================================================
# HAMMING WINDOW
# ============================================================

def build_hamming_window():

    n = np.arange(
        FRAME_SIZE,
        dtype=np.float32
    )

    return (
        0.54
        - 0.46
        * np.cos(
            2.0
            * np.pi
            * n
            / (FRAME_SIZE - 1)
        )
    ).astype(np.float32)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    pcm16,
    hamming,
    mel_filters
):

    if len(pcm16) != WINDOW_SAMPLES:

        raise ValueError(
            f"Expected {WINDOW_SAMPLES} samples, "
            f"got {len(pcm16)}"
        )

    # Processed audio is already PCM16.
    #
    # IMPORTANT:
    # Do NOT apply the ESP32 raw-I2S >>16 conversion.
    #
    samples = (
        pcm16.astype(np.float32)
        / 32768.0
    )

    num_frames = (
        (
            WINDOW_SAMPLES
            - FRAME_SIZE
        )
        // FRAME_HOP
    ) + 1

    features = np.zeros(
        (
            num_frames,
            NUM_MEL
        ),
        dtype=np.float32
    )

    for frame_idx in range(
        num_frames
    ):

        start = (
            frame_idx
            * FRAME_HOP
        )

        frame = samples[
            start:start + FRAME_SIZE
        ].copy()

        # ----------------------------------------------------
        # Pre-emphasis
        # ----------------------------------------------------

        previous = frame.copy()

        frame[1:] = (
            frame[1:]
            - PRE_EMPHASIS
            * previous[:-1]
        )

        # ----------------------------------------------------
        # Hamming window
        # ----------------------------------------------------

        frame *= hamming

        # ----------------------------------------------------
        # Zero padding
        # ----------------------------------------------------

        fft_input = np.zeros(
            FFT_SIZE,
            dtype=np.float32
        )

        fft_input[
            :FRAME_SIZE
        ] = frame

        # ----------------------------------------------------
        # FFT
        # ----------------------------------------------------

        spectrum = np.fft.rfft(
            fft_input
        )

        # ----------------------------------------------------
        # Power spectrum
        # ----------------------------------------------------

        power = (
            spectrum.real
            * spectrum.real
            +
            spectrum.imag
            * spectrum.imag
        ).astype(np.float32)

        # ----------------------------------------------------
        # Mel filter bank
        # ----------------------------------------------------

        mel_energy = (
            mel_filters @ power
        )

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        mel_energy = np.maximum(
            mel_energy,
            LOG_FLOOR
        )

        log_mel = np.log(
            mel_energy
        ).astype(np.float32)

        features[
            frame_idx
        ] = log_mel

    return features


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    split_name,
    hamming,
    mel_filters
):

    split_dir = (
        PROCESSED_DIR
        / split_name
    )

    keyword_dir = (
        split_dir
        / "keyword"
    )

    negative_dir = (
        split_dir
        / "negative"
    )

    keyword_files = sorted(
        keyword_dir.glob("*.wav")
    )

    negative_files = sorted(
        negative_dir.glob("*.wav")
    )

    if not keyword_files:
        raise RuntimeError(
            f"No keyword WAVs found in "
            f"{keyword_dir}"
        )

    if not negative_files:
        raise RuntimeError(
            f"No negative WAVs found in "
            f"{negative_dir}"
        )

    files = []

    for path in keyword_files:
        files.append(
            (path, 1)
        )

    for path in negative_files:
        files.append(
            (path, 0)
        )

    # Keep deterministic ordering.
    files.sort(
        key=lambda item: str(item[0]).lower()
    )

    X = []
    y = []

    print()
    print("-" * 70)
    print(
        f"PROCESSING {split_name.upper()}"
    )
    print("-" * 70)

    print(
        f"Keyword files : "
        f"{len(keyword_files)}"
    )

    print(
        f"Negative files: "
        f"{len(negative_files)}"
    )

    total = len(files)

    for index, (path, label) in enumerate(
        files,
        start=1
    ):

        pcm16 = load_pcm16_wav(
            path
        )

        feature_matrix = (
            extract_features(
                pcm16,
                hamming,
                mel_filters
            )
        )

        X.append(
            feature_matrix
        )

        y.append(
            label
        )

        if (
            index == 1
            or index % 500 == 0
            or index == total
        ):

            print(
                f"Processed "
                f"{index}/{total}"
            )

    X = np.stack(
        X
    ).astype(
        np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    output_x = (
        FEATURE_DIR
        / f"{split_name}_X.npy"
    )

    output_y = (
        FEATURE_DIR
        / f"{split_name}_y.npy"
    )

    np.save(
        output_x,
        X
    )

    np.save(
        output_y,
        y
    )

    keyword_count = int(
        np.sum(y == 1)
    )

    negative_count = int(
        np.sum(y == 0)
    )

    print()
    print(
        f"{split_name.upper()} RESULTS"
    )

    print(
        f"Samples       : {len(y)}"
    )

    print(
        f"Keyword       : {keyword_count}"
    )

    print(
        f"Negative      : {negative_count}"
    )

    print(
        f"Feature shape : {X.shape}"
    )

    print(
        f"Feature dtype : {X.dtype}"
    )

    print(
        f"Feature min   : {X.min():.6f}"
    )

    print(
        f"Feature max   : {X.max():.6f}"
    )

    print(
        f"Feature mean  : {X.mean():.6f}"
    )

    print(
        f"Feature std   : {X.std():.6f}"
    )

    print(
        f"Saved X       : {output_x}"
    )

    print(
        f"Saved y       : {output_y}"
    )

    return X, y


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "LOG-MEL FEATURE EXTRACTION"
    )
    print("=" * 70)

    print()
    print(
        f"Sample rate       : "
        f"{SAMPLE_RATE} Hz"
    )

    print(
        f"Window            : "
        f"{WINDOW_SAMPLES} samples"
    )

    print(
        f"Frame size        : "
        f"{FRAME_SIZE} samples"
    )

    print(
        f"Frame hop         : "
        f"{FRAME_HOP} samples"
    )

    print(
        f"FFT size          : "
        f"{FFT_SIZE}"
    )

    print(
        f"Mel filters       : "
        f"{NUM_MEL}"
    )

    print(
        f"Pre-emphasis      : "
        f"{PRE_EMPHASIS}"
    )

    print(
        f"Log floor         : "
        f"{LOG_FLOOR}"
    )

    expected_frames = (
        (
            WINDOW_SAMPLES
            - FRAME_SIZE
        )
        // FRAME_HOP
    ) + 1

    print(
        f"Frames per sample : "
        f"{expected_frames}"
    )

    print(
        f"Feature dimensions : "
        f"{expected_frames} x {NUM_MEL}"
    )

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "Building feature windows..."
    )

    hamming = (
        build_hamming_window()
    )

    mel_filters = (
        build_mel_filterbank()
    )

    print(
        "Hamming window    : ready"
    )

    print(
        "Mel filter bank   : ready"
    )

    for split_name in (
        "train",
        "val",
        "test"
    ):

        process_split(
            split_name,
            hamming,
            mel_filters
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()