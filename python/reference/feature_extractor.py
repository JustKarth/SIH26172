import numpy as np


# ============================================================
# Feature extraction configuration
# ============================================================

SAMPLE_RATE = 16000.0
FRAME_SIZE = 400          # 25 ms at 16 kHz
FFT_SIZE = 512
NUM_MEL = 40

LOW_FREQ = 20.0
HIGH_FREQ = 8000.0

PRE_EMPH = 0.97
LOG_FLOOR = 1e-10


# ============================================================
# Frequency / Mel conversions
# ============================================================

def _hz_to_mel(hz: float) -> float:
    """Match the ESP32 C implementation."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Match the ESP32 C implementation."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


# ============================================================
# Hamming window
# ============================================================

def _init_hamming_window() -> np.ndarray:
    """
    Create the same 400-point Hamming window used by the ESP32.

    C equivalent:

        0.54f - 0.46f *
        cosf(2.0f * PI * n / (FRAME_SIZE - 1))
    """

    n = np.arange(FRAME_SIZE, dtype=np.float32)

    window = (
        0.54
        - 0.46
        * np.cos(
            2.0 * np.pi * n / (FRAME_SIZE - 1)
        )
    )

    return window.astype(np.float32)


# ============================================================
# Mel filter bank
# ============================================================

def _init_mel_filters() -> np.ndarray:
    """
    Construct the 40 triangular Mel filters.

    FFT_SIZE = 512
    Therefore the one-sided spectrum contains:

        512 / 2 + 1 = 257 bins
    """

    low_mel = _hz_to_mel(LOW_FREQ)
    high_mel = _hz_to_mel(HIGH_FREQ)

    # 40 filters require 42 boundary points.
    num_points = NUM_MEL + 2

    mel_points = np.zeros(
        num_points,
        dtype=np.float32,
    )

    bin_points = np.zeros(
        num_points,
        dtype=np.int32,
    )

    # Generate Mel-spaced points and convert to FFT bins.
    for i in range(num_points):

        mel_points[i] = (
            low_mel
            + (
                float(i)
                / float(NUM_MEL + 1)
            )
            * (high_mel - low_mel)
        )

        hz = _mel_to_hz(mel_points[i])

        bin_points[i] = int(
            np.floor(
                ((FFT_SIZE + 1) * hz)
                / SAMPLE_RATE
            )
        )

    num_bins = FFT_SIZE // 2 + 1

    filters = np.zeros(
        (NUM_MEL, num_bins),
        dtype=np.float32,
    )

    # Construct each triangular filter.
    for m in range(1, NUM_MEL + 1):

        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        # Rising edge.
        for k in range(left, center):

            if (
                0 <= k < num_bins
                and center != left
            ):
                filters[m - 1, k] = (
                    float(k - left)
                    / float(center - left)
                )

        # Falling edge.
        for k in range(center, right):

            if (
                0 <= k < num_bins
                and right != center
            ):
                filters[m - 1, k] = (
                    float(right - k)
                    / float(right - center)
                )

    return filters


# ============================================================
# Precomputed feature-extraction state
# ============================================================

_HAMMING_WINDOW = _init_hamming_window()
_MEL_FILTERS = _init_mel_filters()


# ============================================================
# Feature extraction
# ============================================================

def extract_log_mel(raw_frame: np.ndarray) -> np.ndarray:
    """
    Convert one 400-sample ESP32 raw I2S frame into
    40 log-Mel filter-bank features.

    Input:
        raw_frame:
            1-D array containing exactly 400 signed int32
            values corresponding to the raw ESP32 I2S samples.

    Output:
        1-D float32 array containing exactly 40 log-Mel
        filter-bank energies.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if raw_frame.shape != (FRAME_SIZE,):
        raise ValueError(
            f"Input must be a 1D array of "
            f"{FRAME_SIZE} samples."
        )

    raw_frame = np.asarray(
        raw_frame,
        dtype=np.int32,
    )

    # --------------------------------------------------------
    # 1. Convert raw I2S sample to int16 PCM
    # --------------------------------------------------------
    #
    # ESP32 C:
    #
    #     int16_t pcm =
    #         (int16_t)(raw_frame[i] >> 16);
    #
    # np.right_shift on signed int32 performs the
    # corresponding arithmetic right shift.
    #

    shifted = np.right_shift(
        raw_frame,
        16,
    )

    pcm = shifted.astype(
        np.int16,
    )

    # --------------------------------------------------------
    # 2. Normalize PCM
    # --------------------------------------------------------

    samples = (
        pcm.astype(np.float32)
        / np.float32(32768.0)
    )

    # --------------------------------------------------------
    # 3. Pre-emphasis
    # --------------------------------------------------------
    #
    # ESP32:
    #
    #     y[0] = x[0]
    #     y[n] = x[n] - 0.97 * x[n-1]
    #

    emphasized = np.empty(
        FRAME_SIZE,
        dtype=np.float32,
    )

    emphasized[0] = samples[0]

    emphasized[1:] = (
        samples[1:]
        - np.float32(PRE_EMPH)
        * samples[:-1]
    )

    # --------------------------------------------------------
    # 4. Hamming window
    # --------------------------------------------------------

    windowed = (
        emphasized
        * _HAMMING_WINDOW
    )

    # --------------------------------------------------------
    # 5. Zero-pad to FFT_SIZE
    # --------------------------------------------------------

    padded = np.zeros(
        FFT_SIZE,
        dtype=np.float32,
    )

    padded[:FRAME_SIZE] = windowed

    # --------------------------------------------------------
    # 6. FFT
    # --------------------------------------------------------
    #
    # NumPy uses a highly optimized FFT implementation.
    #
    # The ESP32 uses the project's handwritten float32
    # radix-2 FFT.
    #
    # Therefore tiny floating-point differences are expected.
    #

    fft_out = np.fft.fft(
        padded,
    )

    # --------------------------------------------------------
    # 7. Power spectrum
    # --------------------------------------------------------

    num_bins = FFT_SIZE // 2 + 1

    real = np.real(
        fft_out[:num_bins]
    )

    imag = np.imag(
        fft_out[:num_bins]
    )

    power = (
        real * real
        + imag * imag
    ).astype(np.float32)

    # --------------------------------------------------------
    # 8. Apply Mel filter bank
    # --------------------------------------------------------

    mel_energies = np.dot(
        _MEL_FILTERS,
        power,
    )

    mel_energies = mel_energies.astype(
        np.float32
    )

    # --------------------------------------------------------
    # 9. Energy floor
    # --------------------------------------------------------

    mel_energies = np.maximum(
        mel_energies,
        np.float32(LOG_FLOOR),
    )

    # --------------------------------------------------------
    # 10. Natural logarithm
    # --------------------------------------------------------

    features = np.log(
        mel_energies,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    assert features.shape == (
        NUM_MEL,
    )

    return features