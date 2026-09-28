import numpy as np

from .feature_extractor import extract_log_mel


FRAME_SIZE = 400


def make_silence() -> np.ndarray:
    """All-zero raw I2S samples."""
    return np.zeros(FRAME_SIZE, dtype=np.int32)


def make_ramp() -> np.ndarray:
    """
    Deterministic ramp.

    Values are chosen so that the ESP32 >> 16 conversion
    produces a useful range of signed int16 PCM samples.
    """
    pcm = np.linspace(
        -16000,
        16000,
        FRAME_SIZE,
        dtype=np.int32,
    )

    return pcm.astype(np.int32) << 16


def make_pseudorandom() -> np.ndarray:
    """
    Deterministic pseudo-random int16 PCM signal.

    Uses the same uint32 LCG that the ESP32 parity
    harness will use.
    """

    state = 0x12345678

    raw = np.empty(
        FRAME_SIZE,
        dtype=np.int32,
    )

    for i in range(FRAME_SIZE):
        state = (
            (1664525 * state + 1013904223)
            & 0xFFFFFFFF
        )

        # Upper 16 bits become signed PCM.
        pcm = (state >> 16) & 0xFFFF

        if pcm & 0x8000:
            pcm -= 0x10000

        raw[i] = np.int32(pcm << 16)

    return raw


def print_case(name: str, raw: np.ndarray) -> None:
    features = extract_log_mel(raw)

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("input checksum:", int(np.sum(raw.astype(np.int64))))
    print("feature count:", len(features))

    print("features:")

    for i, value in enumerate(features):
        print(f"{i:02d}: {value:.9f}")


def main() -> None:
    print_case(
        "SILENCE",
        make_silence(),
    )

    print_case(
        "RAMP",
        make_ramp(),
    )

    print_case(
        "PSEUDORANDOM",
        make_pseudorandom(),
    )


if __name__ == "__main__":
    main()