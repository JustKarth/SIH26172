import numpy as np
import soundfile as sf
from pathlib import Path


FILES = [
    "dataset/raw/keyword/ovos/tng-computer-335.wav",
    "dataset/raw/keyword/ovos/computer-en-780c5c6f-ff12-4789-8dfe-dd8cfe57bfc6.wav",
]

FRAME = 400   # 25 ms
HOP = 160     # 10 ms


for path in FILES:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)

    rms = []

    for start in range(0, len(audio) - FRAME + 1, HOP):
        x = audio[start:start + FRAME]
        rms.append(np.sqrt(np.mean(x * x) + 1e-12))

    rms = np.asarray(rms)

    order = np.argsort(rms)[::-1]

    print()
    print("=" * 70)
    print(Path(path).name)
    print("=" * 70)

    print(f"Sample rate : {sr}")
    print(f"Duration    : {len(audio) / sr:.3f} s")
    print(f"Overall RMS : {np.sqrt(np.mean(audio * audio)):.6f}")
    print(f"Peak        : {np.max(np.abs(audio)):.6f}")

    print("\nRMS percentiles:")
    for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  P{p:3d}: {np.percentile(rms, p):.6f}")

    print("\n10 loudest 25-ms frames:")
    for idx in order[:10]:
        print(
            f"  {idx * HOP / sr:6.3f}s"
            f"  -> RMS {rms[idx]:.6f}"
        )

    # Count frames above relative thresholds.
    print("\nFrames above RMS thresholds:")
    for multiplier in [1.5, 2, 3, 5, 10]:
        threshold = np.percentile(rms, 20) * multiplier
        count = np.sum(rms >= threshold)
        print(
            f"  {multiplier:>4}x P20 = {threshold:.6f}"
            f"  ({count}/{len(rms)} frames)"
        )