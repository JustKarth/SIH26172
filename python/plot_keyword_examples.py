import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path


FILES = [
    "dataset/raw/keyword/ovos/tng-computer-335.wav",
    "dataset/raw/keyword/ovos/computer-en-780c5c6f-ff12-4789-8dfe-dd8cfe57bfc6.wav",
]


for path in FILES:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)

    audio = audio.mean(axis=1)

    time = [i / sr for i in range(len(audio))]

    plt.figure(figsize=(14, 4))
    plt.plot(time, audio)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(Path(path).name)
    plt.grid(True)
    plt.tight_layout()

    output = Path(path).stem + "_waveform.png"
    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Saved: {output}")


print("Done.")