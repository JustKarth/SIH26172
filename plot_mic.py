import serial
import matplotlib.pyplot as plt

PORT = "COM8"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)

plt.ion()

fig, ax = plt.subplots()
line, = ax.plot([])

ax.set_title("INMP441 Microphone")
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.grid(True)

while True:
    raw = ser.readline().decode(errors="ignore").strip()

    if not raw.startswith("DATA "):
        continue

    try:
        samples = [
            int(x)
            for x in raw[5:].split(",")
            if x
        ]
    except ValueError:
        continue

    if not samples:
        continue

    line.set_data(range(len(samples)), samples)

    ax.set_xlim(0, len(samples))

    margin = max(abs(min(samples)), abs(max(samples)), 1)
    ax.set_ylim(-margin * 1.1, margin * 1.1)

    fig.canvas.draw()
    fig.canvas.flush_events()

ser.close()