# SIH 26172 — Low Latency and Efficient Voice Activator for Edge Devices

## Project Status

**Current milestone: Audio acquisition and feature-extraction pipeline completed and verified on ESP32.**

The current firmware successfully performs:

```text
INMP441 MEMS Microphone
        ↓
ESP32 I2S
        ↓
I2S DMA
        ↓
Software Ring Buffer
        ↓
25 ms Audio Frames
        ↓
10 ms Hop
        ↓
Hamming Window
        ↓
512-point FFT
        ↓
40-bin Mel Filter Bank
        ↓
Log-Mel Features
```

The next major milestone is **Python ↔ ESP32 feature validation**, followed by the actual Keyword Spotting (KWS) ML model.

---

# 1. Problem Statement

**SIH Problem Statement:** 26172 / PS 172

**Title:** Low Latency and Efficient Voice Activator for Edge Devices

The system is intended to provide an always-listening, low-power voice activation system on an ESP32-class edge device.

The basic architecture is:

```text
                ┌─────────────────────┐
                │      INMP441        │
                │    MEMS Microphone  │
                └──────────┬──────────┘
                           │ I2S
                           ↓
                ┌─────────────────────┐
                │       ESP32         │
                │                     │
                │ Audio Capture       │
                │       ↓             │
                │ Feature Extraction  │
                │       ↓             │
                │ Tiny KWS Model      │
                └──────────┬──────────┘
                           │
                    Keyword detected
                           ↓
                     Wi-Fi / TCP
                           ↓
                   Remote ASR Server
                           ↓
                      Transcription
```

The edge device should continuously listen for a wake word while keeping memory usage and idle CPU utilization low.

Important project requirements include:

* Open-source software/components.
* No proprietary/commercial wake-word SDKs.
* Local keyword detection.
* Low false-activation rate.
* Low keyword-detection latency.
* Low RAM/Flash usage.
* Low idle CPU utilization.
* Audio streaming to a remote ASR server only after wake-word detection.

---

# 2. Hardware

## ESP32

Current board:

* ESP32 DevKit V1-class board
* ESP32-WROOM-32-class module
* 30-pin board
* ESP32-D0WD-V3
* Chip revision: v3.1
* Dual-core
* Current configured CPU frequency: **160 MHz**
* Flash: **4 MB**

This is **not an ESP32-S3**.

## Microphone

Microphone:

**INMP441 I2S MEMS microphone**

Current wiring:

| INMP441    | ESP32   |
| ---------- | ------- |
| VDD        | 3.3 V   |
| GND        | GND     |
| SD         | GPIO 33 |
| WS / LRCL  | GPIO 25 |
| SCK / BCLK | GPIO 26 |
| L/R        | GND     |

The microphone outputs digital I2S audio.

---

# 3. Development Environment

## Operating System

Development is currently being done on Windows.

## ESP-IDF

```text
ESP-IDF: v6.1
Location: C:\esp\v6.1\esp-idf
```

## Project

```text
D:\Projects\SIH26172
```

Firmware:

```text
D:\Projects\SIH26172\firmware
```

## Python

ESP-IDF environment:

```text
C:\Users\Karthik\.espressif\python_env\idf6.1_py3.14_env
```

Python version selected:

```text
Python 3.14.2
```

## Serial Port

ESP32 is currently connected as:

```text
COM8
```

---

# 4. Building and Flashing

Open an ESP-IDF-enabled command prompt.

```bat
cd /d C:\esp\v6.1\esp-idf
call export.bat
```

Then:

```bat
cd /d D:\Projects\SIH26172\firmware
idf.py build
```

Flash:

```bat
idf.py -p COM8 flash
```

Flash and monitor:

```bat
idf.py -p COM8 flash monitor
```

Exit monitor with:

```text
Ctrl+]
```

---

# 5. Firmware Structure

Current firmware source files:

```text
firmware/
├── CMakeLists.txt
├── main.c
├── audio_pipeline.h
├── audio_pipeline.c
├── audio_frames.h
├── audio_frames.c
├── audio_features.h
└── audio_features.c
```

Current CMake configuration:

```cmake
idf_component_register(
    SRCS
        "main.c"
        "audio_pipeline.c"
        "audio_frames.c"
        "audio_features.c"
    INCLUDE_DIRS
        "."
    REQUIRES
        esp_driver_i2s
        esp_timer
)
```

---

# 6. Audio Configuration

Current audio target:

```text
Sample rate:       16000 Hz
Channels:          1 (mono)
I2S sample size:   32-bit
Output ML format:  PCM16
```

The INMP441 provides 24-bit audio in I2S slots. The ESP32 currently captures 32-bit I2S samples.

The current conversion used by the feature pipeline is:

```c
int16_t sample = (int16_t)(raw_sample >> 16);
```

This was tested experimentally.

Raw sample statistics showed sensible behavior:

* Quiet audio produced relatively small PCM values.
* Speech produced substantially larger values.
* Values remained within the expected int16 range after conversion.

The exact bit alignment should eventually be validated against a Python/reference recording pipeline.

---

# 7. Audio Capture Pipeline

## Architecture

The current capture path is:

```text
INMP441
   ↓
I2S peripheral
   ↓
I2S DMA
   ↓
i2s_channel_read()
   ↓
temporary DMA sample buffer
   ↓
software ring buffer
```

The ESP-IDF 6.1 I2S API is used:

```text
i2s_new_channel()
i2s_channel_init_std_mode()
i2s_channel_enable()
i2s_channel_read()
```

The old Arduino-style `driver/i2s.h` API is not being used in the final firmware.

---

# 8. DMA

Current DMA configuration:

```text
DMA descriptors: 8
DMA samples/frame: 256
```

At 16 kHz:

```text
256 samples / 16000 samples/sec
≈ 16 ms
```

Each DMA buffer therefore represents roughly 16 ms of audio.

Conceptually, DMA performs the repetitive transfer:

```text
I2S peripheral → RAM
```

without requiring the CPU to manually service every sample.

The CPU still copies the DMA-delivered chunk into the application-level ring buffer.

---

# 9. Software Ring Buffer

The application maintains approximately one second of audio history:

```c
#define AUDIO_RING_SAMPLES 16000
```

This is important because the eventual KWS system will need continuous audio history and eventually a **pre-roll buffer**.

The ring buffer supports:

* writing incoming audio
* checking available samples
* reading samples
* non-destructive peeking
* tracking total samples captured
* tracking dropped samples

The total sample counter is 64-bit so it can be used as an absolute audio timeline:

```c
uint64_t
```

This is useful for associating frames with exact sample positions.

---

# 10. Frame Extraction

Current frame configuration:

```text
Frame size: 400 samples
Frame duration: 25 ms

Frame hop: 160 samples
Hop duration: 10 ms
```

Therefore:

```text
400 / 16000 = 25 ms
160 / 16000 = 10 ms
```

Consecutive frames overlap by:

```text
400 - 160 = 240 samples
240 / 16000 = 15 ms
```

So the system analyzes audio every 10 ms using 25 ms windows.

Example:

```text
Frame 0:     [-------------------------] 25 ms
Frame 1:             [-------------------------] 25 ms
Frame 2:                     [-------------------------] 25 ms
                     ↑
                  10 ms hop
```

The frame extractor keeps track of an absolute sample index.

For example:

```text
frame 99  → sample 15840
frame 199 → sample 31840
```

Difference:

```text
16000 samples
```

which corresponds to exactly one second.

This confirms the frame timing is behaving correctly.

---

# 11. Why 25 ms Frames?

Speech is non-stationary.

The frequency content changes rapidly as different phonemes are spoken.

A full-second FFT would mix together many different sounds.

Short-time analysis allows the system to observe:

```text
time → changing frequency content
```

25 ms is a conventional speech-processing scale and is appropriate for the current prototype.

---

# 12. Feature Extraction

Current feature representation:

```text
Raw audio
    ↓
PCM normalization
    ↓
Pre-emphasis
    ↓
Hamming window
    ↓
Zero padding
    ↓
512-point FFT
    ↓
Power spectrum
    ↓
40 Mel filters
    ↓
Log
    ↓
40-dimensional feature vector
```

The current ESP32 implementation uses:

```text
FFT size: 512
Mel filters: 40
Frequency range: 20 Hz – 8000 Hz
Sample rate: 16000 Hz
```

---

# 13. Hamming Window

Each 400-sample frame is multiplied by a Hamming window.

The purpose is to reduce discontinuities at the frame boundaries.

The FFT treats a finite frame as though it repeats periodically.

Without windowing, an artificial discontinuity can appear at the boundaries and cause spectral leakage.

The Hamming coefficients are precomputed during feature-extractor initialization rather than recalculated for every frame.

---

# 14. FFT

Current FFT:

```text
Input samples: 400
FFT size: 512
```

The remaining:

```text
512 - 400 = 112 samples
```

are zero padded.

Zero padding does not add information to the audio.

It allows the FFT to operate at a convenient power-of-two size.

Current frequency-bin spacing:

```text
16000 / 512 = 31.25 Hz
```

For a real-valued input signal, only bins:

```text
0 ... 256
```

contain unique frequency information.

The remaining FFT bins contain the conjugate-symmetric mirror.

---

# 15. Power Spectrum

The FFT produces complex values:

```text
X[k] = real + j*imaginary
```

The power spectrum is computed as:

```text
power[k] = real[k]^2 + imag[k]^2
```

The input audio is real-valued, but the FFT output can contain imaginary components.

For real input:

```text
X[N-k] = conjugate(X[k])
```

so the second half of the FFT is redundant.

---

# 16. Mel Filter Bank

The current implementation uses:

```text
40 Mel filters
```

covering approximately:

```text
20 Hz → 8000 Hz
```

The filters are triangular and map the FFT power spectrum into perceptually motivated frequency bands.

The purpose is to avoid giving the ML model unnecessarily fine-grained frequency resolution.

The result for one frame is:

```text
40 Mel energies
```

---

# 17. Log Compression

The Mel energies are converted using logarithmic compression.

This reduces the enormous dynamic range of raw spectral power and produces a feature representation more suitable for speech/KWS models.

The final output of one frame is therefore:

```text
40 floating-point log-Mel values
```

A sequence of frames produces a time-frequency matrix.

For approximately one second:

```text
~98 frames × 40 features
```

The exact number depends on the framing convention.

---

# 18. Current ESP32 Feature Extractor

The current implementation precomputes:

* Hamming window
* Mel filter-bank coefficients

and uses static buffers for:

* FFT real values
* FFT imaginary values
* Mel filter bank

The feature extraction function accepts:

```c
const int32_t *raw_frame
```

and produces:

```c
float *features
```

with:

```text
40 features
```

for one frame.

---

# 19. Current Feature-Extraction Benchmark

The latest successful ESP32 run produced approximately:

```text
3181–3194 microseconds
```

per 25 ms frame.

Typical values:

```text
frame=99   → 3194 us
frame=199  → 3181 us
frame=299  → 3194 us
frame=399  → 3182 us
...
```

So the current feature extraction cost is approximately:

```text
3.19 ms per frame
```

Since frames arrive every:

```text
10 ms
```

continuous feature extraction would consume approximately:

```text
3.19 / 10 × 100
≈ 31.9%
```

of one CPU's available processing time at the current 160 MHz CPU frequency.

This is a **baseline measurement**, not yet an optimization target.

Do not optimize the feature pipeline until its numerical correctness has been validated against the Python implementation.

---

# 20. Current Test Task

The current `main.c` starts:

```text
audio_pipeline
audio_frames
audio_features
```

and then runs a feature-test task.

Every 100 frames it prints:

```text
frame number
sample index
processing time
mel[0]
mel[20]
mel[39]
```

Example:

```text
FEATURE_TEST: frame=999 sample=159840 time=3181 us ...
```

The feature values change with the captured audio, indicating that the pipeline is processing changing microphone input rather than returning a constant result.

---

# 21. Watchdog Issue

Earlier versions of the feature-test task caused Task Watchdog Timer warnings.

The issue was associated with the polling/scheduling behavior of the feature-processing task.

The test task was adjusted to:

* lower its priority
* use conditional delays
* avoid continuously polling without yielding

The latest run completed without watchdog warnings.

**Do not disable the watchdog to hide the issue.**

If the issue returns under heavier processing, investigate task scheduling/core affinity rather than simply increasing watchdog timeouts.

A possible future approach is to pin the feature-processing task to CPU1 so that CPU0 remains available for other system work, but this has not been made part of the final architecture yet.

---

# 22. Current Memory Situation

The ESP32 currently reports approximately:

```text
DRAM:
~6 KiB
~70 KiB
~14 KiB D/IRAM
~111 KiB D/IRAM
~82 KiB IRAM
```

The important application allocations include:

* ~64 KB raw ring buffer (`16000 × 4 bytes`)
* Mel filter bank
* FFT arrays
* Hamming window
* DMA buffers
* FreeRTOS task stacks
* application/runtime overhead

The project must eventually measure **actual runtime free heap and tensor arena usage** before claiming compliance with the `<256 KB RAM` requirement.

---

# 23. Important Current Caveat: Feature Specification

The original ECE reference code and the current ESP32 implementation are not completely identical.

The current ESP32 feature extractor includes:

```text
pre-emphasis
```

The original ECE code supplied during development did not actually apply pre-emphasis.

Therefore, before ML training begins, the project must **freeze the exact feature-extraction specification**.

The Python reference implementation must use exactly the same:

* sample conversion
* normalization
* pre-emphasis decision
* frame length
* frame hop
* window function
* FFT size
* FFT convention
* power calculation
* Mel filter-bank construction
* frequency range
* number of Mel filters
* logarithm/flooring convention

Otherwise, the model trained in Python and the features produced on ESP32 may not match.

---

# 24. What Is NOT Implemented Yet

The following components are still pending:

## KWS model

No neural network has been deployed yet.

Planned direction:

```text
Tiny CNN / DS-CNN
```

with possible classes such as:

```text
KEYWORD
UNKNOWN
SILENCE
```

The exact wake word has not been permanently fixed.

An earlier candidate was "Nova", but the final keyword should depend on dataset availability and training considerations.

---

## Dataset

The final KWS dataset still needs to be created/selected.

It should contain:

### Positive examples

Multiple speakers and conditions:

* different voices
* accents
* pitch
* speaking speed
* volume
* microphone distance
* room conditions
* pronunciation variation

### Negative examples

Including:

* normal speech
* unrelated words
* acoustically similar words
* silence
* TV/audio
* music
* keyboard sounds
* fan/room noise
* traffic/background noise
* hard negatives

Speaker-disjoint train/validation/test splits should be used.

---

# 25. Planned ML Pipeline

Target architecture:

```text
Audio
 ↓
25 ms / 10 ms frames
 ↓
40-bin log-Mel
 ↓
Tiny KWS CNN / DS-CNN
 ↓
KEYWORD / UNKNOWN / SILENCE
```

Training:

```text
Dataset
 ↓
Python preprocessing
 ↓
Feature extraction
 ↓
Model training
 ↓
Validation
 ↓
Quantization
 ↓
INT8 TFLite model
```

Deployment:

```text
INT8 TFLite
 ↓
TensorFlow Lite Micro
 ↓
Tensor arena
 ↓
ESP32
```

---

# 26. Wake-Word Controller

After KWS is working, the firmware should implement a state machine similar to:

```text
BOOT
  ↓
INITIALIZING
  ↓
CONNECTING
  ↓
IDLE
  ↓
WAKE_DETECTED
  ↓
STREAMING
  ↓
FINALIZING
  ↓
IDLE
```

The KWS output should not trigger streaming from a single isolated frame.

The controller should eventually use:

* score threshold
* temporal smoothing
* consecutive detections
* debounce
* cooldown

---

# 27. Pre-Roll

The existing approximately one-second ring buffer is useful for implementing pre-roll.

When the keyword is detected, some audio immediately before the detected keyword can be sent to the server.

This prevents loss of the first part of the user's command due to detection latency.

Target pre-roll:

```text
~500–1000 ms
```

The exact amount should be experimentally selected.

---

# 28. Voice Activity Detection

After the wake word:

```text
ESP32 → continuously stream audio
```

The stream eventually needs to stop.

Planned mechanism:

```text
VAD / energy detection
+
silence timeout
+
maximum recording duration
+
server safety timeout
```

Possible end reasons:

```text
vad_silence
max_duration
user_cancel
error
```

The exact VAD algorithm has not yet been implemented.

---

# 29. Networking Architecture

Planned architecture:

```text
ESP32
  ↓
Wi-Fi
  ↓
TCP/IP
  ↓
WebSocket
  ↓
Laptop / Remote ASR Server
```

WebSocket was selected because it provides a convenient persistent bidirectional connection while running over TCP.

TCP/WebSocket already provides ordered, reliable delivery.

Therefore the current protocol does **not** require application-level sequence numbers for every audio packet.

---

# 30. Planned Audio Streaming Format

Audio:

```text
Sample rate: 16000 Hz
Channels: 1
Format: PCM signed 16-bit little-endian
```

Raw PCM bandwidth:

```text
16000 samples/sec × 2 bytes
= 32000 bytes/sec
≈ 32 KB/s
```

Audio should be transmitted as **binary WebSocket data**, not Base64.

---

# 31. Planned WebSocket Protocol

### Start

```json
{
  "type": "start",
  "session_id": "01J...",
  "audio": {
    "sample_rate": 16000,
    "channels": 1,
    "format": "pcm_s16le"
  }
}
```

Server response:

```json
{
  "type": "start_ack",
  "session_id": "01J..."
}
```

Audio:

```text
Binary WebSocket messages
↓
Raw PCM16 bytes
```

End:

```json
{
  "type": "end",
  "reason": "vad_silence"
}
```

Possible reasons:

```text
vad_silence
max_duration
user_cancel
error
```

Final result:

```json
{
  "type": "result",
  "session_id": "01J...",
  "status": "complete",
  "text": "..."
}
```

---

# 32. Networking State Machine

The server should maintain states similar to:

```text
CONNECTED
    ↓
STARTED
    ↓
RECEIVING
    ↓
ENDED
    ↓
COMPLETED
```

The receiver should validate:

* control-message ordering
* session ID
* message type
* binary/audio data placement
* stream termination
* malformed messages

---

# 33. Latency Measurement

The final system needs real latency measurements.

Planned timestamps:

```text
T0 = keyword ends
T1 = keyword detection occurs
T2 = audio streaming begins
T3 = server receives audio
T4 = ASR produces partial result
T5 = ASR produces final result
```

The SIH-relevant measurement includes:

```text
T3 - T0
```

for keyword-end → server-audio-received latency.

Other useful measurements:

```text
T1 - T0
```

Wake-word detection latency.

```text
T5 - T0
```

End-to-end transcription latency.

---

# 34. Final Evaluation Metrics

The finished system should measure at least:

## KWS

* True positive rate / recall
* False positive rate
* False activations per hour
* Confusion matrix
* Detection latency

## Edge resource usage

* Model Flash size
* Tensor arena RAM
* Total working RAM
* Free heap
* CPU utilization
* Idle CPU utilization
* Inference time

## Audio/networking

* Keyword-end → stream-start latency
* Keyword-end → server receive latency
* Network throughput
* Packet/message behavior
* ASR latency

## End-to-end

```text
Keyword spoken
        ↓
Keyword detected
        ↓
Audio streamed
        ↓
Server receives audio
        ↓
ASR
        ↓
Final transcription
```

---

# 35. Current Baseline

As of the current implementation:

| Component                         | Status                 |
| --------------------------------- | ---------------------- |
| ESP32 setup                       | Complete               |
| ESP-IDF 6.1                       | Complete               |
| INMP441 wiring                    | Complete               |
| I2S capture                       | Working                |
| DMA capture                       | Working                |
| 16 kHz mono audio                 | Working                |
| Audio ring buffer                 | Working                |
| 25 ms frames                      | Working                |
| 10 ms hop                         | Working                |
| 512-point FFT                     | Working                |
| 40 Mel filters                    | Working                |
| Log-Mel features                  | Working                |
| Continuous feature processing     | Working                |
| Watchdog stability                | Working in latest test |
| Feature benchmark                 | ~3.19 ms/frame         |
| Python reference                  | **Next**               |
| Python/ESP32 numerical validation | Pending                |
| KWS dataset                       | Pending                |
| KWS model                         | Pending                |
| INT8 quantization                 | Pending                |
| TFLite Micro                      | Pending                |
| Wake-word state machine           | Pending                |
| VAD                               | Pending                |
| Wi-Fi streaming                   | Pending                |
| WebSocket protocol                | Pending                |
| Remote ASR                        | Pending                |
| End-to-end latency measurement    | Pending                |
| Final CPU/RAM optimization        | Pending                |
| SIH evaluation                    | Pending                |

---

# 36. Immediate Next Step

**Do not optimize the ESP32 feature extractor yet.**

The next task is:

## Build the Python reference feature extractor

It must reproduce the ESP32 pipeline exactly.

Then:

1. Record/capture a known audio segment.
2. Feed identical audio into Python.
3. Feed identical audio into ESP32.
4. Extract the corresponding 40-dimensional features.
5. Compare the numerical outputs.
6. Fix any mismatch.
7. Freeze the feature specification.
8. Only then begin KWS model development.

The most important goal at this stage is:

```text
Python feature vector ≈ ESP32 feature vector
```

Once this is verified, the ML pipeline can be developed confidently.

---

# 37. Design Principle Going Forward

The current implementation is intentionally a **correctness-first prototype**.

Do not prematurely optimize based solely on the current ~32% CPU estimate.

First establish:

```text
correct audio
      ↓
correct features
      ↓
correct ML model
      ↓
correct embedded inference
      ↓
correct wake detection
      ↓
correct streaming
      ↓
correct ASR
```

Then optimize the complete system against the SIH constraints.

The current feature-extraction benchmark (~3.19 ms per 10-ms hop at 160 MHz) should be treated as the baseline against which future optimizations are measured.
