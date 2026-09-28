# Agents.md — SIH 26172 Engineering Rules

## Project

**SIH Problem Statement 26172 — Low Latency and Efficient Voice Activator for Edge Devices**

The system is an ultra-lightweight local wake-word detector running continuously on an ESP32, followed by low-overhead audio streaming to a remote ASR server after wake-word detection.

The primary target hardware is:

* ESP32-D0WD-V3, revision 3.1
* ESP32-WROOM-32-class development board
* 4 MB flash
* Current ESP-IDF: 6.1
* Current development environment: Windows
* Microphone: INMP441
* I2S:

  * BCLK/SCK: GPIO26
  * WS/LRCLK: GPIO25
  * SD: GPIO33
  * L/R: GND
  * 3.3 V power

---

# 1. Highest-Priority Rules

These rules apply to **every coding agent**.

### 1.1 Inspect before editing

Before modifying anything:

1. Run `git status`.
2. Inspect the relevant files.
3. Inspect the current diff.
4. Understand existing interfaces and data flow.
5. Identify whether another agent may already be working on the relevant component.

Never blindly overwrite existing work.

### 1.2 Never revert unrelated work

Do not:

* reset the repository;
* checkout files to discard changes;
* use `git clean`;
* overwrite another agent's changes;
* revert changes merely because they are unfamiliar.

If existing changes appear unrelated to your task, leave them untouched.

### 1.3 Stay inside your assigned scope

Modify only files required for the assigned task.

Do not "clean up" unrelated code.

Do not refactor working code merely because you prefer another style.

Do not redesign the architecture without explicit approval.

### 1.4 Preserve established interfaces

Before changing a public API, header, struct, function signature, file format, or protocol:

* determine who uses it;
* explain why the change is required;
* minimize compatibility impact.

Prefer adapting your implementation around an existing interface rather than changing the interface.

### 1.5 Test every meaningful change

After modifying code:

1. Build/test the affected component.
2. Run the relevant existing tests.
3. Inspect warnings/errors.
4. Report exactly what was tested.

Never claim something works without actually testing it.

### 1.6 Do not silently weaken requirements

Never solve a failing test by:

* increasing tolerances without investigating;
* disabling assertions;
* reducing test coverage;
* removing problematic test cases;
* lowering performance requirements;
* ignoring memory/CPU failures.

If a requirement cannot currently be met, report the failure and investigate the actual cause.

---

# 2. Project Architecture

The intended architecture is:

```text
                    EDGE / ESP32
┌───────────────────────────────────────────────────────┐
│                                                       │
│  INMP441                                               │
│      │                                                │
│      ▼                                                │
│  I2S + DMA                                             │
│      │                                                │
│      ▼                                                │
│  Audio Ring Buffer                                     │
│      │                                                │
│      ▼                                                │
│  25 ms frames / 10 ms hop                              │
│      │                                                │
│      ▼                                                │
│  Feature Extraction                                    │
│      │                                                │
│      ▼                                                │
│  Tiny KWS CNN                                          │
│      │                                                │
│      ▼                                                │
│  Temporal Wake Controller                              │
│      │                                                │
│      ▼                                                │
│  WAKE DETECTED                                         │
│      │                                                │
│      └──────────────► Audio/network handoff            │
│                                                       │
└───────────────────────────────────────────────────────┘
                             │
                             ▼
                    Wi-Fi / Network
                             │
                             ▼
                       Python server
                             │
                             ▼
                     Open-source ASR
```

The system is divided into seven logical stages:

1. Hardware + I2S
2. Audio Pipeline
3. KWS ML
4. Embedded Inference
5. Wake-Word Controller
6. Networking
7. ASR + Evaluation

---

# 3. Ownership Boundaries

## ESP32 / Edge Owner

The ESP32 owner controls:

* I2S configuration
* DMA configuration
* audio capture
* audio ring buffer
* framing
* feature extraction
* embedded inference
* model deployment
* quantization
* tensor arena
* ESP32 RAM/Flash measurements
* CPU measurements
* wake-word temporal controller
* wake detection
* pre-roll audio
* audio handoff to networking

The ESP32 owner is responsible for ensuring the complete edge-side system actually runs continuously on the target hardware.

## Networking / ASR Owner

The networking owner controls:

* Wi-Fi transport
* ESP32-to-server protocol
* Python server
* audio receiving
* server-side buffering
* ASR integration
* network latency measurements
* server/ASR latency measurements
* end-to-end latency instrumentation

The networking owner must not redesign the ESP32 feature extractor or KWS model.

## ML / Model Owner

The ML owner controls:

* training scripts
* model conversion
* quantization experiments
* model-size analysis
* PC-side evaluation
* model accuracy/recall/precision analysis

The ML owner must not alter the production ESP32 audio pipeline to make a model work.

---

# 4. Current Audio Specification

The canonical audio representation is:

* Sample rate: **16,000 Hz**
* Channels: **1**
* PCM representation downstream: **signed 16-bit**
* Frame size: **400 samples**
* Frame duration: **25 ms**
* Hop: **160 samples**
* Hop duration: **10 ms**
* FFT size: **512**
* Mel filters: **40**
* Mel frequency range: **20 Hz–8 kHz**
* Pre-emphasis: **alpha = 0.97**
* Window: **Hamming**
* Log floor: **1e-10**
* 1-second feature representation: **98 frames × 40 features**

Do not change these parameters casually.

Any change must be treated as an explicit experiment and must preserve the existing reference implementation.

---

# 5. Feature Extractor Is a Contract

The Python and ESP32 feature extractors have already been brought into numerical parity.

The feature pipeline is:

```text
PCM16
  ↓
/ 32768
  ↓
Pre-emphasis (alpha=0.97)
  ↓
Hamming window
  ↓
512-point zero-padded FFT
  ↓
Power spectrum
  ↓
40 triangular Mel filters
  ↓
log()
  ↓
40 log-Mel features
```

The ESP32 implementation uses a handwritten float32 radix-2 FFT.

NumPy may produce tiny numerical differences because it uses a different FFT implementation.

### Important

Do **not**:

* replace the FFT without approval;
* change the Mel filter definition;
* remove pre-emphasis;
* change frame/hop sizes;
* change normalization;
* change the log floor;
* silently change PCM alignment;
* introduce a different feature representation.

If feature extraction is modified, parity with Python must be re-established.

---

# 6. I2S / INMP441 Rules

The INMP441 provides I2S audio using 24-bit data in a 32-bit slot.

The ESP32 capture path currently reads 32-bit values.

Do not assume that the microphone's meaningful 24-bit sample occupies a particular bit range without verification.

The current feature extractor uses:

```c
(int16_t)(raw_frame[i] >> 16)
```

followed by normalization.

This alignment has been experimentally established for the current hardware/configuration.

If changing I2S configuration, re-verify sample alignment experimentally.

Never silently change:

* slot mode;
* bit width;
* channel selection;
* clock configuration;
* sample extraction;
* sign extension.

---

# 7. Audio Pipeline Rules

The intended data flow is:

```text
I2S peripheral
      ↓
DMA-managed buffers
      ↓
i2s_channel_read()
      ↓
local DMA read buffer
      ↓
software ring buffer
      ↓
frame generator
      ↓
feature extractor
```

DMA is a hardware transfer mechanism. Do not describe or implement it as if it were an independent application-level storage layer.

Current relevant constants:

```text
AUDIO_SAMPLE_RATE       = 16000
AUDIO_CHANNELS          = 1
AUDIO_RING_SAMPLES      = 16000
AUDIO_DMA_READ_SAMPLES  = 256
```

Current framing:

```text
AUDIO_FRAME_SIZE = 400
AUDIO_FRAME_HOP  = 160
```

Do not redesign the buffering architecture unless explicitly requested.

---

# 8. Current Production Files

Important existing files include:

```text
firmware/main/audio_pipeline.c
firmware/main/audio_pipeline.h

firmware/main/audio_frames.c
firmware/main/audio_frames.h

firmware/main/audio_features.c
firmware/main/audio_features.h

firmware/main/main.c

python/reference/feature_extractor.py
python/reference/parity_test.py

python/preprocess_audio.py
python/extract_features.py
python/validate_features.py
python/train_kws_baseline.py
python/evaluate_kws_thresholds.py
python/evaluate_kws_long_negative.py
```

Before editing any of these, inspect the current version rather than relying on this document alone.

---

# 9. Feature Parity Test Rules

The approved parity experiment used:

* deterministic silence;
* deterministic ramp;
* deterministic pseudo-random int32 vectors;
* explicitly safe `int32_t` / `uint32_t` arithmetic;
* identical input vectors on C and Python sides;
* all 40 feature values;
* initial comparison:

  * `atol = 1e-3`
  * `rtol = 1e-3`

Report:

* maximum absolute error;
* mean absolute error;
* maximum relative error;
* feature index of maximum error.

If tolerance fails:

**Investigate the source of divergence before loosening tolerance.**

Do not simply increase tolerance.

The temporary ESP32 parity harness was experimental.

Production `main.c` should remain free of temporary parity-test code unless explicitly requested.

---

# 10. ML Model Rules

Current baseline model:

```text
Conv2D 16, 5×5
BatchNorm
ReLU
MaxPool 2×2

Conv2D 32, 3×3
BatchNorm
ReLU
MaxPool 2×2

Conv2D 64, 3×3
BatchNorm
ReLU

GlobalAveragePooling

Dense 32
ReLU
Dropout 0.25

Dense 2
Softmax
```

Approximately:

```text
26,034 parameters
~101.7 KB FP32 weights
```

The current baseline achieved approximately:

```text
Test accuracy:          99.02%
Keyword recall:         96.99%
Keyword precision:      97.73%
Negative recall:        99.48%
```

These numbers are **window-level test results**, not continuous-listening false-activation performance.

Never present the 99% accuracy number as proof that the deployed wake-word detector is solved.

---

# 11. Threshold Rules

The CNN output is a probability-like score for the keyword class.

Threshold selection must consider:

* keyword recall;
* false positives;
* false activation events/hour;
* temporal confirmation;
* latency.

Do not select a final production threshold merely because it gives zero false positives on a small test set.

Current threshold `0.60` is only an experimental operating point.

It is **not a permanently approved production threshold**.

---

# 12. Long-Negative Evaluation

The long-negative evaluator scans spontaneous English speech using:

* 1-second windows;
* 10 ms hop;
* the existing feature extractor;
* the existing trained model;
* no retraining;
* no temporal smoothing;
* no debounce.

Adjacent positive windows are grouped into one event per source file.

The metric is:

```text
false activation events / hour
```

This is more relevant to continuous wake-word operation than ordinary classification accuracy.

### Important evaluation caveat

The current Common Voice long-negative scan includes data from the Common Voice corpus that overlaps with the negative data used during model development.

Therefore:

**The full Common Voice scan is an operational stress test, not a perfectly independent held-out false-activation benchmark.**

Do not claim otherwise.

---

# 13. Dataset Rules

Current working wake-word candidate:

```text
computer
```

This candidate was selected because relevant open datasets contain recordings associated with it.

The wake word is not to be changed casually.

Current corpus includes:

* Picovoice wake-word benchmark data;
* OVOS wake-word/community data;
* supplementary synthetic data where applicable;
* Common Voice spontaneous English negatives.

Do not download huge datasets unnecessarily.

Do not create a new dataset unless explicitly assigned.

Do not introduce proprietary wake-word data or proprietary SDKs.

---

# 14. Open-Source Requirement

The project must use open-source components.

Do not introduce:

* proprietary wake-word SDKs;
* commercial cloud-only wake-word APIs;
* closed-source inference dependencies;
* licensing-incompatible datasets;
* software with unclear redistribution restrictions.

If licensing is unclear, flag it instead of silently assuming it is acceptable.

---

# 15. ESP32 Embedded Inference

The embedded inference implementation should target an open-source embedded runtime such as TensorFlow Lite Micro, subject to compatibility testing.

Required measurements:

### Model

* `.tflite` size
* FP32/INT8 size
* number of parameters
* operators used
* input/output tensor shapes
* quantization details

### ESP32

* model Flash usage
* tensor arena size
* runtime RAM
* stack/heap usage
* minimum free heap
* inference latency
* CPU usage

The project target is:

```text
<256 KB total relevant working RAM
<10% CPU while idle/continuously listening
```

Do not confuse:

```text
model size
```

with:

```text
total deployment memory
```

The tensor arena, buffers, stacks, audio ring buffer, runtime state, and other allocations matter.

---

# 16. Quantization Rules

INT8 quantization should be evaluated against FP32.

Report:

```text
FP32 model size
INT8 model size

FP32:
  recall
  precision
  false-positive behavior

INT8:
  recall
  precision
  false-positive behavior

ESP32:
  RAM
  Flash
  inference latency
```

Do not declare INT8 acceptable solely because the file became smaller.

Do not retrain unless explicitly requested.

---

# 17. Wake-Word Controller

The CNN should not necessarily trigger a wake event from one isolated probability sample.

The temporal controller may use:

* thresholding;
* consecutive positive windows;
* score accumulation;
* smoothing;
* temporal voting;
* cooldown/debounce.

Any controller must balance:

```text
false activation rate
        vs
missed wake words
        vs
detection latency
```

The controller must be deterministic and cheap enough for the ESP32.

Avoid dynamic allocation in the hot path.

Avoid unnecessary floating-point computation if an equivalent fixed-point/integer implementation is practical.

Do not sacrifice correctness for micro-optimizations before measuring.

---

# 18. Networking Boundary

The ESP32 KWS system should expose a clean handoff after wake detection.

Conceptually:

```c
wake_detected()
get_preroll_audio()
start_audio_stream()
```

The exact API may differ, but the ownership boundary should remain clear.

The ESP32 owner should not implement the server's ASR stack.

The networking owner should not rewrite the ESP32 feature extractor or KWS pipeline.

Initially, networking may use a manual/synthetic wake trigger to unblock independent development.

---

# 19. Performance Measurement Rules

Performance claims must be measured on the actual target whenever possible.

Do not estimate ESP32 performance from:

* PC execution time;
* Python execution time;
* model parameter count alone;
* theoretical FLOPs.

Measure:

```text
feature extraction latency
inference latency
end-to-end edge detection latency
free heap
minimum heap
audio drops
CPU utilization
Flash usage
tensor arena
continuous runtime stability
```

For continuous listening, run long enough to expose:

* buffer drift;
* dropped samples;
* memory leaks;
* watchdog issues;
* unstable CPU behavior.

---

# 20. Coding Style

For C:

* use fixed-width integer types where representation matters;
* use `size_t` for buffer sizes;
* use `uint64_t` for absolute sample counters;
* avoid implicit signed/unsigned conversions;
* avoid integer overflow;
* avoid hidden narrowing conversions;
* keep ISR/hot-path code minimal;
* avoid dynamic allocation in continuous audio processing.

For Python:

* make random operations deterministic when used for experiments;
* expose seeds;
* preserve reproducibility;
* use explicit paths relative to the project root where practical;
* fail clearly rather than silently skipping data.

---

# 21. Reproducibility

Experiments must record:

* random seed;
* model version;
* dataset/split;
* preprocessing configuration;
* feature configuration;
* threshold;
* relevant command;
* resulting metrics.

Do not modify preprocessing between experiments without recording the change.

Do not compare two model results if their preprocessing differs without explicitly stating that fact.

---

# 22. Git Rules

Agents must frequently inspect:

```bash
git status
git diff
```

Before finishing a task, report:

```text
Files changed:
...

Why:
...

Tests run:
...

Results:
...

Known limitations:
...
```

Do not commit unless explicitly instructed.

Do not push unless explicitly instructed.

Do not create branches unless explicitly instructed.

---

# 23. Agent Handoff Rules

When an agent finishes:

1. Leave the working tree in a buildable/testable state.
2. Do not leave temporary debugging code enabled in production paths.
3. Clearly identify generated files.
4. Clearly identify commands used.
5. Clearly identify failures.
6. Clearly identify assumptions.
7. Do not hide incomplete work.

If work is incomplete, say:

```text
INCOMPLETE:
<what remains>
```

Do not present partial implementation as finished.

---

# 24. Temporary Experiments

Experimental code must be clearly separated from production code.

Temporary files should be removed after the experiment unless they provide lasting reproducibility value.

Before removing experimental code:

* preserve useful Python/reference scripts;
* restore production firmware behavior;
* verify the production build again.

Never leave a test harness silently changing production behavior.

---

# 25. What Agents Must NOT Do

Without explicit approval, agents must not:

* redesign the architecture;
* replace the feature extractor;
* change frame size/hop;
* change sample rate;
* change the microphone configuration;
* change the I2S sample interpretation;
* replace the FFT;
* change the wake word;
* retrain the model;
* create a new dataset;
* introduce proprietary SDKs;
* optimize away tests;
* weaken tolerances;
* delete another agent's changes;
* reset the repository;
* modify unrelated modules;
* claim benchmark success without measurements.

---

# 26. Definition of Done

A task is complete only when:

```text
[ ] Scope was inspected before editing
[ ] Existing changes were preserved
[ ] Only necessary files were modified
[ ] Code builds
[ ] Relevant tests run
[ ] Results were inspected
[ ] Performance was measured when relevant
[ ] No temporary production debugging remains
[ ] Known limitations are documented
[ ] Changed files are reported
```

For ESP32 tasks additionally:

```text
[ ] Tested on actual ESP32 where applicable
[ ] RAM measured
[ ] Flash measured
[ ] Latency measured
[ ] CPU measured where applicable
[ ] Continuous operation checked where applicable
```

---

# 27. Golden Rule

When uncertain:

**Do less, inspect more, measure first, and never silently change an established contract.**

The goal is not to produce the most code.

The goal is to produce a **measurably correct, low-latency, low-memory, continuously running wake-word system on the actual ESP32 hardware.**
