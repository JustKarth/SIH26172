# SIH26172 — Agent Engineering Rules

## 0. Mission

This repository implements SIH Problem Statement 26172:

"Low Latency and Efficient Voice Activator for Edge Devices."

The system is an embedded keyword-spotting device using an ESP32 and INMP441 microphone, followed by Wi-Fi audio streaming to a remote ASR server.

The existing audio capture and feature-extraction pipeline is a working baseline.

DO NOT rewrite working subsystems without evidence that they are incorrect.

---

# 1. Architectural ownership

The human developer owns architectural decisions.

The agent is responsible for implementation, testing, debugging, and proposing improvements.

The agent MUST NOT:
- silently redesign architecture
- replace an existing subsystem merely because another design is more familiar
- introduce a framework/library without justification
- merge multiple responsibilities into one module
- create duplicate implementations of an existing responsibility
- move files merely for stylistic reasons
- change public APIs unnecessarily

If an architectural change appears necessary:

1. Stop implementation.
2. Explain the problem.
3. Propose the change.
4. Explain alternatives.
5. Wait for approval.

---

# 2. Development process

Every feature MUST follow:

REQUIREMENTS
    ↓
DESIGN
    ↓
INTERFACES
    ↓
IMPLEMENTATION
    ↓
UNIT/COMPONENT TEST
    ↓
INTEGRATION TEST
    ↓
DOCUMENTATION

Do not jump directly from requirements to implementation.

---

# 3. Before editing

Before modifying anything:

1. Read README.md.
2. Read AGENTS.md.
3. Inspect the relevant source files.
4. Identify the module responsible for the requested behavior.
5. Identify dependencies and consumers of that module.
6. Inspect git status.
7. Explain the intended change.

Do not edit files during the analysis phase.

---

# 4. Modular architecture

Each module must have ONE primary responsibility.

A module should answer:

"What single reason would cause this module to change?"

If a module has multiple unrelated reasons to change, split it.

Prefer:

module A → module B → module C

over:

everything → giant manager module

Avoid "god" files such as:
- system.c
- manager.c
- app.c
- utils.c

unless their responsibilities are genuinely narrow.

---

# 5. Interfaces first

Before implementing a new subsystem, define its interface.

For C modules:

module.h
    ↓
public types
public functions
ownership/lifetime rules
error semantics

module.c
    ↓
implementation

The header should expose the minimum required API.

Do not expose internal buffers, state machines, implementation structures, or synchronization primitives unless necessary.

---

# 6. Dependency direction

Dependencies must point toward lower-level infrastructure.

Current intended direction:

Application
    ↓
Wake Controller
    ↓
KWS
    ↓
Feature Extraction
    ↓
Audio Frames
    ↓
Audio Pipeline
    ↓
ESP-IDF / hardware

Networking should be a separate subsystem.

ASR server should not be coupled to ESP32 audio capture internals.

The audio pipeline must not know about KWS.

The feature extractor must not know about Wi-Fi.

The KWS module must not know about WebSocket.

The networking module must not manipulate the audio ring buffer directly.

---

# 7. Data ownership

Every buffer must have an explicit owner.

For every non-trivial buffer document:

- who allocates it
- who writes it
- who reads it
- when it becomes invalid
- whether it is copied or borrowed
- whether it is thread-safe

Avoid unnecessary copies.

But NEVER eliminate a copy merely for performance without measuring whether it matters.

---

# 8. Concurrency

Every FreeRTOS task must have:

- one clearly defined responsibility
- documented input/output
- explicit synchronization mechanism
- documented priority
- documented stack requirement
- documented CPU affinity if relevant

Do not introduce polling loops when a notification/queue/semaphore is appropriate.

Do not use busy waiting.

Do not disable watchdogs to hide timing problems.

---

# 9. Memory

ESP32 RAM is a hard constraint.

For every substantial allocation identify:

FLASH / STATIC RAM / HEAP / DMA / STACK / TENSOR ARENA

Do not allocate large buffers repeatedly.

Prefer static allocation where appropriate.

Do not move buffers to heap merely because it is convenient.

The final system target is:

RAM < 256 KB
idle CPU < 10%

Measurements must be real measurements.

Never invent benchmark values.

---

# 10. Error handling

Do not ignore return values from:

- ESP-IDF APIs
- memory allocation
- I2S
- networking
- filesystem
- model loading
- queue/semaphore operations

Use ESP_ERROR_CHECK only when failure should be fatal during initialization.

Runtime failures should normally be handled explicitly.

---

# 11. Logging

Logs should describe system behavior, not spam raw data.

Use appropriate ESP_LOG levels.

Do not print entire feature matrices, PCM buffers, or repeated per-frame information in production paths.

Debug instrumentation should be easy to disable.

---

# 12. Testing

Every subsystem needs an acceptance test.

A successful compilation is NOT an acceptance test.

Examples:

Audio:
- continuous 16 kHz capture
- no unexplained overruns

Framing:
- 400 sample frame
- 160 sample hop

Features:
- numerical parity with Python reference

KWS:
- held-out speaker evaluation
- false activation measurement

Inference:
- latency
- RAM
- model size

Networking:
- valid session state transitions
- PCM integrity

End-to-end:
- wake detection
- audio streaming
- ASR result
- latency measurement

---

# 13. Change size

Prefer small, reviewable changes.

A task should normally modify only the files necessary for that task.

Do not perform unrelated refactoring while implementing a feature.

If unrelated problems are discovered:
- document them
- do not automatically fix them unless they block the current task

---

# 14. Build discipline

After modifying firmware:

    idf.py build

Do not claim the implementation works until the build succeeds.

If hardware testing is possible, distinguish:

BUILD VERIFIED

from:

HARDWARE VERIFIED

Do not confuse the two.

---

# 15. Git discipline

Before substantial changes:

    git status
    git diff

Never overwrite the working baseline without a recoverable git state.

Prefer logically grouped commits.

Do not rewrite history unless explicitly requested.

---

# 16. Definition of done

A feature is DONE only when:

[ ] implementation exists
[ ] interface is documented
[ ] build passes
[ ] relevant test passes
[ ] resource usage is understood
[ ] error cases are considered
[ ] README/documentation is updated
[ ] git diff contains no unrelated changes

---

# 17. When uncertain

Do NOT guess silently.

Classify uncertainty as:

A. implementation detail
   → decide and proceed

B. requirement ambiguity
   → state assumption

C. architectural decision
   → stop and ask

D. safety/correctness issue
   → stop and explain

The agent should be autonomous about implementation,
but conservative about architecture.