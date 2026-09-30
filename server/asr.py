"""Open-source local ASR service using faster-whisper.

Per SIH 26172 and AGENTS.md rules:
- Completely open-source (no proprietary APIs, no commercial cloud dependencies).
- Runs locally on the server with CPU int8 quantization.
- Optimized for ultra-low latency on 16 kHz mono PCM16 audio streams.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Tuple

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger("ASR")


class ASRService:
    """Manages the lifecycle and inference for the local Whisper model."""

    def __init__(
        self,
        model_size: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size or os.environ.get("ASR_MODEL_SIZE", "base.en")
        self.device = device
        self.compute_type = compute_type
        self._model: WhisperModel | None = None

    def load(self) -> None:
        """Load the model into memory and run a short warm-up pass."""
        logger.info(
            "Loading ASR model '%s' (device=%s, compute_type=%s)...",
            self.model_size,
            self.device,
            self.compute_type,
        )
        t0 = time.perf_counter()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        load_time = (time.perf_counter() - t0) * 1000.0
        logger.info("ASR model '%s' loaded in %.1f ms", self.model_size, load_time)

        # Warm-up inference on 0.5s dummy silence to JIT-compile execution graph
        try:
            dummy_pcm = np.zeros(8000, dtype=np.float32)
            list(self._model.transcribe(dummy_pcm, beam_size=1, language="en")[0])
            logger.info("ASR warm-up pass completed successfully")
        except Exception as e:
            logger.warning("ASR warm-up pass warning: %s", e)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe_pcm16(
        self, raw_bytes: bytes, sample_rate: int = 16000
    ) -> Tuple[str, float]:
        """Transcribe raw PCM16LE audio bytes in memory.

        Args:
            raw_bytes: Raw signed 16-bit little-endian mono PCM bytes.
            sample_rate: Audio sampling frequency (canonical: 16000 Hz).

        Returns:
            Tuple of (transcribed_text, inference_duration_ms).
        """
        if self._model is None:
            raise RuntimeError("ASR model is not loaded. Call load() first.")

        if not raw_bytes or len(raw_bytes) < 320:  # < 10 ms
            return "", 0.0

        # Convert PCM16 bytes directly to normalized float32 array [-1.0, 1.0]
        audio_array = (
            np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )

        t0 = time.perf_counter()
        # Greedy decoding (beam_size=1) for minimal inference latency on edge commands
        segments, _ = self._model.transcribe(
            audio_array,
            beam_size=1,
            language="en",
            vad_filter=True,
        )
        transcript = " ".join(s.text.strip() for s in segments).strip()
        inference_ms = (time.perf_counter() - t0) * 1000.0

        return transcript, inference_ms
