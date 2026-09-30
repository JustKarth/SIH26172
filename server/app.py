"""FastAPI WebSocket receiver & ASR service for ESP32 PCM16 audio streams.

Milestones implemented:
-----------------------
- Step 1: Open-source local ASR (faster-whisper) with lifespan pre-loading
          and in-memory PCM16 conversion (zero disk I/O latency).
- Step 2: Protocol extensions for transcribed text dispatching to both
          the edge client (/ws/audio) and the browser monitor (/ws/ui),
          with full inference latency metrics.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import time
import uuid
import wave
from pathlib import Path
from typing import AsyncGenerator, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from server.asr import ASRService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SERVER")

# ---------------------------------------------------------------------------
# Configuration & Globals
# ---------------------------------------------------------------------------

RECORDINGS_DIR    = Path(os.environ.get("AUDIO_RECORDINGS_DIR", "recordings"))
MAX_AUDIO_SECONDS = 120
SAMPLE_RATE       = 16_000

_HERE = Path(__file__).parent

# Local ASR service instance (base.en, int8 quantization on CPU)
asr_service = ASRService()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context: load ASR once on startup to avoid cold starts."""
    logger.info("Initializing server & pre-loading ASR model...")
    # Run model load in a thread to keep async startup smooth
    await asyncio.to_thread(asr_service.load)
    logger.info("ASR ready for incoming requests.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(title="ESP32 Audio Receiver & ASR", version="2.1.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# UI client registry & broadcast
# ---------------------------------------------------------------------------

_ui_clients: Set[WebSocket] = set()


async def _broadcast_ui(data: dict) -> None:
    """Send *data* as JSON to every connected dashboard browser tab."""
    dead: Set[WebSocket] = set()
    for client in list(_ui_clients):
        try:
            await client.send_json(data)
        except Exception:
            dead.add(client)
    _ui_clients.difference_update(dead)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def dashboard() -> FileResponse:
    """Serve the single-file wake-word dashboard."""
    return FileResponse(_HERE / "dashboard.html", media_type="text/html")


@app.get("/health")
async def health() -> dict[str, object]:
    """Health check for server process and ASR engine."""
    return {
        "status": "ok",
        "asr_loaded": asr_service.is_loaded,
        "asr_model": asr_service.model_size,
    }


@app.websocket("/ws/ui")
async def ui_endpoint(websocket: WebSocket) -> None:
    """Browser dashboard connection — receives state events and pings."""
    await websocket.accept()
    _ui_clients.add(websocket)
    # Send initial idle state so the dashboard renders correctly on connect.
    await websocket.send_json({"event": "idle", "active": False})

    async def _drain() -> None:
        try:
            while True:
                await websocket.receive_text()
        except Exception:
            pass

    drain_task = asyncio.create_task(_drain())

    try:
        while True:
            await asyncio.sleep(15)
            try:
                await websocket.send_json({"event": "ping"})
            except Exception:
                break
    except asyncio.CancelledError:
        pass
    finally:
        drain_task.cancel()
        _ui_clients.discard(websocket)


@app.websocket("/ws/audio")
async def receive_audio(websocket: WebSocket) -> None:
    """Receive start JSON, binary PCM16 chunks, then end JSON message.

    Performs in-memory ASR transcription and returns text + latency metrics.
    """
    await websocket.accept()
    writer: wave.Wave_write | None = None
    session_id: str | None = None
    audio_buffer = bytearray()
    total_bytes = 0
    completed = False

    try:
        start = await websocket.receive_json()
        if not isinstance(start, dict) or start.get("type") != "start":
            await websocket.close(code=1008, reason="Expected start message")
            return

        audio = start.get("audio")
        if not isinstance(audio, dict) or (
            audio.get("sample_rate") != SAMPLE_RATE
            or audio.get("channels") != 1
            or audio.get("format") != "pcm_s16le"
        ):
            await websocket.close(code=1008, reason="Unsupported audio format")
            await _broadcast_ui({
                "event": "error",
                "active": False,
                "reason": "Unsupported audio format",
            })
            return

        session_id = uuid.uuid4().hex
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = RECORDINGS_DIR / f"{session_id}.wav"
        writer = wave.open(str(wav_path), "wb")
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)

        await websocket.send_json({"type": "start_ack", "session_id": session_id})

        # ── Broadcast wake-detected event to UI clients ──────────────────
        await _broadcast_ui({
            "event": "wake_detected",
            "active": True,
            "session_id": session_id,
        })

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))

            payload = message.get("bytes")
            if payload is not None:
                if not payload or len(payload) % 2:
                    await websocket.close(code=1003, reason="PCM16 chunks must contain whole samples")
                    return
                total_bytes += len(payload)
                if total_bytes > MAX_AUDIO_SECONDS * SAMPLE_RATE * 2:
                    await websocket.close(code=1009, reason="Audio session exceeds 120 seconds")
                    return

                # Accumulate in memory for zero-disk-delay ASR
                audio_buffer.extend(payload)
                # Write to disk for permanent artifact/verification
                writer.writeframesraw(payload)
                continue

            text = message.get("text")
            if text is None:
                await websocket.close(code=1003, reason="Expected binary audio or end JSON")
                return

            try:
                control = json.loads(text)
            except json.JSONDecodeError:
                await websocket.close(code=1007, reason="Invalid JSON control message")
                return

            if not isinstance(control, dict) or control.get("type") != "end":
                await websocket.close(code=1008, reason="Expected end message")
                return

            # Finish disk file write
            writer.close()
            writer = None
            completed = True
            duration_s = total_bytes / (SAMPLE_RATE * 2)

            # ── Run ASR in worker thread ──────────────────────────────────
            t_asr_start = time.perf_counter()
            transcript = ""
            asr_ms = 0.0

            if asr_service.is_loaded:
                try:
                    transcript, asr_ms = await asyncio.to_thread(
                        asr_service.transcribe_pcm16, bytes(audio_buffer), SAMPLE_RATE
                    )
                except Exception as e:
                    logger.error("ASR transcription error for session %s: %s", session_id, e)
                    transcript = ""

            total_server_ms = (time.perf_counter() - t_asr_start) * 1000.0
            logger.info(
                "Session %s transcribed in %.1f ms: '%s'",
                session_id[:8],
                asr_ms,
                transcript,
            )

            # ── Send result back to edge device ───────────────────────────
            await websocket.send_json({
                "type": "result",
                "session_id": session_id,
                "status": "complete",
                "text": transcript,
                "audio_file": wav_path.name,
                "bytes": total_bytes,
                "duration_seconds": duration_s,
                "metrics": {
                    "asr_inference_ms": round(asr_ms, 1),
                    "total_server_time_ms": round(total_server_ms, 1),
                },
            })
            await websocket.close()

            # ── Broadcast transcription & metrics to UI clients ────────────
            await _broadcast_ui({
                "event": "stream_ended",
                "active": False,
                "session_id": session_id,
                "text": transcript,
                "bytes": total_bytes,
                "duration_seconds": duration_s,
                "latency": {
                    "asr_inference_ms": round(asr_ms, 1),
                    "total_server_time_ms": round(total_server_ms, 1),
                },
            })
            return

    except WebSocketDisconnect:
        pass
    finally:
        if writer is not None:
            writer.close()
        # Keep only complete recordings. An interrupted transfer is discarded.
        if session_id is not None and not completed:
            partial_path = RECORDINGS_DIR / f"{session_id}.wav"
            partial_path.unlink(missing_ok=True)
            await _broadcast_ui({
                "event": "error",
                "active": False,
                "reason": "ESP32 disconnected before stream completed",
            })
