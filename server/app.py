"""Small WebSocket receiver for ESP32 PCM16 audio streams."""

import os
import uuid
import wave
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


app = FastAPI(title="ESP32 Audio Receiver", version="1.0.0")
RECORDINGS_DIR = Path(os.environ.get("AUDIO_RECORDINGS_DIR", "recordings"))
MAX_AUDIO_SECONDS = 120
SAMPLE_RATE = 16_000


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check for the server process."""
    return {"status": "ok"}


@app.websocket("/ws/audio")
async def receive_audio(websocket: WebSocket) -> None:
    """Receive start JSON, binary PCM16 chunks, then an end JSON message."""
    await websocket.accept()
    writer: wave.Wave_write | None = None
    session_id: str | None = None
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
            return

        # Generate the filename on the server; never use client text as a path.
        session_id = uuid.uuid4().hex
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = RECORDINGS_DIR / f"{session_id}.wav"
        writer = wave.open(str(wav_path), "wb")
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        await websocket.send_json({"type": "start_ack", "session_id": session_id})

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
                writer.writeframesraw(payload)
                continue

            text = message.get("text")
            if text is None:
                await websocket.close(code=1003, reason="Expected binary audio or end JSON")
                return

            import json

            try:
                control = json.loads(text)
            except json.JSONDecodeError:
                await websocket.close(code=1007, reason="Invalid JSON control message")
                return
            if not isinstance(control, dict) or control.get("type") != "end":
                await websocket.close(code=1008, reason="Expected end message")
                return

            writer.close()
            writer = None
            completed = True
            await websocket.send_json(
                {
                    "type": "result",
                    "session_id": session_id,
                    "status": "received",
                    "audio_file": wav_path.name,
                    "bytes": total_bytes,
                    "duration_seconds": total_bytes / (SAMPLE_RATE * 2),
                }
            )
            await websocket.close()
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
