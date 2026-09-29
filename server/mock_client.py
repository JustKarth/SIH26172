"""Mock client simulating an ESP32 streaming PCM16 audio to the server.

Use this tool to test the WebSocket receiver and local ASR pipeline without
flashing the physical ESP32 hardware.

Usage:
    # 1. Speak custom text via built-in TTS and stream to ASR:
    py -m server.mock_client --text "turn on the living room lights"

    # 2. Stream an existing WAV file:
    py -m server.mock_client --file my_recording.wav

    # 3. Stream a synthetic 440 Hz test tone (produces silence in ASR):
    py -m server.mock_client --duration 2.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import wave

import numpy as np
import websockets


def generate_synthetic_audio(duration_s: float = 2.0, freq_hz: float = 440.0) -> bytes:
    """Generate canonical 16 kHz mono signed 16-bit PCM synthetic tone."""
    sample_rate = 16000
    n_samples = int(duration_s * sample_rate)
    samples = [
        int(math.sin(2.0 * math.pi * freq_hz * i / sample_rate) * 12000)
        for i in range(n_samples)
    ]
    return struct.pack(f"<{n_samples}h", *samples)


def load_wav_pcm16(wav_path: str) -> bytes:
    """Load and resample an arbitrary WAV file to canonical 16 kHz mono PCM16."""
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"File '{wav_path}' does not exist. Please provide an existing .wav file, "
            f"or use --text \"your command\" to generate speech automatically."
        )

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(data, dtype=np.int16)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth * 8}-bit (must be 16-bit)")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)[:, 0]

    if framerate != 16000:
        target_len = int(len(samples) * 16000 / framerate)
        samples = np.interp(
            np.linspace(0, len(samples), target_len, endpoint=False),
            np.arange(len(samples)),
            samples,
        ).astype(np.int16)

    return samples.tobytes()


def synthesize_speech(text: str) -> bytes:
    """Synthesize speech on Windows using built-in SAPI for quick testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    safe_text = text.replace("'", "''")
    ps_cmd = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{tmp_path}'); "
        f"$s.Speak('{safe_text}'); "
        f"$s.Dispose()"
    )
    try:
        proc = subprocess.run(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        return load_wav_pcm16(str(tmp_path))
    except Exception as e:
        print(f"Warning: Local TTS generation failed: {e}. Falling back to synthetic tone.")
        return generate_synthetic_audio(2.0)
    finally:
        tmp_path.unlink(missing_ok=True)


async def stream_audio(server_url: str, audio_bytes: bytes, chunk_ms: int = 100) -> None:
    """Stream PCM16 audio in real-time chunks to the server over WebSocket."""
    chunk_samples = int(16000 * (chunk_ms / 1000.0))
    chunk_bytes = chunk_samples * 2  # 2 bytes per 16-bit sample

    print(f"Connecting to {server_url}...")
    try:
        async with websockets.connect(server_url) as ws:
            # Handshake: start
            start_msg = {
                "type": "start",
                "audio": {"sample_rate": 16000, "channels": 1, "format": "pcm_s16le"},
            }
            await ws.send(json.dumps(start_msg))
            ack_raw = await ws.recv()
            ack = json.loads(ack_raw)
            session_id = ack.get("session_id", "unknown")
            print(f"Session started: ID={session_id}")

            total_chunks = (len(audio_bytes) + chunk_bytes - 1) // chunk_bytes
            duration_s = len(audio_bytes) / 32000.0
            print(f"Streaming {len(audio_bytes)} bytes ({duration_s:.2f} s) in {total_chunks} chunks ({chunk_ms} ms/chunk)...")

            # Stream chunks at real-time rate
            for i in range(0, len(audio_bytes), chunk_bytes):
                chunk = audio_bytes[i : i + chunk_bytes]
                await ws.send(chunk)
                await asyncio.sleep(chunk_ms / 1000.0)

            # End message
            end_msg = {"type": "end", "reason": "vad_silence"}
            await ws.send(json.dumps(end_msg))
            print("Stream ended. Waiting for server ASR response...")

            result_raw = await ws.recv()
            result = json.loads(result_raw)
            print("\n=== Server Response ===")
            print(f"Session ID : {result.get('session_id')}")
            print(f"Status     : {result.get('status')}")
            transcript = result.get("text", "")
            print(f"Transcript : '{transcript}'" if transcript else "Transcript : (none / silence)")
            print(f"Audio File : {result.get('audio_file')} ({result.get('duration_seconds', 0):.2f} s)")
            metrics = result.get("metrics", {})
            print(f"ASR Latency: {metrics.get('asr_inference_ms', 0):.1f} ms")
            print(f"Total Time : {metrics.get('total_server_time_ms', 0):.1f} ms")

    except ConnectionRefusedError:
        print(f"Error: Connection refused at {server_url}. Make sure the server is running.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ESP32 Mock Streaming Client")
    parser.add_argument("--server", default="ws://localhost:8000/ws/audio", help="WebSocket URL")
    parser.add_argument("--text", "--say", help="Text to speak and stream (synthesized locally)")
    parser.add_argument("--file", help="Path to existing WAV file to stream")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds if synthetic tone")
    args = parser.parse_args()

    if args.text:
        print(f"Synthesizing test speech: \"{args.text}\"...")
        audio = synthesize_speech(args.text)
    elif args.file:
        try:
            audio = load_wav_pcm16(args.file)
        except (FileNotFoundError, ValueError) as err:
            print(f"Error: {err}")
            sys.exit(1)
    else:
        print(f"No --text or --file specified. Generating {args.duration}s synthetic tone (note: pure tones transcribe as silence in ASR).")
        audio = generate_synthetic_audio(args.duration)

    asyncio.run(stream_audio(args.server, audio))


if __name__ == "__main__":
    main()
