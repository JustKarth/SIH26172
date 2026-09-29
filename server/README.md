# ESP32 Audio Receiver & Local ASR Service

This service accepts mono 16 kHz signed PCM16 little-endian audio over WebSocket,
runs local open-source speech-to-text inference using `faster-whisper`, delivers
transcribed text and latency metrics back to the edge device, and broadcasts live
telemetry to a real-time web monitor.

## Features

- **Open-source ASR engine:** Local `faster-whisper` (`base.en` with INT8 CPU quantization).
- **Zero cold-start delay:** Model preloaded and warmed up in memory during FastAPI startup.
- **In-memory inference:** Audio buffers converted directly to normalized float32 arrays without disk I/O bottlenecks.
- **Web monitor dashboard:** Visual wake indicator and live transcription card at `http://<server>:8000/`.
- **Latency instrumentation:** Returns inference duration and total server turnaround times.
- **Mock client tool:** Test audio streaming and ASR from PC without flashing hardware.

## Run locally

From the repository root, install dependencies and start the service:

```powershell
py -m pip install -r server/requirements.txt
py -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

- **Dashboard:** `http://<server-host>:8000/`
- **Health check:** `http://<server-host>:8000/health`
- **Edge audio endpoint:** `ws://<server-host>:8000/ws/audio`
- **UI event broadcast:** `ws://<server-host>:8000/ws/ui`

On the ESP32's Wi-Fi network, use the computer's LAN IP address for `<server-host>` (not `localhost`). Allow inbound TCP port 8000 through your firewall.

Completed WAV files are written to `recordings/` under the server's working directory. Set `AUDIO_RECORDINGS_DIR` to choose another location.

## Testing with the Mock Client

You can simulate an ESP32 streaming audio to the server using the included test client:

```powershell
# Stream synthetic tone
py -m server.mock_client --duration 2.0

# Stream an actual speech recording
py -m server.mock_client --file path/to/speech.wav
```

## WebSocket Protocol

1. **Start handshake (JSON text):**
   ```json
   {
     "type": "start",
     "session_id": "optional-client-id",
     "audio": {"sample_rate": 16000, "channels": 1, "format": "pcm_s16le"}
   }
   ```
2. **Start acknowledgement (server JSON):**
   ```json
   {"type": "start_ack", "session_id": "01J..."}
   ```
3. **Audio streaming (binary WebSocket frames):**
   Send raw signed 16-bit little-endian PCM bytes in real-time chunks (e.g. 50–100 ms).
4. **End signal (JSON text):**
   ```json
   {"type": "end", "reason": "vad_silence"}
   ```
5. **ASR Result (server JSON):**
   ```json
   {
     "type": "result",
     "session_id": "01J...",
     "status": "complete",
     "text": "turn on the living room lights",
     "audio_file": "01J....wav",
     "bytes": 76932,
     "duration_seconds": 2.40,
     "metrics": {
       "asr_inference_ms": 115.4,
       "total_server_time_ms": 118.2
     }
   }
   ```
