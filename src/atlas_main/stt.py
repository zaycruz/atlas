"""Speech-to-text helpers: microphone capture, VAD, and transcription."""
from __future__ import annotations

import queue
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency
    sd = None

try:  # pragma: no cover - optional dependency
    import webrtcvad
except Exception:  # pragma: no cover - optional dependency
    webrtcvad = None

try:  # pragma: no cover - optional dependency
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional dependency
    WhisperModel = None


@dataclass
class Segment:
    text: str
    start: float
    end: float


class Microphone:
    def __init__(self, *, sample_rate: int = 16_000, device: Optional[int] = None, block_size: int = 1024) -> None:
        if sd is None:  # pragma: no cover - optional dependency
            raise RuntimeError("sounddevice is required for microphone capture")
        self.sample_rate = sample_rate
        self.device = device
        self.block_size = block_size
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: Optional[sd.InputStream] = None

    def start(self) -> None:
        if self._stream is not None:
            return

        def _callback(indata, frames, time_info, status):  # pragma: no cover - audio callback
            if status:
                return
            self._queue.put(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=self.device,
            channels=1,
            dtype="int16",
            blocksize=self.block_size,
            callback=_callback,
        )
        self._stream.start()

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        while not self._queue.empty():
            self._queue.get()


class VadSegmenter:
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16_000, frame_ms: int = 30, silence_ms: int = 600) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.silence_frames = max(1, silence_ms // frame_ms)
        if webrtcvad is None:  # pragma: no cover - optional dependency
            raise RuntimeError("webrtcvad is required for VAD segmentation")
        self._vad = webrtcvad.Vad(aggressiveness)
        self._buffer: bytearray = bytearray()
        self._speech_frames: List[bytes] = []
        self._trailing_silence = 0
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2

    def feed(self, chunk: np.ndarray) -> List[bytes]:
        self._buffer.extend(chunk.tobytes())
        segments: List[bytes] = []
        while len(self._buffer) >= self.frame_bytes:
            frame = bytes(self._buffer[: self.frame_bytes])
            del self._buffer[: self.frame_bytes]
            is_speech = self._vad.is_speech(frame, self.sample_rate)
            if is_speech:
                self._speech_frames.append(frame)
                self._trailing_silence = 0
            else:
                if self._speech_frames:
                    self._trailing_silence += 1
                    if self._trailing_silence >= self.silence_frames:
                        segments.append(b"".join(self._speech_frames))
                        self._speech_frames = []
                        self._trailing_silence = 0
        return segments

    def flush(self) -> List[bytes]:
        if not self._speech_frames:
            return []
        segment = b"".join(self._speech_frames)
        self._speech_frames = []
        self._trailing_silence = 0
        return [segment]


class WhisperTranscriber:
    def __init__(self, model_name: str = "medium.en", device: str = "cpu") -> None:
        if WhisperModel is None:  # pragma: no cover - optional dependency
            raise RuntimeError("faster-whisper is required for transcription")
        self.model = WhisperModel(model_name, device=device)

    def transcribe(self, audio_bytes: bytes) -> dict:
        if not audio_bytes:
            return {"text": "", "segments": []}
        import numpy as np

        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments_info = []
        text_parts = []
        for segment in self.model.transcribe(audio_array, language="en")[0]:  # pragma: no cover - heavy dependency
            text_parts.append(segment.text.strip())
            segments_info.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
            )
        return {"text": " ".join(text_parts).strip(), "segments": segments_info}


__all__ = ["Microphone", "VadSegmenter", "WhisperTranscriber", "Segment"]
