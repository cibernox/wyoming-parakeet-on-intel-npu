"""Wyoming protocol event handler.

Adapted with thanks from https://github.com/tboby/wyoming-onnx-asr (MIT).
Buffers AudioChunk events into a temp WAV; on AudioStop, runs ASR exactly once.
"""
import asyncio
import logging
import os
import tempfile
import wave
from typing import Optional

import numpy as np
import soundfile as sf
from onnx_asr.adapters import AsrAdapter
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


class ParakeetEventHandler(AsyncEventHandler):
    """Wyoming event handler for Parakeet TDT 0.6B v3 multilingual."""

    def __init__(
        self,
        wyoming_info: Info,
        model: AsrAdapter,
        model_lock: asyncio.Lock,
        *args,
        default_language: str = "en",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.default_language = default_language
        self.request_language: Optional[str] = None
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped. Transcribing.")
            assert self._wav_file is not None
            self._wav_file.close()
            self._wav_file = None

            waveform, sample_rate = sf.read(self._wav_path, dtype="float32")
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)

            lang = self.request_language or self.default_language
            _LOGGER.info("Language requested: %s", lang)

            async with self.model_lock:
                try:
                    text = self.model.recognize(
                        waveform, language=lang, sample_rate=sample_rate
                    )
                except Exception as e:
                    _LOGGER.error("Recognition failed: %s", e)
                    await self.write_event(
                        Transcript(text=f"ERROR: {e}").event()
                    )
                    return False

            _LOGGER.info("%s: %s", lang, text)
            await self.write_event(Transcript(text=text).event())
            self.request_language = None
            return False

        if Transcribe.is_type(event.type):
            self.request_language = Transcribe.from_event(event).language
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        return True
