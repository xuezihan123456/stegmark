"""Echo-hiding audio watermark for 16-bit mono PCM WAV.

Encodes one bit per audio segment by mixing in an echo with delay ``delta_d0``
(bit=0) or ``delta_d1`` (bit=1). Extraction inspects autocorrelation peaks
at the two candidate lags and reports the larger one.

This is a baseline implementation; real-world deployment should use spread
spectrum or psychoacoustic-shaped echo for robustness against MP3/AAC
re-encoding.
"""
from __future__ import annotations

import struct
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stegmark.exceptions import InvalidInputError, MessageTooLongError


@dataclass(frozen=True)
class AudioEmbedResult:
    output_path: Path
    capacity_used_bits: int
    segment_samples: int


class AudioWatermarkEngine:
    """Echo-hiding watermark for mono PCM WAV files."""

    name = "audio"
    segment_samples: int = 4096
    echo_amplitude: float = 0.5

    def capacity_bits(self, wav_path: Path) -> int:
        n_samples = self._wav_sample_count(wav_path)
        return n_samples // self.segment_samples

    def embed(
        self,
        wav_path: Path,
        payload_bits: Iterable[int],
        out_path: Path,
        delta_d0: int = 100,
        delta_d1: int = 150,
    ) -> AudioEmbedResult:
        bits = [int(b) & 1 for b in payload_bits]
        samples, params = self._read_wav(wav_path)
        capacity = samples.size // self.segment_samples
        if len(bits) > capacity:
            raise MessageTooLongError(
                f"audio capacity is {capacity} bits, payload is {len(bits)}",
                hint="Use a longer audio file or shorter payload.",
            )

        max_delay = max(delta_d0, delta_d1)
        if max_delay >= self.segment_samples:
            raise InvalidInputError(
                "echo delay must be smaller than segment length",
                hint=f"segment_samples is {self.segment_samples}",
            )

        watermarked = samples.astype(np.float32, copy=True)
        for i, bit in enumerate(bits):
            start = i * self.segment_samples
            end = start + self.segment_samples
            delay = delta_d1 if bit == 1 else delta_d0
            segment = watermarked[start:end]
            echo = np.zeros_like(segment)
            echo[delay:] = segment[:-delay] * self.echo_amplitude
            watermarked[start:end] = segment + echo

        clipped = np.clip(watermarked, -32768, 32767).astype(np.int16)
        self._write_wav(out_path, clipped, params)
        return AudioEmbedResult(
            output_path=out_path,
            capacity_used_bits=len(bits),
            segment_samples=self.segment_samples,
        )

    def extract(
        self,
        wav_path: Path,
        num_bits: int,
        delta_d0: int = 100,
        delta_d1: int = 150,
    ) -> list[int]:
        samples, _ = self._read_wav(wav_path)
        bits: list[int] = []
        for i in range(num_bits):
            start = i * self.segment_samples
            end = start + self.segment_samples
            if end > samples.size:
                break
            segment = samples[start:end].astype(np.float32)
            corr0 = self._correlation_at(segment, delta_d0)
            corr1 = self._correlation_at(segment, delta_d1)
            bits.append(1 if corr1 > corr0 else 0)
        return bits

    @staticmethod
    def _correlation_at(segment: np.ndarray, delay: int) -> float:
        if delay <= 0 or delay >= segment.size:
            return 0.0
        # Use absolute value of zero-mean correlation to be sign-robust.
        a = segment[:-delay] - segment[:-delay].mean()
        b = segment[delay:] - segment[delay:].mean()
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-9)
        return float(abs(np.sum(a * b)) / denom)

    @staticmethod
    def _read_wav(path: Path) -> tuple[np.ndarray, wave._wave_params]:
        with wave.open(str(path), "rb") as fh:
            if fh.getnchannels() != 1:
                raise InvalidInputError("audio engine requires mono WAV", hint="Convert to mono first.")
            if fh.getsampwidth() != 2:
                raise InvalidInputError("audio engine requires 16-bit PCM", hint="Re-export as 16-bit PCM WAV.")
            params = fh.getparams()
            frames = fh.readframes(fh.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
        return samples, params

    @staticmethod
    def _write_wav(path: Path, samples: np.ndarray, params: wave._wave_params) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(params.nchannels)
            fh.setsampwidth(params.sampwidth)
            fh.setframerate(params.framerate)
            fh.writeframes(samples.tobytes())

    @staticmethod
    def _wav_sample_count(path: Path) -> int:
        with wave.open(str(path), "rb") as fh:
            return int(fh.getnframes())

    @staticmethod
    def synth_sine(out_path: Path, duration_seconds: float = 1.0, rate: int = 16000, freq: float = 440.0) -> Path:
        t = np.arange(int(duration_seconds * rate), dtype=np.float32)
        wave_data = (0.5 * 32767 * np.sin(2 * np.pi * freq * t / rate)).astype(np.int16)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(rate)
            fh.writeframes(wave_data.tobytes())
        return out_path

    @staticmethod
    def synth_noise(out_path: Path, duration_seconds: float = 1.0, rate: int = 16000, seed: int = 0) -> Path:
        rng = np.random.default_rng(seed)
        wave_data = (0.3 * 32767 * rng.standard_normal(int(duration_seconds * rate))).astype(np.int16)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(rate)
            fh.writeframes(wave_data.tobytes())
        return out_path


# Silence unused-import warnings on platforms where wave types differ.
_ = struct

__all__ = ["AudioEmbedResult", "AudioWatermarkEngine"]
