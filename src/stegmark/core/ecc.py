"""Forward error correction (FEC) for watermark bit-streams.

Provides four protection levels:

* ``NONE``      — raw bits, identity.
* ``LIGHT``     — Hamming(7,4): corrects 1-bit errors per code-word.
* ``MEDIUM``    — Repetition x3 + Hamming(7,4): tolerates ~10% BER.
* ``HEAVY``     — Repetition x5 + Hamming(7,4): tolerates ~20% BER.

Pure-numpy; no external FEC library.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class ECCLevel(IntEnum):
    NONE = 0
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3


@dataclass(frozen=True)
class ECCStats:
    encoded_bits: int
    corrected_errors: int


class RepetitionCode:
    """Repeat each bit ``factor`` times; decode by majority vote."""

    def __init__(self, factor: int):
        if factor < 1 or factor % 2 == 0:
            raise ValueError("repetition factor must be a positive odd integer")
        self.factor = factor

    def encode(self, bits: np.ndarray) -> np.ndarray:
        return np.repeat(bits.astype(np.uint8, copy=False), self.factor)

    def decode(self, bits: np.ndarray, original_length: int) -> tuple[np.ndarray, int]:
        if bits.size < original_length * self.factor:
            raise ValueError("encoded length too short for repetition decode")
        reshaped = bits[: original_length * self.factor].reshape(original_length, self.factor)
        votes = reshaped.sum(axis=1)
        decoded = (votes >= (self.factor // 2 + 1)).astype(np.uint8)
        # Count corrected = positions where the majority differs from at least one input.
        disagreement = np.abs(reshaped - decoded[:, None]).sum()
        return decoded, int(disagreement)


# Hamming(7,4) generator and parity matrices.
_HAMMING_G = np.array(
    [
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)

_HAMMING_H = np.array(
    [
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ],
    dtype=np.uint8,
)

# Syndrome -> bit position (0-based) to flip. Index by int(syndrome.dot([4,2,1])).
_HAMMING_SYNDROME_TABLE = {
    0b000: -1,
    0b110: 0,
    0b101: 1,
    0b011: 2,
    0b111: 3,
    0b100: 4,
    0b010: 5,
    0b001: 6,
}


class HammingCode74:
    """Standard Hamming(7,4) — single-error correcting."""

    def encode(self, bits: np.ndarray) -> np.ndarray:
        if bits.size % 4 != 0:
            padding = 4 - (bits.size % 4)
            bits = np.concatenate([bits.astype(np.uint8, copy=False), np.zeros(padding, dtype=np.uint8)])
        groups = bits.reshape(-1, 4)
        encoded = (groups @ _HAMMING_G) % 2
        return encoded.reshape(-1).astype(np.uint8, copy=False)

    def decode(self, bits: np.ndarray, original_length: int) -> tuple[np.ndarray, int]:
        if bits.size % 7 != 0:
            raise ValueError("Hamming(7,4) input must be multiple of 7 bits")
        words = bits.reshape(-1, 7).astype(np.uint8, copy=False)
        corrected_count = 0
        for idx in range(words.shape[0]):
            syndrome = (_HAMMING_H @ words[idx]) % 2
            key = int(syndrome[0] * 4 + syndrome[1] * 2 + syndrome[2])
            flip_pos = _HAMMING_SYNDROME_TABLE.get(key, -1)
            if flip_pos >= 0:
                words[idx, flip_pos] ^= 1
                corrected_count += 1
        data_bits = words[:, :4].reshape(-1)
        return data_bits[:original_length].astype(np.uint8, copy=False), corrected_count


class ECCCodec:
    """Composite codec dispatching to a level-specific pipeline."""

    def encode(self, bits: np.ndarray, level: ECCLevel) -> tuple[np.ndarray, ECCStats]:
        clean = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if level == ECCLevel.NONE:
            return clean.copy(), ECCStats(encoded_bits=clean.size, corrected_errors=0)
        if level == ECCLevel.LIGHT:
            encoded = HammingCode74().encode(clean)
            return encoded, ECCStats(encoded_bits=encoded.size, corrected_errors=0)
        if level == ECCLevel.MEDIUM:
            hammed = HammingCode74().encode(clean)
            repeated = RepetitionCode(3).encode(hammed)
            return repeated, ECCStats(encoded_bits=repeated.size, corrected_errors=0)
        if level == ECCLevel.HEAVY:
            hammed = HammingCode74().encode(clean)
            repeated = RepetitionCode(5).encode(hammed)
            return repeated, ECCStats(encoded_bits=repeated.size, corrected_errors=0)
        raise ValueError(f"unknown ECCLevel: {level}")

    def decode(self, bits: np.ndarray, level: ECCLevel, original_length: int) -> tuple[np.ndarray, ECCStats]:
        clean = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if level == ECCLevel.NONE:
            return clean[:original_length].copy(), ECCStats(encoded_bits=clean.size, corrected_errors=0)

        hamming_word_count = (original_length + 3) // 4
        hamming_total = hamming_word_count * 7

        if level == ECCLevel.LIGHT:
            decoded, corrected = HammingCode74().decode(clean[:hamming_total], original_length)
            return decoded, ECCStats(encoded_bits=clean.size, corrected_errors=corrected)

        repetition_factor = 3 if level == ECCLevel.MEDIUM else 5
        rep_total = hamming_total * repetition_factor
        if clean.size < rep_total:
            raise ValueError(
                f"encoded buffer too short: need {rep_total} bits, got {clean.size}"
            )
        hammed, rep_corrected = RepetitionCode(repetition_factor).decode(clean[:rep_total], hamming_total)
        decoded, ham_corrected = HammingCode74().decode(hammed, original_length)
        return decoded, ECCStats(
            encoded_bits=clean.size, corrected_errors=int(rep_corrected) + int(ham_corrected)
        )

    @staticmethod
    def encoded_length(original_bits: int, level: ECCLevel) -> int:
        if level == ECCLevel.NONE:
            return original_bits
        hamming_words = (original_bits + 3) // 4
        hamming_total = hamming_words * 7
        if level == ECCLevel.LIGHT:
            return hamming_total
        if level == ECCLevel.MEDIUM:
            return hamming_total * 3
        if level == ECCLevel.HEAVY:
            return hamming_total * 5
        raise ValueError(f"unknown ECCLevel: {level}")


__all__ = [
    "ECCCodec",
    "ECCLevel",
    "ECCStats",
    "HammingCode74",
    "RepetitionCode",
]
