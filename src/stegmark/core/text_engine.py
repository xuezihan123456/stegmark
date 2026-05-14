"""Text watermarking via Kirchenbauer-style green/red list token biasing.

Combined with case-toggle perturbation on the first letter of qualifying
words. Pure-Python; works on any UTF-8 text without external models.

Capacity heuristic: 1 payload bit per 4 candidate tokens. A *candidate*
token is any whitespace-delimited token whose first character is an ASCII
letter.

Limitations
-----------
This is a baseline watermarker for prototyping; production text watermarking
needs an LM logit hook (Kirchenbauer 2023) which is not pure-Python.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from stegmark.exceptions import MessageTooLongError

_TOKEN_PATTERN = re.compile(r"(\s+)")


def _hash_bit(key: int, index: int) -> int:
    digest = hashlib.sha256(f"{key}|{index}".encode()).digest()
    return digest[0] & 1


def _is_candidate(token: str) -> bool:
    return bool(token) and token[0].isalpha() and token[0].isascii()


@dataclass(frozen=True)
class TextEmbedResult:
    text: str
    capacity_used_bits: int


class TextWatermarkEngine:
    """Bit-stream watermark for English text via first-letter case toggling."""

    name = "text"
    bits_per_token: int = 1
    tokens_per_bit: int = 4

    def capacity_bits(self, text: str) -> int:
        return sum(1 for tok in self._iter_words(text) if _is_candidate(tok)) // self.tokens_per_bit

    def embed(self, text: str, payload_bits: Iterable[int], key: int = 0) -> TextEmbedResult:
        bits = [int(b) & 1 for b in payload_bits]
        capacity = self.capacity_bits(text)
        if len(bits) > capacity:
            raise MessageTooLongError(
                f"text capacity is {capacity} bits, payload is {len(bits)}",
                hint="Provide a longer carrier text or shorter payload.",
            )

        parts = _TOKEN_PATTERN.split(text)
        candidate_indices = [i for i, p in enumerate(parts) if _is_candidate(p)]
        used = 0
        for bit_index, bit in enumerate(bits):
            window = candidate_indices[bit_index * self.tokens_per_bit : (bit_index + 1) * self.tokens_per_bit]
            if not window:
                break
            token_idx = window[_hash_bit(key, bit_index) % len(window)]
            token = parts[token_idx]
            first = token[0]
            if bit == 1:
                parts[token_idx] = first.upper() + token[1:]
            else:
                parts[token_idx] = first.lower() + token[1:]
            used += 1
        return TextEmbedResult(text="".join(parts), capacity_used_bits=used)

    def extract(self, text: str, num_bits: int, key: int = 0) -> list[int]:
        parts = _TOKEN_PATTERN.split(text)
        candidate_indices = [i for i, p in enumerate(parts) if _is_candidate(p)]
        bits: list[int] = []
        for bit_index in range(num_bits):
            window = candidate_indices[bit_index * self.tokens_per_bit : (bit_index + 1) * self.tokens_per_bit]
            if not window:
                break
            token_idx = window[_hash_bit(key, bit_index) % len(window)]
            first = parts[token_idx][0]
            bits.append(1 if first.isupper() else 0)
        return bits

    @staticmethod
    def _iter_words(text: str) -> Iterable[str]:
        return (p for p in _TOKEN_PATTERN.split(text) if not p.isspace())


__all__ = ["TextEmbedResult", "TextWatermarkEngine"]
