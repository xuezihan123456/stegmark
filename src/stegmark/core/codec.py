from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np

from stegmark.exceptions import InvalidInputError, MessageTooLongError
from stegmark.types import DecodedPayload

FRAME_VERSION = 1
HEADER_BYTES = 3
CRC_BYTES = 4
MAX_PAYLOAD_BYTES = 65_535


def encode_text(message: str, *, encoding: str = "utf-8") -> list[int]:
    return encode_payload(message.encode(encoding))


def encode_bits_hex(hex_payload: str) -> list[int]:
    normalized = hex_payload.strip().lower()
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    if len(normalized) % 2 != 0:
        raise InvalidInputError(
            "hex payload must contain an even number of characters",
            hint="Provide a byte-aligned hex string such as deadbeef.",
        )
    try:
        payload = bytes.fromhex(normalized)
    except ValueError as exc:
        raise InvalidInputError(
            "hex payload contains invalid characters",
            hint="Use hexadecimal characters only: 0-9 and a-f.",
        ) from exc
    return encode_payload(payload)


def encode_payload(payload: bytes) -> list[int]:
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise MessageTooLongError(
            "message payload exceeds the supported framed payload size",
            hint="Use a shorter message or a larger-capacity engine.",
        )

    frame = bytearray()
    frame.append(FRAME_VERSION)
    frame.extend(len(payload).to_bytes(2, "big"))
    frame.extend(payload)
    frame.extend(zlib.crc32(payload).to_bytes(4, "big"))
    return bytes_to_bits(bytes(frame))


def bytes_to_bits(data: bytes) -> list[int]:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big").tolist()


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    normalized = normalize_bits(bits)
    if len(normalized) % 8 != 0:
        raise InvalidInputError(
            "bitstream length must be byte-aligned",
            hint="Pad or trim the bitstream so its length is divisible by 8.",
        )

    result = bytearray()
    for index in range(0, len(normalized), 8):
        byte = 0
        for bit in normalized[index : index + 8]:
            byte = (byte << 1) | bit
        result.append(byte)
    return bytes(result)


def payload_to_hex(payload: bytes | None) -> str | None:
    return payload.hex() if payload is not None else None


def resolve_payload_bits(
    message: str | None,
    payload_bits: Sequence[int] | None,
    *,
    encoding: str = "utf-8",
) -> tuple[int, ...]:
    if message is not None and payload_bits is not None:
        raise InvalidInputError(
            "provide either message or payload bits, not both",
            hint="Use text mode or bits mode for a single embed operation.",
        )
    if payload_bits is not None:
        return tuple(int(bit) for bit in payload_bits)
    if message is None:
        raise InvalidInputError(
            "missing message or payload bits",
            hint="Provide a text message or a hex bits payload.",
        )
    return tuple(encode_text(message, encoding=encoding))


def decode_bitstream(bits: Sequence[int], *, encoding: str = "utf-8") -> DecodedPayload:
    normalized = tuple(normalize_bits(bits))
    try:
        frame = bits_to_bytes(normalized)
    except InvalidInputError:
        return DecodedPayload(valid=False, bits=normalized, error="invalid_bitstream")

    if len(frame) < HEADER_BYTES + CRC_BYTES:
        return DecodedPayload(valid=False, bits=normalized, error="truncated_header")

    version = frame[0]
    payload_length = int.from_bytes(frame[1:3], "big")
    if payload_length > MAX_PAYLOAD_BYTES:
        raise InvalidInputError(f"payload too large: {payload_length} bytes", hint=f"Maximum payload is {MAX_PAYLOAD_BYTES} bytes.")
    payload_end = HEADER_BYTES + payload_length
    crc_end = payload_end + CRC_BYTES

    if len(frame) < crc_end:
        return DecodedPayload(valid=False, bits=normalized, version=version, error="truncated_payload")

    payload = frame[HEADER_BYTES:payload_end]
    expected_crc = int.from_bytes(frame[payload_end:crc_end], "big")
    actual_crc = zlib.crc32(payload)
    if actual_crc != expected_crc:
        return DecodedPayload(valid=False, bits=normalized, version=version, error="crc_mismatch")

    try:
        message = payload.decode(encoding)
    except UnicodeDecodeError:
        message = None
    return DecodedPayload(
        valid=True,
        bits=normalized,
        payload=payload,
        message=message,
        version=version,
    )


def normalize_bits(bits: Sequence[int]) -> list[int]:
    normalized: list[int] = []
    for bit in bits:
        if bit not in (0, 1):
            raise InvalidInputError(
                "bitstream contains values other than 0 or 1",
                hint="Use only binary values when decoding a framed payload.",
            )
        normalized.append(int(bit))
    return normalized
