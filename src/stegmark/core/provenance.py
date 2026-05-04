from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from stegmark.core.codec import encode_text, decode_bitstream
from stegmark.exceptions import MessageTooLongError

PROVENANCE_LAYERS: list[tuple[tuple[int,int], tuple[int,int]]] = [
    ((1,2),(2,1)), ((3,2),(2,3)), ((5,4),(4,5)), ((7,6),(6,7)),
]
MAX_PROVENANCE_LAYERS = 4
BLOCK_SIZE = 8
BASE_DELTA = 20.0


@dataclass(frozen=True)
class ProvenanceEntry:
    operator: str
    action: str
    timestamp: str
    message: str
    layer: int
    def to_dict(self) -> dict[str,Any]:
        return {"operator":self.operator,"action":self.action,"timestamp":self.timestamp,"message":self.message,"layer":self.layer}


@dataclass(frozen=True)
class ProvenanceChain:
    entries: tuple[ProvenanceEntry,...] = ()
    @property
    def is_empty(self) -> bool: return len(self.entries)==0
    @property
    def depth(self) -> int: return len(self.entries)
    def to_list(self) -> list[dict[str,Any]]: return [e.to_dict() for e in self.entries]


def _rgb_to_ycbcr_simple(image):
    rgb = np.asarray(image, dtype=np.float32)
    y = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    cb = 128.0 - 0.168736*rgb[:,:,0] - 0.331264*rgb[:,:,1] + 0.5*rgb[:,:,2]
    cr = 128.0 + 0.5*rgb[:,:,0] - 0.418688*rgb[:,:,1] - 0.081312*rgb[:,:,2]
    return y, cb, cr


def _ycbcr_to_rgb_simple(y, cb, cr):
    r = y + 1.402*(cr-128.0)
    g = y - 0.344136*(cb-128.0) - 0.714136*(cr-128.0)
    b = y + 1.772*(cb-128.0)
    return np.clip(np.stack([r,g,b], axis=2), 0, 255).astype(np.uint8)


def _dct2_block(block):
    N = block.shape[0]
    m = np.zeros((N,N), dtype=np.float64)
    for k in range(N):
        for n in range(N):
            m[k,n] = np.cos(np.pi*k*(2*n+1)/(2*N))
    m[0,:] *= np.sqrt(1.0/N)
    m[1:,:] *= np.sqrt(2.0/N)
    return m @ block @ m.T


def _idct2_block(coeffs):
    N = coeffs.shape[0]
    m = np.zeros((N,N), dtype=np.float64)
    for k in range(N):
        for n in range(N):
            m[k,n] = np.cos(np.pi*k*(2*n+1)/(2*N))
    m[0,:] *= np.sqrt(1.0/N)
    m[1:,:] *= np.sqrt(2.0/N)
    return m.T @ coeffs @ m


def multi_layer_embed(image, entries, *, strength=1.0):
    if len(entries) > MAX_PROVENANCE_LAYERS:
        raise MessageTooLongError("too many provenance layers", hint="At most 4 layers supported.")
    h, w = image.shape[:2]
    uh = h - (h % BLOCK_SIZE)
    uw = w - (w % BLOCK_SIZE)
    br, bc = uh // BLOCK_SIZE, uw // BLOCK_SIZE
    y, cb, cr = _rgb_to_ycbcr_simple(image)
    yr = y.copy()
    for ei, entry in enumerate(entries):
        bits = encode_text(entry.message)
        ca, cb_coeff = PROVENANCE_LAYERS[ei]
        delta = max(2.0, BASE_DELTA * strength * (ei+1) / MAX_PROVENANCE_LAYERS)
        bi = 0
        for r in range(br):
            if bi >= len(bits): break
            for c in range(bc):
                if bi >= len(bits): break
                y0, y1 = r*BLOCK_SIZE, (r+1)*BLOCK_SIZE
                x0, x1 = c*BLOCK_SIZE, (c+1)*BLOCK_SIZE
                blk = yr[y0:y1, x0:x1].astype(np.float64)
                coeffs = _dct2_block(blk)
                mid = (coeffs[ca] + coeffs[cb_coeff]) / 2.0
                s = delta/2.0 if bits[bi]==1 else -delta/2.0
                coeffs[ca] = mid + s
                coeffs[cb_coeff] = mid - s
                yr[y0:y1, x0:x1] = _idct2_block(coeffs)
                bi += 1
    return _ycbcr_to_rgb_simple(yr, cb, cr)


def multi_layer_extract(image):
    h, w = image.shape[:2]
    uh = h - (h % BLOCK_SIZE)
    uw = w - (w % BLOCK_SIZE)
    br, bc = uh // BLOCK_SIZE, uw // BLOCK_SIZE
    y, _, _ = _rgb_to_ycbcr_simple(image)
    entries: list[ProvenanceEntry] = []
    for li, (ca, cb_coeff) in enumerate(PROVENANCE_LAYERS):
        bits: list[int] = []
        for r in range(br):
            for c in range(bc):
                y0, y1 = r*BLOCK_SIZE, (r+1)*BLOCK_SIZE
                x0, x1 = c*BLOCK_SIZE, (c+1)*BLOCK_SIZE
                blk = y[y0:y1, x0:x1].astype(np.float64)
                coeffs = _dct2_block(blk)
                bits.append(1 if coeffs[ca] >= coeffs[cb_coeff] else 0)
        decoded = decode_bitstream(bits)
        if decoded.valid and decoded.message:
            entries.append(ProvenanceEntry(operator="unknown",action="unknown",timestamp="unknown",message=decoded.message,layer=li))
        elif decoded.valid and decoded.payload:
            entries.append(ProvenanceEntry(operator="unknown",action="unknown",timestamp="unknown",message=decoded.payload.hex(),layer=li))
    return ProvenanceChain(entries=tuple(entries))


def build_provenance_entry(operator, action, message, layer):
    return ProvenanceEntry(operator=operator, action=action, timestamp=datetime.now(timezone.utc).isoformat(), message=message, layer=layer)


__all__ = ["ProvenanceEntry","ProvenanceChain","multi_layer_embed","multi_layer_extract","build_provenance_entry","MAX_PROVENANCE_LAYERS","PROVENANCE_LAYERS"]
