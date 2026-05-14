"""Zero-knowledge ownership proof via Pedersen commitment + Schnorr Sigma protocol.

Educational MVP using secp256k1.  Only stdlib is required (secrets, hashlib,
dataclasses).  This is NOT production-grade cryptography; it is intended for
learning and prototyping within the StegMark v0.4 research branch.

Schnorr Sigma protocol (honest-verifier ZK, non-interactive via Fiat-Shamir):

  Setup:
    secret_int s  (derived from secret_key via SHA-256 mod n)
    blinding   b  (random or caller-supplied)
    H           = independent generator (sha256("H") · G)
    C           = s·G + b·H            ← Pedersen commitment

  Prove:
    1. Sample random v, t ∈ [1, n-1]
    2. V = v·G + t·H                   ← commitment to randomness
    3. c = SHA-256(image_hash ‖ C ‖ V) mod n   ← Fiat-Shamir challenge
    4. s_resp = (v + c·s) mod n        ← response for secret
    5. r_resp = (t + c·b) mod n        ← response for blinding

  Verify:
    1. Recompute c from stored (image_hash, C, V')
    2. V' = s_resp·G + r_resp·H - c·C
    3. Accept iff c == c'
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# secp256k1 curve parameters (NIST / SEC2 standard constants)
# ---------------------------------------------------------------------------
_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F  # field prime
_A = 0  # curve coefficient a
_B = 7  # curve coefficient b
_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798  # noqa: E501
_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8  # noqa: E501
_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141  # group order

# Sentinel for the point at infinity
_INF = (None, None)


# ---------------------------------------------------------------------------
# ECPoint
# ---------------------------------------------------------------------------
class ECPoint:
    """Affine coordinate point on secp256k1.

    The point at infinity is represented as ``ECPoint(None, None)``.
    All arithmetic is performed mod *p* in the base field and mod *n*
    only at the scalar level.
    """

    __slots__ = ("x", "y")

    def __init__(self, x: int | None, y: int | None) -> None:
        self.x = x
        self.y = y

    # ------------------------------------------------------------------
    # predicates
    # ------------------------------------------------------------------

    @property
    def is_infinity(self) -> bool:
        return self.x is None

    def eq(self, other: ECPoint) -> bool:
        return self.x == other.x and self.y == other.y

    # ------------------------------------------------------------------
    # arithmetic
    # ------------------------------------------------------------------

    def neg(self) -> ECPoint:
        """Return the additive inverse (reflection over x-axis)."""
        if self.is_infinity:
            return ECPoint(None, None)
        return ECPoint(self.x, (-self.y) % _P)  # type: ignore[operator]

    def double(self) -> ECPoint:
        """Point doubling: 2·P."""
        if self.is_infinity or self.y == 0:
            return ECPoint(None, None)
        x, y = self.x, self.y
        lam = (3 * x * x + _A) * pow(2 * y, -1, _P) % _P  # type: ignore[operator]
        x3 = (lam * lam - 2 * x) % _P  # type: ignore[operator]
        y3 = (lam * (x - x3) - y) % _P  # type: ignore[operator]
        return ECPoint(x3, y3)

    def add(self, other: ECPoint) -> ECPoint:
        """Point addition: P + Q."""
        if self.is_infinity:
            return other
        if other.is_infinity:
            return self
        if self.x == other.x:
            if self.y != other.y or self.y == 0:
                return ECPoint(None, None)
            return self.double()
        x1, y1 = self.x, self.y
        x2, y2 = other.x, other.y
        lam = (y2 - y1) * pow(x2 - x1, -1, _P) % _P  # type: ignore[operator]
        x3 = (lam * lam - x1 - x2) % _P  # type: ignore[operator]
        y3 = (lam * (x1 - x3) - y1) % _P  # type: ignore[operator]
        return ECPoint(x3, y3)

    def scalar_mul(self, k: int) -> ECPoint:
        """Double-and-add scalar multiplication: k·P."""
        k = k % _N
        if k == 0 or self.is_infinity:
            return ECPoint(None, None)
        result = ECPoint(None, None)
        addend = ECPoint(self.x, self.y)
        while k:
            if k & 1:
                result = result.add(addend)
            addend = addend.double()
            k >>= 1
        return result

    def __repr__(self) -> str:
        if self.is_infinity:
            return "ECPoint(∞)"
        return f"ECPoint(x=0x{self.x:x}, y=0x{self.y:x})"  # type: ignore[str-bytes-safe]


# ---------------------------------------------------------------------------
# Curve-level helpers
# ---------------------------------------------------------------------------

def _G() -> ECPoint:
    """Return the secp256k1 base point G."""
    return ECPoint(_GX, _GY)


def get_H_point() -> ECPoint:
    """Return a second independent generator H = hash_to_point(``sha256("H")``).

    We derive H by hashing the ASCII string "H" and using the resulting
    integer (mod n) as a scalar multiplied by G.  This gives a point whose
    discrete-log relationship to G is computationally unknown, which is the
    binding property required by Pedersen commitments.
    """
    h_scalar = int.from_bytes(hashlib.sha256(b"H").digest(), "big") % _N
    return _G().scalar_mul(h_scalar)


# ---------------------------------------------------------------------------
# Pedersen commitment
# ---------------------------------------------------------------------------

def pedersen_commit(secret: int, blinding: int, H: ECPoint) -> ECPoint:
    """Compute C = secret·G + blinding·H (Pedersen commitment).

    Args:
        secret:   The secret integer to commit to (0 < secret < n).
        blinding: A random blinding factor (0 < blinding < n).
        H:        An independent generator point (use ``get_H_point()``).

    Returns:
        The commitment point C on secp256k1.
    """
    return _G().scalar_mul(secret).add(H.scalar_mul(blinding))


# ---------------------------------------------------------------------------
# Proof dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OwnershipProof:
    """Non-interactive Schnorr ownership proof (Fiat-Shamir heuristic).

    Attributes:
        commitment_xy: (x, y) coordinates of the Pedersen commitment C.
        challenge:     32-byte Fiat-Shamir challenge hash.
        response_s:    Scalar response for the secret (v + c·s mod n).
        response_r:    Scalar response for the blinding factor (t + c·b mod n).
    """

    commitment_xy: tuple[int, int]
    challenge: bytes
    response_s: int
    response_r: int


# ---------------------------------------------------------------------------
# Prove
# ---------------------------------------------------------------------------

def prove_ownership(
    secret_key: bytes,
    image_hash: bytes,
    blinding: int | None = None,
) -> OwnershipProof:
    """Generate a non-interactive Schnorr ownership proof.

    Args:
        secret_key: Raw secret bytes (e.g. 32 bytes from os.urandom).
                    The actual secret integer is ``sha256(secret_key) mod n``.
        image_hash: SHA-256 digest of the image being watermarked; used as
                    the Fiat-Shamir domain separator.
        blinding:   Optional blinding factor.  Randomly generated when None.

    Returns:
        An :class:`OwnershipProof` that can be verified with
        :func:`verify_ownership`.
    """
    G = _G()
    H = get_H_point()

    # Derive secret integer from key bytes
    secret_int = int.from_bytes(hashlib.sha256(secret_key).digest(), "big") % _N

    # Choose blinding factor
    if blinding is None:
        blinding = secrets.randbelow(_N - 1) + 1

    # Pedersen commitment C = s·G + b·H
    C = pedersen_commit(secret_int, blinding, H)

    # Random nonces for the proof
    v = secrets.randbelow(_N - 1) + 1
    t = secrets.randbelow(_N - 1) + 1

    # Commitment to randomness: V = v·G + t·H
    V = G.scalar_mul(v).add(H.scalar_mul(t))

    # Fiat-Shamir challenge: c = sha256(image_hash ‖ Cx ‖ Cy ‖ Vx ‖ Vy) mod n
    cx_bytes = C.x.to_bytes(32, "big")  # type: ignore[union-attr]
    cy_bytes = C.y.to_bytes(32, "big")  # type: ignore[union-attr]
    vx_bytes = V.x.to_bytes(32, "big")  # type: ignore[union-attr]
    vy_bytes = V.y.to_bytes(32, "big")  # type: ignore[union-attr]
    challenge_hash = hashlib.sha256(image_hash + cx_bytes + cy_bytes + vx_bytes + vy_bytes).digest()
    c = int.from_bytes(challenge_hash, "big") % _N

    # Responses
    s_resp = (v + c * secret_int) % _N
    r_resp = (t + c * blinding) % _N

    return OwnershipProof(
        commitment_xy=(C.x, C.y),  # type: ignore[arg-type]
        challenge=challenge_hash,
        response_s=s_resp,
        response_r=r_resp,
    )


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_ownership(proof: OwnershipProof, image_hash: bytes, claimed_commitment: ECPoint) -> bool:  # noqa: E501
    """Verify a Schnorr ownership proof against a Pedersen commitment.

    The verifier reconstructs the randomness commitment V' from the proof
    responses and checks that the Fiat-Shamir challenge matches.

    Args:
        proof:               The :class:`OwnershipProof` produced by :func:`prove_ownership`.
        image_hash:          SHA-256 digest of the image (same as used during proving).
        claimed_commitment:  The Pedersen commitment C the prover claims.

    Returns:
        ``True`` if the proof is valid; ``False`` otherwise.
    """
    G = _G()
    H = get_H_point()

    # Recover challenge integer from stored bytes
    c = int.from_bytes(proof.challenge, "big") % _N

    # Reconstruct V' = s·G + r·H - c·C
    sG = G.scalar_mul(proof.response_s)
    rH = H.scalar_mul(proof.response_r)
    cC = claimed_commitment.scalar_mul(c)
    V_prime = sG.add(rH).add(cC.neg())

    if V_prime.is_infinity:
        return False

    # Recompute Fiat-Shamir challenge from (image_hash, C, V')
    cx_bytes = claimed_commitment.x.to_bytes(32, "big")  # type: ignore[union-attr]
    cy_bytes = claimed_commitment.y.to_bytes(32, "big")  # type: ignore[union-attr]
    vx_bytes = V_prime.x.to_bytes(32, "big")  # type: ignore[union-attr]
    vy_bytes = V_prime.y.to_bytes(32, "big")  # type: ignore[union-attr]
    expected_challenge = hashlib.sha256(image_hash + cx_bytes + cy_bytes + vx_bytes + vy_bytes).digest()

    return proof.challenge == expected_challenge


__all__ = [
    "ECPoint",
    "OwnershipProof",
    "get_H_point",
    "pedersen_commit",
    "prove_ownership",
    "verify_ownership",
]
