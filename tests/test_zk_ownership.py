"""Tests for zero-knowledge ownership proof (Pedersen commitment + Schnorr).

Run with:
    python -m pytest tests/test_zk_ownership.py -v
"""
from __future__ import annotations

import hashlib

from stegmark.core.zk_ownership import (
    _G,  # noqa: PLC2701  (internal helper, acceptable in tests)
    _N,
    ECPoint,
    OwnershipProof,
    get_H_point,
    pedersen_commit,
    prove_ownership,
    verify_ownership,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_n_times(point: ECPoint, n: int) -> ECPoint:
    """Add *point* to itself *n* times using repeated .add()."""
    result = ECPoint(None, None)
    for _ in range(n):
        result = result.add(point)
    return result


# ---------------------------------------------------------------------------
# Test 1 – G + (-G) == point at infinity
# ---------------------------------------------------------------------------

def test_point_plus_negation_is_infinity() -> None:
    G = _G()
    neg_G = G.neg()
    result = G.add(neg_G)
    assert result.is_infinity, "G + (-G) must equal the point at infinity"


# ---------------------------------------------------------------------------
# Test 2 – scalar_mul(k=2) == double(G)
# ---------------------------------------------------------------------------

def test_scalar_mul_2_equals_double() -> None:
    G = _G()
    via_scalar = G.scalar_mul(2)
    via_double = G.double()
    assert via_scalar.eq(via_double), "scalar_mul(2) must equal double()"


# ---------------------------------------------------------------------------
# Test 3 – scalar_mul(k=5) == adding G five times
# ---------------------------------------------------------------------------

def test_scalar_mul_5_equals_repeated_add() -> None:
    G = _G()
    via_scalar = G.scalar_mul(5)
    via_add = _add_n_times(G, 5)
    assert via_scalar.eq(via_add), "scalar_mul(5) must equal G added 5 times"


# ---------------------------------------------------------------------------
# Test 4 – pedersen_commit is deterministic (same inputs → same output)
# ---------------------------------------------------------------------------

def test_pedersen_commit_is_deterministic() -> None:
    H = get_H_point()
    secret = 0xDEADBEEF_CAFEF00D
    blinding = 0x1234567890ABCDEF

    C1 = pedersen_commit(secret, blinding, H)
    C2 = pedersen_commit(secret, blinding, H)

    assert C1.eq(C2), "pedersen_commit must be deterministic for identical inputs"
    assert not C1.is_infinity, "Commitment must not be the point at infinity"


# ---------------------------------------------------------------------------
# Test 5 – prove_ownership returns a complete OwnershipProof
# ---------------------------------------------------------------------------

def test_prove_ownership_returns_complete_proof() -> None:
    secret_key = b"super-secret-key-for-testing-001"
    image_hash = hashlib.sha256(b"test-image-bytes").digest()

    proof = prove_ownership(secret_key, image_hash)

    assert isinstance(proof, OwnershipProof)
    assert isinstance(proof.commitment_xy, tuple) and len(proof.commitment_xy) == 2
    assert isinstance(proof.challenge, bytes) and len(proof.challenge) == 32
    assert isinstance(proof.response_s, int) and proof.response_s > 0
    assert isinstance(proof.response_r, int) and proof.response_r > 0


# ---------------------------------------------------------------------------
# Test 6 – verify_ownership accepts a legitimate proof
# ---------------------------------------------------------------------------

def test_verify_ownership_accepts_valid_proof() -> None:
    secret_key = b"legitimate-owner-key-v0.4-stegmk"
    image_hash = hashlib.sha256(b"real-image-data").digest()
    blinding = 0xABCDEF1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF1234567890  # noqa: E501

    proof = prove_ownership(secret_key, image_hash, blinding=blinding)

    # Reconstruct the commitment independently to simulate a verifier
    secret_int = int.from_bytes(hashlib.sha256(secret_key).digest(), "big") % _N
    H = get_H_point()
    C = pedersen_commit(secret_int, blinding, H)

    assert verify_ownership(proof, image_hash, C), "Valid proof must be accepted by verifier"


# ---------------------------------------------------------------------------
# Test 7 – tampered challenge causes verify to fail
# ---------------------------------------------------------------------------

def test_tampered_challenge_fails_verification() -> None:
    secret_key = b"owner-key-tamper-test-stegmark04"
    image_hash = hashlib.sha256(b"image-for-tamper-test").digest()
    blinding = 42

    proof = prove_ownership(secret_key, image_hash, blinding=blinding)

    secret_int = int.from_bytes(hashlib.sha256(secret_key).digest(), "big") % _N
    H = get_H_point()
    C = pedersen_commit(secret_int, blinding, H)

    # Flip one byte in the challenge
    bad_challenge = bytes([proof.challenge[0] ^ 0xFF]) + proof.challenge[1:]
    tampered_proof = OwnershipProof(
        commitment_xy=proof.commitment_xy,
        challenge=bad_challenge,
        response_s=proof.response_s,
        response_r=proof.response_r,
    )

    assert not verify_ownership(tampered_proof, image_hash, C), \
        "Proof with tampered challenge must be rejected"


# ---------------------------------------------------------------------------
# Test 8 – wrong commitment causes verify to fail
# ---------------------------------------------------------------------------

def test_wrong_commitment_fails_verification() -> None:
    secret_key = b"real-owner-key-for-stegmark-0004"
    image_hash = hashlib.sha256(b"original-image").digest()
    blinding = 99

    proof = prove_ownership(secret_key, image_hash, blinding=blinding)

    # Attacker presents a different commitment (G itself)
    wrong_commitment = _G()

    assert not verify_ownership(proof, image_hash, wrong_commitment), \
        "Proof verified against wrong commitment must be rejected"


# ---------------------------------------------------------------------------
# Bonus: sanity-check that G lies on the curve y² = x³ + 7 (mod p)
# ---------------------------------------------------------------------------

def test_generator_point_on_curve() -> None:
    from stegmark.core.zk_ownership import _P  # noqa: PLC2701
    G = _G()
    lhs = (G.y * G.y) % _P  # type: ignore[operator]
    rhs = (G.x * G.x * G.x + 7) % _P  # type: ignore[operator]
    assert lhs == rhs, "G must satisfy the secp256k1 curve equation"
