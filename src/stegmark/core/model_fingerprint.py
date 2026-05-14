"""Neural model fingerprint for generative model output traceability.

Embeds a unique, owner-bound fingerprint into a model's feature space so that
outputs can later be attributed to a specific model instance.

Algorithm
---------
1. Derive a pseudo-random basis from HMAC-SHA256(secret_key, model_id||owner)
   via a seeded PRNG, then orthogonalise it with QR decomposition.
2. Embed: add alpha-scaled basis patterns to the feature vector.
3. Detect: project features onto the basis and compute a z-score against the
   expected projection magnitude.
4. Verify the HMAC signature stored inside the fingerprint object.

Pure-numpy; no torch / external ML dependency.
"""
from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFingerprint:
    """Immutable fingerprint bound to a model and its owner."""

    model_id: str
    owner: str
    created_at: str  # ISO 8601
    signature: bytes  # HMAC-SHA256[:32] over (model_id || owner || basis_bytes)
    public_basis: np.ndarray  # shape (k, d), orthonormal rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derive_seed(secret_key: bytes, model_id: str, owner: str) -> int:
    """Derive a 64-bit seed via HMAC-SHA256."""
    msg = model_id.encode() + owner.encode()
    digest = hmac.new(secret_key, msg, hashlib.sha256).digest()
    # Take the first 8 bytes as a little-endian uint64.
    return struct.unpack("<Q", digest[:8])[0]


def _orthonormal_basis(rng: np.random.Generator, k: int, d: int) -> np.ndarray:
    """Return a (k, d) matrix with orthonormal rows via QR decomposition."""
    raw = rng.standard_normal((d, k)).astype(np.float64)
    q, _ = np.linalg.qr(raw)  # q: (d, k)
    basis = q.T  # (k, d)
    # Normalise each row to unit length (QR guarantees orthonormality, but be explicit).
    norms = np.linalg.norm(basis, axis=1, keepdims=True)
    return (basis / np.maximum(norms, 1e-12)).astype(np.float64)


def _compute_signature(secret_key: bytes, model_id: str, owner: str, basis: np.ndarray) -> bytes:
    """HMAC-SHA256 over model_id || owner || basis_bytes, truncated to 32 bytes."""
    basis_bytes = basis.astype(np.float64).tobytes()
    msg = model_id.encode() + owner.encode() + basis_bytes
    return hmac.new(secret_key, msg, hashlib.sha256).digest()[:32]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_fingerprint(
    model_id: str,
    owner: str,
    secret_key: bytes,
    dim: int = 64,
    k: int = 8,
) -> ModelFingerprint:
    """Generate a unique fingerprint for a model.

    Parameters
    ----------
    model_id:
        Unique identifier for the model instance.
    owner:
        Owner / organisation name.
    secret_key:
        Secret bytes used to derive the PRNG seed and HMAC signature.
    dim:
        Feature dimension ``d`` the fingerprint will be applied to.
    k:
        Number of orthogonal basis vectors.

    Returns
    -------
    ModelFingerprint
        Immutable fingerprint object.
    """
    if k > dim:
        raise ValueError(f"k ({k}) cannot exceed dim ({dim})")

    seed = _derive_seed(secret_key, model_id, owner)
    rng = np.random.default_rng(seed)
    basis = _orthonormal_basis(rng, k, dim)
    signature = _compute_signature(secret_key, model_id, owner, basis)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return ModelFingerprint(
        model_id=model_id,
        owner=owner,
        created_at=created_at,
        signature=signature,
        public_basis=basis,
    )


def encode_into_features(
    features: np.ndarray,
    fp: ModelFingerprint,
    alpha: float = 0.01,
) -> np.ndarray:
    """Inject the fingerprint pattern into a feature vector or batch.

    Parameters
    ----------
    features:
        Array of shape ``(..., d)`` where ``d`` matches ``fp.public_basis.shape[1]``.
    fp:
        Fingerprint whose basis rows are summed and scaled.
    alpha:
        Injection strength (additive scale factor).

    Returns
    -------
    np.ndarray
        Array of the same shape and dtype as *features*.
    """
    d = fp.public_basis.shape[1]
    if features.shape[-1] != d:
        raise ValueError(
            f"Last dimension of features ({features.shape[-1]}) must match "
            f"fingerprint dim ({d})"
        )

    original_dtype = features.dtype
    pattern = fp.public_basis.sum(axis=0)  # (d,) — sum of k orthonormal rows
    pattern = pattern / np.maximum(np.linalg.norm(pattern), 1e-12)

    feat_f64 = features.astype(np.float64)
    feat_norm = float(np.linalg.norm(feat_f64))
    # alpha is interpreted as a relative perturbation strength; scale the pattern by the feature norm.
    result = feat_f64 + alpha * feat_norm * pattern
    return result.astype(original_dtype)


def detect_fingerprint(
    features: np.ndarray,
    fp: ModelFingerprint,
    threshold: float = 3.0,
) -> tuple[bool, float]:
    """Detect whether a fingerprint is present in a feature vector.

    The projection of *features* onto each basis row is computed; the mean
    projection is compared against the null distribution (zero-mean unit-norm
    random vector) to produce a z-score.

    Parameters
    ----------
    features:
        1-D array of shape ``(d,)`` or 2-D array of shape ``(n, d)``.
        When 2-D the mean feature vector is used.
    fp:
        Fingerprint to check against.
    threshold:
        z-score threshold above which a fingerprint is declared present.

    Returns
    -------
    tuple[bool, float]
        ``(detected, z_score)``
    """
    feat = np.asarray(features, dtype=np.float64)
    if feat.ndim == 2:
        feat = feat.mean(axis=0)
    if feat.ndim != 1:
        raise ValueError("features must be 1-D or 2-D")

    d = fp.public_basis.shape[1]
    if feat.shape[0] != d:
        raise ValueError(f"features dim ({feat.shape[0]}) does not match fingerprint dim ({d})")

    # Normalise the feature vector.
    feat_norm = feat / np.maximum(np.linalg.norm(feat), 1e-12)

    # Project onto each basis row; expected value under H0 ~ N(0, 1/sqrt(d)).
    projections = fp.public_basis @ feat_norm  # (k,)
    mean_proj = float(np.mean(projections))

    # Under H0: E[proj_i] = 0, Var[proj_i] = 1/d  (unit-norm random vector)
    # => std = 1/sqrt(d), z = mean_proj / std * sqrt(k) for k independent projections
    k, _ = fp.public_basis.shape
    std_null = 1.0 / np.sqrt(d)
    z_score = mean_proj / (std_null / np.sqrt(k))

    detected = float(z_score) > threshold
    return detected, float(z_score)


def verify_fingerprint(
    fp: ModelFingerprint,
    model_id: str,
    owner: str,
    secret_key: bytes,
) -> bool:
    """Verify that a fingerprint's HMAC signature is authentic.

    Parameters
    ----------
    fp:
        The fingerprint object to verify.
    model_id:
        Expected model identifier.
    owner:
        Expected owner name.
    secret_key:
        Secret key used during generation.

    Returns
    -------
    bool
        ``True`` if the signature matches and the metadata is consistent.
    """
    if fp.model_id != model_id or fp.owner != owner:
        return False

    expected_sig = _compute_signature(secret_key, model_id, owner, fp.public_basis)
    return hmac.compare_digest(fp.signature, expected_sig)


__all__ = [
    "ModelFingerprint",
    "detect_fingerprint",
    "encode_into_features",
    "generate_fingerprint",
    "verify_fingerprint",
]
