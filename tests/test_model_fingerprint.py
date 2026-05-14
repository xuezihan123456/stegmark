"""Tests for neural model fingerprint traceability."""
from __future__ import annotations

import numpy as np
import pytest

from stegmark.core.model_fingerprint import (
    ModelFingerprint,
    detect_fingerprint,
    encode_into_features,
    generate_fingerprint,
    verify_fingerprint,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SECRET = b"test-secret-key-0"
_MODEL_ID = "gpt-sentinel-v1"
_OWNER = "acme-corp"
_DIM = 64
_K = 8


@pytest.fixture(scope="module")
def fp() -> ModelFingerprint:
    return generate_fingerprint(_MODEL_ID, _OWNER, _SECRET, dim=_DIM, k=_K)


# ---------------------------------------------------------------------------
# 1. generate_fingerprint contains model_id
# ---------------------------------------------------------------------------


def test_generate_contains_model_id():
    # Arrange
    model_id = "unique-model-xyz"

    # Act
    result = generate_fingerprint(model_id, _OWNER, _SECRET, dim=_DIM, k=_K)

    # Assert
    assert result.model_id == model_id


# ---------------------------------------------------------------------------
# 2. Different secret keys produce different basis matrices
# ---------------------------------------------------------------------------


def test_different_secret_keys_produce_different_basis():
    # Arrange
    secret_a = b"secret-aaaaaa"
    secret_b = b"secret-bbbbbb"

    # Act
    fp_a = generate_fingerprint(_MODEL_ID, _OWNER, secret_a, dim=_DIM, k=_K)
    fp_b = generate_fingerprint(_MODEL_ID, _OWNER, secret_b, dim=_DIM, k=_K)

    # Assert
    assert not np.allclose(fp_a.public_basis, fp_b.public_basis)


# ---------------------------------------------------------------------------
# 3. Basis rows are orthonormal
# ---------------------------------------------------------------------------


def test_basis_rows_are_orthonormal(fp: ModelFingerprint):
    # Arrange
    basis = fp.public_basis  # (k, d)

    # Act
    gram = basis @ basis.T  # should be identity

    # Assert
    assert basis.shape == (_K, _DIM)
    np.testing.assert_allclose(gram, np.eye(_K), atol=1e-10)


# ---------------------------------------------------------------------------
# 4. encode_into_features preserves shape
# ---------------------------------------------------------------------------


def test_encode_preserves_shape(fp: ModelFingerprint):
    # Arrange
    rng = np.random.default_rng(42)
    features = rng.standard_normal((_DIM,)).astype(np.float32)

    # Act
    encoded = encode_into_features(features, fp, alpha=0.01)

    # Assert
    assert encoded.shape == features.shape
    assert encoded.dtype == features.dtype


# ---------------------------------------------------------------------------
# 5. encode -> detect: z-score exceeds threshold
# ---------------------------------------------------------------------------


def test_encode_then_detect_exceeds_threshold(fp: ModelFingerprint):
    # Arrange — accumulate evidence over many model outputs (realistic detection scenario)
    rng = np.random.default_rng(0)
    features = rng.standard_normal((200, _DIM)).astype(np.float64)

    # Act
    encoded = encode_into_features(features, fp, alpha=0.1)
    detected, z_score = detect_fingerprint(encoded, fp, threshold=3.0)

    # Assert
    assert detected is True
    assert z_score > 3.0


# ---------------------------------------------------------------------------
# 6. Un-encoded features: z-score below threshold
# ---------------------------------------------------------------------------


def test_unencoded_features_not_detected(fp: ModelFingerprint):
    # Arrange – use a fixed seed for reproducibility
    rng = np.random.default_rng(999)
    z_scores = []

    # Act – average over many random vectors to confirm H0 behaviour
    for _ in range(50):
        features = rng.standard_normal((_DIM,))
        _, z = detect_fingerprint(features, fp, threshold=3.0)
        z_scores.append(z)

    mean_z = float(np.mean(z_scores))

    # Assert – mean z-score of random features must be well below threshold
    assert mean_z < 3.0


# ---------------------------------------------------------------------------
# 7. verify_fingerprint passes with correct secret key
# ---------------------------------------------------------------------------


def test_verify_correct_secret_passes(fp: ModelFingerprint):
    # Arrange / Act
    result = verify_fingerprint(fp, _MODEL_ID, _OWNER, _SECRET)

    # Assert
    assert result is True


# ---------------------------------------------------------------------------
# 8. verify_fingerprint fails with wrong secret key
# ---------------------------------------------------------------------------


def test_verify_wrong_secret_fails(fp: ModelFingerprint):
    # Arrange
    wrong_key = b"completely-wrong-secret"

    # Act
    result = verify_fingerprint(fp, _MODEL_ID, _OWNER, wrong_key)

    # Assert
    assert result is False


# ---------------------------------------------------------------------------
# Bonus: verify fails when model_id or owner is wrong
# ---------------------------------------------------------------------------


def test_verify_wrong_model_id_fails(fp: ModelFingerprint):
    # Arrange / Act
    result = verify_fingerprint(fp, "different-model", _OWNER, _SECRET)

    # Assert
    assert result is False


def test_verify_wrong_owner_fails(fp: ModelFingerprint):
    # Arrange / Act
    result = verify_fingerprint(fp, _MODEL_ID, "different-owner", _SECRET)

    # Assert
    assert result is False
