"""Tests for the C2PA-compatible manifest builder."""
from __future__ import annotations

import json

from stegmark.core.c2pa_manifest import C2PAManifestBuilder


def test_empty_manifest_has_instance_id_and_generator():
    manifest = C2PAManifestBuilder().sign()

    assert manifest["claim_generator"].startswith("StegMark")
    assert manifest["instance_id"].startswith("urn:uuid:")
    assert manifest["assertions"] == []


def test_add_action_records_assertion():
    manifest = C2PAManifestBuilder().add_action("c2pa.created").sign()

    labels = [a["label"] for a in manifest["assertions"]]
    assert "c2pa.actions" in labels


def test_add_watermark_assertion_includes_engine_and_payload():
    manifest = (
        C2PAManifestBuilder()
        .add_watermark_assertion(engine="native", payload_hex="deadbeef", strength=1.2)
        .sign()
    )
    wm = next(a for a in manifest["assertions"] if a["label"] == "org.stegmark.watermark")

    assert wm["data"]["engine"] == "native"
    assert wm["data"]["payload_hex"] == "deadbeef"
    assert wm["data"]["strength"] == 1.2


def test_to_json_is_parseable():
    builder = C2PAManifestBuilder().add_action("c2pa.created").add_creator("Alice", "did:example:alice")

    text = builder.to_json()
    parsed = json.loads(text)

    assert parsed["instance_id"].startswith("urn:uuid:")


def test_signed_manifest_verifies():
    key = b"secret-key-32bytes-long-padding!!"
    builder = (
        C2PAManifestBuilder()
        .add_action("c2pa.created")
        .add_watermark_assertion("native", "cafebabe", 1.0)
    )
    manifest = builder.sign(secret_key=key)

    assert C2PAManifestBuilder.verify(manifest, secret_key=key) is True


def test_tampered_assertion_fails_verification():
    key = b"secret-key"
    builder = C2PAManifestBuilder().add_action("c2pa.created")
    manifest = builder.sign(secret_key=key)

    manifest["assertions"][0]["data"]["actions"][0]["action"] = "c2pa.edited"

    assert C2PAManifestBuilder.verify(manifest, secret_key=key) is False


def test_tampered_instance_id_fails_verification():
    key = b"secret-key"
    manifest = C2PAManifestBuilder().add_creator("A", "did:1").sign(secret_key=key)

    manifest["instance_id"] = "urn:uuid:00000000-0000-0000-0000-000000000000"

    assert C2PAManifestBuilder.verify(manifest, secret_key=key) is False


def test_unsigned_manifest_still_chains_hashes():
    builder = C2PAManifestBuilder().add_action("c2pa.created").add_action("c2pa.edited")
    manifest = builder.sign()

    assert C2PAManifestBuilder.verify(manifest) is True
    manifest["claim_generator"] = "MaliciousTool"
    assert C2PAManifestBuilder.verify(manifest) is False


def test_ai_generated_assertion_marks_training_disallowed():
    manifest = (
        C2PAManifestBuilder()
        .add_ai_generated(model="stable-diffusion-3", prompt="cat astronaut")
        .sign()
    )
    ai = next(a for a in manifest["assertions"] if a["label"] == "c2pa.training-mining")

    assert ai["data"]["training_mining"] == "notAllowed"
    assert ai["data"]["model"] == "stable-diffusion-3"
