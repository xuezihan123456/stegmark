"""C2PA-compatible Content Credentials manifest builder.

This is a minimal-compat implementation that emits JSON manifests with
the same structural fields as a C2PA manifest store (claim_generator,
instance_id, assertions, hash chain, signature) but without depending on
the official ``c2pa-python`` SDK. Use it for prototyping and tests;
production deployments should sign with X.509 + COSE per the C2PA spec.

The hash chain is computed as follows:

  * Each assertion's ``hash`` field = ``sha256(canonical_json(data))``.
  * The claim's ``hash`` field = ``sha256(claim_generator || instance_id ||
                                 sorted_assertion_hashes || signature)``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Self


def _canonical_json(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class C2PAAssertion:
    label: str
    data: dict[str, object]
    hash: str = ""

    def with_hash(self) -> C2PAAssertion:
        return C2PAAssertion(label=self.label, data=self.data, hash=_sha256_hex(_canonical_json(self.data)))


@dataclass
class C2PAManifestBuilder:
    claim_generator: str = "StegMark/0.4"
    claim_generator_info: dict[str, str] = field(default_factory=lambda: {"name": "StegMark", "version": "0.4"})
    assertions: list[C2PAAssertion] = field(default_factory=list)
    instance_id: str = field(default_factory=lambda: f"urn:uuid:{uuid.uuid4()}")

    def add_action(self, action: str, when: str | None = None) -> Self:
        timestamp = when or datetime.now(timezone.utc).isoformat()
        assertion = C2PAAssertion(
            label="c2pa.actions",
            data={"actions": [{"action": action, "when": timestamp}]},
        ).with_hash()
        self.assertions.append(assertion)
        return self

    def add_creator(self, name: str, identifier: str) -> Self:
        assertion = C2PAAssertion(
            label="stds.schema-org.CreativeWork",
            data={"@context": "https://schema.org", "@type": "CreativeWork", "author": {"name": name, "identifier": identifier}},
        ).with_hash()
        self.assertions.append(assertion)
        return self

    def add_watermark_assertion(self, engine: str, payload_hex: str, strength: float) -> Self:
        assertion = C2PAAssertion(
            label="org.stegmark.watermark",
            data={"engine": engine, "payload_hex": payload_hex, "strength": float(strength)},
        ).with_hash()
        self.assertions.append(assertion)
        return self

    def add_ai_generated(self, model: str, prompt: str | None = None) -> Self:
        data: dict[str, object] = {"model": model, "training_mining": "notAllowed"}
        if prompt is not None:
            data["prompt"] = prompt
        assertion = C2PAAssertion(label="c2pa.training-mining", data=data).with_hash()
        self.assertions.append(assertion)
        return self

    def sign(self, secret_key: bytes | None = None) -> dict[str, object]:
        assertion_dicts = [
            {"label": a.label, "data": a.data, "hash": a.hash} for a in self.assertions
        ]
        sorted_hashes = sorted(a.hash for a in self.assertions)
        if secret_key is not None:
            sig_payload = self.claim_generator + self.instance_id + "".join(sorted_hashes)
            signature = hmac.new(secret_key, sig_payload.encode("utf-8"), hashlib.sha256).hexdigest()
            sig_algo = "HMAC-SHA256"
        else:
            signature = ""
            sig_algo = "none"

        claim_hash = _sha256_hex(
            (self.claim_generator + self.instance_id + "".join(sorted_hashes) + signature).encode("utf-8")
        )
        return {
            "claim_generator": self.claim_generator,
            "claim_generator_info": self.claim_generator_info,
            "instance_id": self.instance_id,
            "assertions": assertion_dicts,
            "signature": {"alg": sig_algo, "value": signature},
            "hash": claim_hash,
        }

    def to_json(self, secret_key: bytes | None = None, indent: int = 2) -> str:
        return json.dumps(self.sign(secret_key), indent=indent, ensure_ascii=False)

    @staticmethod
    def verify(manifest: dict[str, object], secret_key: bytes | None = None) -> bool:
        try:
            assertions = list(manifest["assertions"])  # type: ignore[arg-type]
            for a in assertions:
                expected = _sha256_hex(_canonical_json(a["data"]))
                if a["hash"] != expected:
                    return False
            sorted_hashes = sorted(a["hash"] for a in assertions)
            signature_block = manifest["signature"]  # type: ignore[index]
            signature = signature_block["value"]  # type: ignore[index]
            instance_id = manifest["instance_id"]
            claim_generator = manifest["claim_generator"]
            if signature and secret_key is not None:
                sig_payload = str(claim_generator) + str(instance_id) + "".join(sorted_hashes)
                expected_sig = hmac.new(secret_key, sig_payload.encode("utf-8"), hashlib.sha256).hexdigest()
                if not hmac.compare_digest(expected_sig, str(signature)):
                    return False
            expected_claim_hash = _sha256_hex(
                (str(claim_generator) + str(instance_id) + "".join(sorted_hashes) + str(signature)).encode("utf-8")
            )
            return manifest["hash"] == expected_claim_hash
        except (KeyError, TypeError):
            return False


__all__ = ["C2PAAssertion", "C2PAManifestBuilder"]
