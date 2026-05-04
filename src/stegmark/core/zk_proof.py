from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class ZKCommitment:
    commitment: str
    salt: str
    engine: str
    def to_json(self) -> str:
        import json
        return json.dumps({"commitment": self.commitment, "salt": self.salt, "engine": self.engine})


@dataclass(frozen=True)
class ZKProof:
    commitment: str
    salt_hash: str
    message_hash: str
    proof_data: dict[str,str]
    def to_json(self) -> str:
        import json
        return json.dumps({
            "commitment": self.commitment,
            "salt_hash": self.salt_hash,
            "message_hash": self.message_hash,
            "proof_data": self.proof_data,
        })

    @classmethod
    def from_json(cls, json_str: str) -> ZKProof:
        import json
        d = json.loads(json_str)
        return cls(
            commitment=d["commitment"],
            salt_hash=d["salt_hash"],
            message_hash=d["message_hash"],
            proof_data=d.get("proof_data", {}),
        )


def generate_salt() -> str: return secrets.token_hex(32)
def compute_commitment(message: str, salt: str) -> str: return hashlib.sha256(f"{message}:{salt}".encode()).hexdigest()


def generate_zk_commitment(message: str, engine_name: str) -> ZKCommitment:
    salt = generate_salt()
    return ZKCommitment(commitment=compute_commitment(message, salt), salt=salt, engine=engine_name)


def embed_with_zk(image, engine, message, *, strength=1.0):
    watermarked = engine.encode(image, message, strength=strength)
    return watermarked, generate_zk_commitment(message, engine.name)


def prove_ownership(image, engine, message, commitment):
    result = engine.decode(image)
    if not result.found or result.message is None or result.message != message: return None
    if compute_commitment(message, commitment.salt) != commitment.commitment: return None
    return ZKProof(commitment=commitment.commitment, salt_hash=hashlib.sha256(commitment.salt.encode("utf-8")).hexdigest(), message_hash=hashlib.sha256(message.encode("utf-8")).hexdigest(), proof_data={"engine":engine.name,"commitment_match":"true"})


def verify_zk_proof(proof, *, salt=None, message=None):
    if not proof.commitment or not proof.salt_hash or not proof.message_hash: return False
    if proof.proof_data.get("commitment_match") != "true": return False
    if salt is not None and message is not None and compute_commitment(message, salt) != proof.commitment: return False
    if salt is not None and hashlib.sha256(salt.encode("utf-8")).hexdigest() != proof.salt_hash: return False
    if message is not None and hashlib.sha256(message.encode("utf-8")).hexdigest() != proof.message_hash: return False
    return True


__all__ = ["ZKCommitment","ZKProof","generate_salt","compute_commitment","generate_zk_commitment","embed_with_zk","prove_ownership","verify_zk_proof"]
