from __future__ import annotations

import numpy as np

from stegmark.core.aigc_fingerprint import AIGCMetadata, compute_prompt_hash, extract_aigc_metadata, stamp_image
from stegmark.core.native import NativeEngine


def _img(h,w): return np.random.default_rng(42).integers(30,220,(h,w,3)).astype(np.uint8)

class TestMetadata:
    def test_roundtrip(self):
        m=AIGCMetadata(generator="sd-xl",seed=42,prompt_hash="abc12345")
        lm=AIGCMetadata.from_json(m.to_json()); assert lm.generator=="sd-xl" and lm.seed==42
    def test_minimal(self):
        m=AIGCMetadata(generator="sd"); lm=AIGCMetadata.from_json(m.to_json()); assert lm.generator=="sd" and lm.seed is None
    def test_c2pa(self):
        m=AIGCMetadata(generator="dalle3",seed=123); c=m.to_c2pa_manifest(); assert c["assertions"][0]["data"]["generator"]=="dalle3"
    def test_custom(self):
        m=AIGCMetadata(generator="mj",custom={"style":"photo"}); lm=AIGCMetadata.from_json(m.to_json()); assert lm.custom["style"]=="photo"

class TestHash:
    def test_deterministic(self):
        assert compute_prompt_hash("a cat")==compute_prompt_hash("a cat")
    def test_different(self):
        assert compute_prompt_hash("a cat")!=compute_prompt_hash("a dog")

class TestStampExtract:
    def test_roundtrip(self):
        img=_img(256,256); e=NativeEngine(); m=AIGCMetadata(generator="test",seed=123,prompt_hash="abcdef01")
        wm=stamp_image(img,e,m); em=extract_aigc_metadata(wm,e); assert em is not None and em.generator=="test"
