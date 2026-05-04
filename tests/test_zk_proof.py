from __future__ import annotations
import numpy as np
from stegmark.core.zk_proof import (ZKCommitment,ZKProof,generate_salt,compute_commitment,generate_zk_commitment,embed_with_zk,prove_ownership,verify_zk_proof)
from stegmark.core.native import NativeEngine

def _img(h,w): return np.random.default_rng(42).integers(30,220,(h,w,3)).astype(np.uint8)

class TestCommitment:
    def test_generation(self):
        s=generate_salt(); assert len(s)==64; assert len(compute_commitment("hello",s))==64
    def test_deterministic(self):
        s="abcd"*16; assert compute_commitment("test",s)==compute_commitment("test",s)
    def test_different_msg(self):
        s="abcd"*16; assert compute_commitment("m1",s)!=compute_commitment("m2",s)
    def test_to_json(self):
        c=ZKCommitment("abc","def","native"); assert "abc" in c.to_json() and "native" in c.to_json()

class TestZKProof:
    def test_json_roundtrip(self):
        p=ZKProof("c","s","m",{"e":"n"}); lp=ZKProof.from_json(p.to_json()); assert lp.commitment=="c"

class TestZKWorkflow:
    def test_full(self):
        img=_img(128,128); engine=NativeEngine(); msg="secret_42"
        wm,c=embed_with_zk(img,engine,msg); assert wm.shape==img.shape
        p=prove_ownership(wm,engine,msg,c); assert p is not None
        assert verify_zk_proof(p); assert verify_zk_proof(p,salt=c.salt,message=msg)
    def test_wrong_message(self):
        img=_img(128,128); engine=NativeEngine(); wm,c=embed_with_zk(img,engine,"correct")
        assert prove_ownership(wm,engine,"wrong",c) is None
    def test_wrong_salt(self):
        img=_img(128,128); engine=NativeEngine(); wm,c=embed_with_zk(img,engine,"test")
        p=prove_ownership(wm,engine,"test",c); assert not verify_zk_proof(p,salt="wrong")
