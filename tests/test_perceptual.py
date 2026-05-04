from __future__ import annotations

import numpy as np

from stegmark.core.native import NativeEngine
from stegmark.core.perceptual_strength import adaptive_delta, compute_jnd_map
from stegmark.evaluation.metrics import compute_psnr


def _img(h,w,v=128): return np.full((h,w,3),v,dtype=np.uint8)
def _to_y(img): return (0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2]).astype(np.float32)

class TestJND:
    def test_uniform(self):
        jnd=compute_jnd_map(_to_y(_img(64,64))); assert jnd.max()<1e-6
    def test_textured(self):
        img=_img(64,64); img[:,32:,:]=np.random.default_rng(42).integers(0,256,(64,32,3)).astype(np.uint8)
        y=_to_y(img); jnd=compute_jnd_map(y); m=jnd.shape[1]//2; assert jnd[:,m:].mean()>jnd[:,:m].mean()*1.1
    def test_small(self): assert compute_jnd_map(_to_y(_img(4,4))).size==0

class TestDelta:
    def test_range(self):
        d=adaptive_delta(np.random.default_rng(42).random((8,8)).astype(np.float32),np.array([1,0]*32),12.0,1.0)
        assert d.min()>=6.0 and d.max()<=22.0

class TestNative:
    def test_roundtrip(self):
        e=NativeEngine(); r=e.decode(e.encode(_img(128,128),"test")); assert r.found and r.message=="test"
    def test_psnr(self):
        img=_img(128,128); assert compute_psnr(img,NativeEngine().encode(img,"hello"))>30
