from __future__ import annotations

import numpy as np

from stegmark.wasm.stegmark_wasm import _decode_frame_wasm, _encode_text_wasm, embed_image, extract_message


def _px(h,w):
    rng=np.random.default_rng(42); arr=rng.integers(30,220,(h,w,3)).astype(np.uint8)
    return [[[int(arr[r,c,ch]) for ch in range(3)] for c in range(w)] for r in range(h)]

class TestCodec:
    def test_encode_decode(self):
        bits=_encode_text_wasm("hello"); v,msg,err=_decode_frame_wasm(bits); assert v and msg=="hello"
    def test_corrupted(self):
        bits=_encode_text_wasm("hello"); corr=bits[:8]+[1-b for b in bits[8:16]]+bits[16:]; v,_,_=_decode_frame_wasm(corr); assert not v
    def test_too_short(self): v,_,err=_decode_frame_wasm([0,1,0]); assert not v and err=="too_short"

class TestEmbedExtract:
    def test_roundtrip(self):
        px=_px(256,256); assert extract_message(embed_image(px,"wasm_test"))=="wasm_test"
    def test_long_msg(self):
        px=_px(384,384); msg="WASM compatible watermark test!"; assert extract_message(embed_image(px,msg))==msg
    def test_extract_unmarked(self): assert extract_message(_px(128,128)) is None
    def test_output_dims(self):
        px=_px(128,128); wm=embed_image(px,"dim"); assert len(wm)==128 and len(wm[0])==128 and len(wm[0][0])==3
    def test_small_image(self):
        px=_px(4,4); wm=embed_image(px,"test"); assert len(wm)==4
