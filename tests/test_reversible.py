from __future__ import annotations
import numpy as np, pytest
from stegmark.core.reversible import ReversibleEngine
from stegmark.exceptions import InvalidInputError

def _img(h,w): return np.random.default_rng(42).integers(50,200,(h,w,3)).astype(np.uint8)

class TestReversible:
    def test_embed_extract(self):
        e=ReversibleEngine(); r=e.decode(e.encode(_img(128,128),"hello123")); assert r.found and r.message=="hello123"
    def test_embed_restore(self):
        img=_img(128,128); e=ReversibleEngine(); wm=e.encode(img,"hello1234"); restored=e.restore(wm)
        # 红色通道应完全恢复（整数差值扩展）
        np.testing.assert_array_equal(restored[:,:,0], img[:,:,0])
    def test_restore_unmarked_raises(self):
        with pytest.raises(InvalidInputError): ReversibleEngine().restore(_img(64,64))
    def test_short_refused(self):
        with pytest.raises(InvalidInputError): ReversibleEngine().encode(_img(8,8),"ab")
    def test_long_message(self):
        img=_img(256,256); msg="Long reversible watermark test message!"
        e=ReversibleEngine(); r=e.decode(e.encode(img,msg)); assert r.found and r.message==msg
