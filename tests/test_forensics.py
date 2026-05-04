from __future__ import annotations
import tempfile, numpy as np
from pathlib import Path
from PIL import Image
from stegmark.evaluation.forensics import generate_diff_heatmap, generate_dct_modification_map, generate_frequency_analysis, generate_full_report

def _img(h,w): return np.random.default_rng(42).integers(30,220,(h,w,3)).astype(np.uint8)
def _mod(img,d=5): return np.clip(img.astype(np.int32)+d,0,255).astype(np.uint8)

class TestDiffHeatmap:
    def test_size(self):
        img=_img(32,32); h=generate_diff_heatmap(img,_mod(img,5),amplify=10.0); assert h.size==(32,32)
    def test_identical(self):
        img=np.full((32,32,3),128,dtype=np.uint8); h=generate_diff_heatmap(img,img); assert abs(np.asarray(h).mean()-128)<2

class TestDCTMap:
    def test_size(self):
        img=_img(128,128); m=generate_dct_modification_map(img,_mod(img,10)); assert m.size==(128,128)

class TestFreq:
    def test_valid(self):
        img=_img(128,128); f=generate_frequency_analysis(img); arr=np.asarray(f); assert arr.min()>=0 and arr.max()<=255

class TestFullReport:
    def test_four_files(self):
        img=_img(128,128); mod=_mod(img,5)
        with tempfile.TemporaryDirectory() as d:
            op=Path(d)/"orig.png"; wp=Path(d)/"wm.png"
            Image.fromarray(img).save(op); Image.fromarray(mod).save(wp)
            od=Path(d)/"report"; files=generate_full_report(op,wp,od)
            assert len(files)==4; assert all(f.exists() and f.stat().st_size>0 for f in files)
