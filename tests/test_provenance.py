from __future__ import annotations
import numpy as np, pytest
from stegmark.core.provenance import (ProvenanceEntry,ProvenanceChain,multi_layer_embed,multi_layer_extract,build_provenance_entry,MAX_PROVENANCE_LAYERS)
from stegmark.exceptions import MessageTooLongError

def _img(h,w): return np.random.default_rng(42).integers(30,220,(h,w,3)).astype(np.uint8)

class TestProvenanceEntry:
    def test_to_dict(self):
        e=ProvenanceEntry("a","created","2026-01-01","msg",0); d=e.to_dict(); assert d["operator"]=="a" and d["layer"]==0

class TestProvenanceChain:
    def test_empty(self):
        c=ProvenanceChain(); assert c.is_empty and c.depth==0
    def test_nonempty(self):
        e1=ProvenanceEntry("a","c","t1","m1",0); e2=ProvenanceEntry("b","d","t2","m2",1)
        assert ProvenanceChain(entries=(e1,e2)).depth==2

class TestMultiLayer:
    def test_single_roundtrip(self):
        img=_img(256,256); e=build_provenance_entry("a","created","orig",0)
        chain=multi_layer_extract(multi_layer_embed(img,[e])); assert chain.depth>=1
    def test_too_many_raises(self):
        img=_img(64,64); entries=[build_provenance_entry("u","a",f"m{i}",i) for i in range(MAX_PROVENANCE_LAYERS+1)]
        with pytest.raises(MessageTooLongError): multi_layer_embed(img,entries)
    def test_auto_timestamp(self):
        e=build_provenance_entry("t","a","hello",0); assert len(e.timestamp)>10
