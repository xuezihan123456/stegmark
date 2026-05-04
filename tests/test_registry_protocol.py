from __future__ import annotations

import sqlite3

import numpy as np

from stegmark.core.registry_protocol import WatermarkRegistration, compute_image_hash, generate_watermark_id


class TestRegistration:
    def test_roundtrip(self):
        r=WatermarkRegistration(watermark_id="abc",image_hash="ih",message_hash="mh",timestamp="2026-01-01",engine="native",extra={"k":"v"})
        lr=WatermarkRegistration.from_dict(r.to_dict()); assert lr.watermark_id=="abc" and lr.extra["k"]=="v"

class TestHash:
    def test_same(self):
        i1=np.full((16,16,3),128,dtype=np.uint8); i2=np.full((16,16,3),128,dtype=np.uint8)
        assert compute_image_hash(i1)==compute_image_hash(i2)
    def test_different(self):
        assert compute_image_hash(np.full((16,16,3),128,dtype=np.uint8))!=compute_image_hash(np.full((16,16,3),200,dtype=np.uint8))

class TestSQLite:
    def test_crud(self):
        conn=sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE IF NOT EXISTS registrations (watermark_id TEXT PRIMARY KEY, image_hash TEXT, message_hash TEXT, timestamp TEXT, engine TEXT, extra_json TEXT DEFAULT '{}')")
        conn.execute("INSERT INTO registrations VALUES (?,?,?,?,?,?)",("id1","img1","msg1","2026-01-01","native","{}"))
        conn.commit()
        row=conn.execute("SELECT * FROM registrations WHERE watermark_id=?",("id1",)).fetchone(); assert row is not None and row[0]=="id1"
        conn.close()

class TestWatermarkID:
    def test_deterministic(self):
        assert generate_watermark_id("ih","mh","2026-01-01")==generate_watermark_id("ih","mh","2026-01-01")
        assert len(generate_watermark_id("a","b","c"))==16
