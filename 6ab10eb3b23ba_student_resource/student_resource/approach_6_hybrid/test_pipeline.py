from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from approach_5_b5.run_pipeline import enrich
from approach_6_hybrid.run_pipeline import build_hybrid_features, parse_devices, record_text


class FakeEncoder:
    def encode(self, texts, description):
        rows = []
        for text in texts:
            value = float(sum(text.encode("utf-8")) % 997) / 997.0
            vector = np.zeros(384, dtype=np.float32)
            vector[0] = value
            vector[1] = 1.0 - value
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)


class Approach6Tests(unittest.TestCase):
    def setUp(self):
        self.s1 = enrich(pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Alpha Ltd", "business_address": "12 Main Road", "country": "India"},
            {"entity_id": "S1-2", "business_name": "Beta Co", "business_address": "20 Park Road", "country": "India"},
        ]))
        self.pool = enrich(pd.DataFrame([
            {"entity_id": "S2-1", "business_name": "Alpha Limited", "business_address": "12 Main Road", "country": "India"},
            {"entity_id": "S2-2", "business_name": "Beta Company", "business_address": "20 Park Road", "country": "India"},
        ]))

    def test_cpu_device_fallback_shape(self):
        devices = parse_devices("0,1")
        self.assertTrue(devices == ["cpu"] or all(device.startswith("cuda:") for device in devices))

    def test_record_text_has_e5_prefix_and_normalized_fields(self):
        text = record_text(self.s1.iloc[0], "query")
        self.assertTrue(text.startswith("query:"))
        self.assertIn("alpha ltd", text)
        self.assertIn("india", text)

    def test_hybrid_features_add_one_cosine_column(self):
        pairs = [("S1-1", 0), ("S1-2", 1), ("S1-1", 1)]
        matrix = build_hybrid_features(self.s1, self.pool, pairs, FakeEncoder(), "unit")
        self.assertEqual(matrix.shape, (3, 12))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertGreaterEqual(float(matrix[:, -1].min()), -1.0)
        self.assertLessEqual(float(matrix[:, -1].max()), 1.0)


if __name__ == "__main__":
    unittest.main()
