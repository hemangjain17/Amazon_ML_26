import os
import sys
import unittest
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset_loader import (
    clean_str,
    format_entity_text,
    build_ground_truth_map,
)


class TestDatasetLoader(unittest.TestCase):

    def test_clean_str(self):
        self.assertEqual(clean_str(" Hello "), "Hello")
        self.assertEqual(clean_str("NaN"), "")
        self.assertEqual(clean_str("None"), "")
        self.assertEqual(clean_str(None), "")
        self.assertEqual(clean_str(pd.NA), "")

    def test_format_entity_text(self):
        df = pd.DataFrame([
            {"name": "Acme Corp", "address": "123 Main St", "city": "Seattle", "postal_code": "98101", "country": "US"},
            {"name": "Beta LLC", "address": "", "city": "Paris", "postal_code": "75001", "country": "France"},
            {"name": "", "address": "", "city": "", "postal_code": "", "country": ""},
        ])
        formatted = format_entity_text(df)
        self.assertEqual(len(formatted), 3)
        self.assertIn("Name: Acme Corp", formatted[0])
        self.assertIn("Address: 123 Main St, Seattle, 98101, US", formatted[0])
        self.assertIn("Address: Paris, 75001, France", formatted[1])
        self.assertEqual(formatted[2], "Entity: Unknown")

    def test_build_ground_truth_map(self):
        gt_df = pd.DataFrame([
            {"s1_id": "s1_1", "s2_id": "s2_1", "s3_id": "s3_1"},
            {"s1_id": "s1_2", "s2_id": "s2_2", "s3_id": "None"},
            {"s1_id": "s1_3", "s2_id": "", "s3_id": ""},
        ])
        gt_map = build_ground_truth_map(gt_df)
        self.assertEqual(gt_map["s1_1"], ["s2_1", "s3_1"])
        self.assertEqual(gt_map["s1_2"], ["s2_2"])
        self.assertEqual(gt_map["s1_3"], [])


if __name__ == "__main__":
    unittest.main()
