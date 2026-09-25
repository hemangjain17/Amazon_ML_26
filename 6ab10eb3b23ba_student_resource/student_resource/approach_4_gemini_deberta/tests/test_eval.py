import os
import sys
import tempfile
import unittest
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval_optimizer import (
    compute_macro_f_beta,
    optimize_f05_threshold,
    generate_submission_files,
    validate_submission_files,
)


class TestEvalOptimizer(unittest.TestCase):

    def test_compute_macro_f_beta(self):
        s1_ids = ["s1_1", "s1_2"]
        gt_matches = {
            "s1_1": ["s2_1"],
            "s1_2": ["s3_2"],
        }

        # Perfect prediction
        pred_perfect = {
            "s1_1": ["s2_1"],
            "s1_2": ["s3_2"],
        }
        score_perfect = compute_macro_f_beta(s1_ids, pred_perfect, gt_matches, beta=0.5)
        self.assertAlmostEqual(score_perfect, 1.0, places=4)

        # Zero prediction
        pred_empty = {
            "s1_1": [],
            "s1_2": [],
        }
        score_empty = compute_macro_f_beta(s1_ids, pred_empty, gt_matches, beta=0.5)
        self.assertAlmostEqual(score_empty, 0.0, places=4)

    def test_optimize_f05_threshold(self):
        s1_ids = ["s1_1", "s1_2"]
        gt_matches = {
            "s1_1": ["s2_1"],
            "s1_2": ["s3_2"],
        }
        predictions = [
            ("s1_1", "s2_1", 0.92),
            ("s1_1", "s2_99", 0.40),
            ("s1_2", "s3_2", 0.88),
            ("s1_2", "s3_99", 0.30),
        ]

        best_thresh, best_score = optimize_f05_threshold(
            s1_ids=s1_ids,
            pair_predictions=predictions,
            ground_truth_matches=gt_matches,
            min_thresh=0.50,
            max_thresh=0.90,
            step=0.05,
        )

        self.assertGreaterEqual(best_thresh, 0.50)
        self.assertAlmostEqual(best_score, 1.0, places=3)

    def test_generate_and_validate_submission_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            s1_ids = ["s1_1", "s1_2"]
            s2_ids_set = {"s2_1"}
            s3_ids_set = {"s3_2"}

            predictions = [
                ("s1_1", "s2_1", 0.95),
                ("s1_2", "s3_2", 0.88),
            ]

            matching_path, cand_pairs_path = generate_submission_files(
                s1_ids=s1_ids,
                pair_predictions=predictions,
                decision_threshold=0.80,
                output_dir=tmp_dir,
            )

            self.assertTrue(os.path.exists(matching_path))
            self.assertTrue(os.path.exists(cand_pairs_path))

            is_valid = validate_submission_files(matching_path, cand_pairs_path)
            self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
