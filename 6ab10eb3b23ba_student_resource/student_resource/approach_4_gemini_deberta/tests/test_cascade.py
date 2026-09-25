import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cascade_pruner import (
    compute_jaccard_similarity,
    compute_string_similarity,
    prune_candidate_list,
)


class TestCascadePruner(unittest.TestCase):

    def test_compute_jaccard_similarity(self):
        s1 = "Acme Corporation Seattle WA"
        s2 = "Acme Corp Seattle Washington"
        s3 = "Totally Different Industry"

        sim1 = compute_jaccard_similarity(s1, s2)
        sim2 = compute_jaccard_similarity(s1, s3)

        self.assertGreater(sim1, sim2)
        self.assertGreater(sim1, 0.2)

    def test_compute_string_similarity(self):
        s1 = "Acme Corp"
        s2 = "Acme Corp"
        s3 = "Zeta Inc"

        sim_identical = compute_string_similarity(s1, s2)
        sim_diff = compute_string_similarity(s1, s3)

        self.assertAlmostEqual(sim_identical, 1.0, places=2)
        self.assertLess(sim_diff, 0.5)

    def test_prune_candidate_list(self):
        s1_text = "Name: Acme Corp | Address: 123 Main St, Seattle"
        candidates = [
            ("c1", "Name: Acme Corp | Address: 123 Main St, Seattle", 0.95),
            ("c2", "Name: Acme Inc | Address: Main St, Seattle", 0.85),
            ("c3", "Name: Random Store | Address: Paris France", 0.20),
            ("c4", "Name: Acme Logistics | Address: Seattle", 0.70),
        ]

        pruned = prune_candidate_list(
            s1_text=s1_text,
            candidates_with_scores=candidates,
            top_k_out=2,
        )

        self.assertEqual(len(pruned), 2)
        # Check that best matching candidate c1 is ranked first
        top_cand_id = pruned[0][0]
        self.assertEqual(top_cand_id, "c1")


if __name__ == "__main__":
    unittest.main()
