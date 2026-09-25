import os
import sys
import unittest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deberta_cross_encoder import CrossEncoderPairDataset


class TestDebertaCrossEncoder(unittest.TestCase):

    def test_pair_dataset(self):
        pairs = [
            ("Name: Acme | Address: Seattle", "Name: Acme | Address: Seattle"),
            ("Name: Acme | Address: Seattle", "Name: Beta | Address: Paris"),
        ]
        labels = [1, 0]

        dataset = CrossEncoderPairDataset(pairs, labels)
        self.assertEqual(len(dataset), 2)

        sample = dataset[0]
        self.assertEqual(sample["s1_text"], pairs[0][0])
        self.assertEqual(sample["cand_text"], pairs[0][1])
        self.assertEqual(sample["label"], 1)


if __name__ == "__main__":
    unittest.main()
