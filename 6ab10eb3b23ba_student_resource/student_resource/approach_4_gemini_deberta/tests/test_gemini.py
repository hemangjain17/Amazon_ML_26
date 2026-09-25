import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gemini_retriever import (
    mock_embed_batch,
    batch_embed_texts,
    search_faiss_candidates,
)


class TestGeminiRetriever(unittest.TestCase):

    def test_mock_embed_batch(self):
        texts = ["Entity 1", "Entity 2", "Entity 3"]
        embeddings = mock_embed_batch(texts, dim=768)

        self.assertEqual(embeddings.shape, (3, 768))
        self.assertEqual(embeddings.dtype, np.float32)

        # Check normalization (L2 norm ~ 1.0)
        norms = np.linalg.norm(embeddings, axis=1)
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=4)

    def test_batch_embed_mock_mode(self):
        texts = ["Name: Acme | Address: Seattle", "Name: Beta | Address: Paris"]
        embeddings = batch_embed_texts(
            texts=texts,
            api_key="",
            mock_mode=True,
        )
        self.assertEqual(embeddings.shape, (2, 768))

    def test_search_faiss_candidates(self):
        q_texts = ["Query 1", "Query 2"]
        c_texts = ["Cand A", "Cand B", "Cand C"]

        q_emb = mock_embed_batch(q_texts, dim=128)
        c_emb = mock_embed_batch(c_texts, dim=128)

        q_ids = ["q1", "q2"]
        c_ids = ["cA", "cB", "cC"]

        results = search_faiss_candidates(
            query_embeddings=q_emb,
            candidate_embeddings=c_emb,
            query_ids=q_ids,
            candidate_ids=c_ids,
            top_k=2,
        )

        self.assertIn("q1", results)
        self.assertIn("q2", results)
        self.assertEqual(len(results["q1"]), 2)
        self.assertEqual(len(results["q2"]), 2)

        # Ensure candidate tuple format (cand_id, score)
        cand_id, score = results["q1"][0]
        self.assertIn(cand_id, c_ids)
        self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
