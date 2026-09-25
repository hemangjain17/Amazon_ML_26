import os
import time
import logging
import concurrent.futures
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("GeminiRetriever")


def mock_embed_batch(texts: List[str], dim: int = 768) -> np.ndarray:
    """Deterministic hash-based mock embedding for testing/offline mode."""
    embeddings = []
    for t in texts:
        # Create deterministic pseudo-random vector based on string hash
        rng = np.random.RandomState(abs(hash(t)) % (2**32 - 1))
        vec = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-9
        embeddings.append(vec / norm)
    return np.vstack(embeddings)


def call_gemini_api(texts: List[str], api_key: str, model_name: str = "models/text-embedding-004") -> np.ndarray:
    """Calls Google Gemini embed_content API with retries."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = genai.embed_content(
                model=model_name,
                content=texts,
                task_type="retrieval_document",
            )
            embeddings = np.array(res["embedding"], dtype=np.float32)
            return embeddings
        except Exception as e:
            logger.warning(f"Gemini API call attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)

    raise RuntimeError("Gemini API call failed after max retries")


def batch_embed_texts(
    texts: List[str],
    api_key: str,
    model_name: str = "models/text-embedding-004",
    batch_size: int = 100,
    max_workers: int = 8,
    mock_mode: bool = False,
) -> np.ndarray:
    """Embeds list of text strings in parallel batches, with cache and mock mode support."""
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    if mock_mode or not api_key or api_key == "your_gemini_api_key_here":
        logger.info(f"Running Gemini embedder in MOCK/OFFLINE mode ({len(texts)} items)...")
        return mock_embed_batch(texts)

    logger.info(f"Embedding {len(texts)} texts via Gemini API ({model_name})...")
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results = [None] * len(batches)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(call_gemini_api, batch, api_key, model_name): idx
            for idx, batch in enumerate(batches)
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Failed batch {idx}: {e}. Falling back to single-item sequential...")
                # Fallback item-by-item for that batch
                batch = batches[idx]
                item_vecs = []
                for item in batch:
                    item_vecs.append(call_gemini_api([item], api_key, model_name))
                results[idx] = np.vstack(item_vecs)

    return np.vstack(results)


def get_or_compute_embeddings(
    cache_path: str,
    texts: List[str],
    api_key: str,
    model_name: str = "models/text-embedding-004",
    batch_size: int = 100,
    max_workers: int = 8,
    mock_mode: bool = False,
) -> np.ndarray:
    """Loads embeddings from cache_path if present, otherwise computes and saves them."""
    if os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}...")
        embeddings = np.load(cache_path)
        if len(embeddings) == len(texts):
            return embeddings
        logger.warning(f"Cache size mismatch ({len(embeddings)} vs {len(texts)}). Re-computing...")

    embeddings = batch_embed_texts(
        texts=texts,
        api_key=api_key,
        model_name=model_name,
        batch_size=batch_size,
        max_workers=max_workers,
        mock_mode=mock_mode,
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    logger.info(f"Saved computed embeddings to {cache_path}")
    return embeddings


def search_faiss_candidates(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    query_ids: List[str],
    candidate_ids: List[str],
    top_k: int = 50,
) -> Dict[str, List[Tuple[str, float]]]:
    """Uses FAISS Inner Product (Cosine Similarity after L2 norm) to search top_k candidates."""
    if len(query_embeddings) == 0 or len(candidate_embeddings) == 0:
        return {}

    q_vecs = query_embeddings.copy()
    c_vecs = candidate_embeddings.copy()

    # Normalize vectors for Cosine Similarity
    faiss.normalize_L2(q_vecs)
    faiss.normalize_L2(c_vecs)

    dim = q_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(c_vecs)

    top_k = min(top_k, len(candidate_ids))
    scores, indices = index.search(q_vecs, top_k)

    retrieval_map: Dict[str, List[Tuple[str, float]]] = {}
    for q_idx, q_id in enumerate(query_ids):
        cands = []
        for rank in range(top_k):
            c_idx = indices[q_idx, rank]
            if c_idx < 0 or c_idx >= len(candidate_ids):
                continue
            cand_id = candidate_ids[c_idx]
            score = float(scores[q_idx, rank])
            cands.append((cand_id, score))
        retrieval_map[q_id] = cands

    return retrieval_map
