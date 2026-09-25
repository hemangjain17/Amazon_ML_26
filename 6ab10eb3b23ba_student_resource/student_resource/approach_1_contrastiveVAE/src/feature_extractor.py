import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm

try:
    from approach_1_contrastiveVAE.src.normalization import clean_text, extract_pin_code
except (ImportError, ValueError):
    try:
        from src.normalization import clean_text, extract_pin_code
    except (ImportError, ValueError):
        from .normalization import clean_text, extract_pin_code


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Computes Jaro-Winkler similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    max_dist = (max(len1, len2) // 2) - 1
    if max_dist < 0:
        max_dist = 0

    match1 = [False] * len1
    match2 = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if match2[j]:
                continue
            if s1[i] != s2[j]:
                continue
            match1[i] = True
            match2[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not match1[i]:
            continue
        while not match2[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3.0

    # Prefix scale
    prefix = 0
    for i in range(min(4, min(len1, len2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1.0 - jaro)


def token_jaccard(s1: str, s2: str) -> float:
    """Computes token set Jaccard similarity."""
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Computes character n-gram Jaccard similarity."""
    if len(s1) < n or len(s2) < n:
        return token_jaccard(s1, s2)
    ngrams1 = set([s1[i:i+n] for i in range(len(s1) - n + 1)])
    ngrams2 = set([s2[i:i+n] for i in range(len(s2) - n + 1)])
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    return len(intersection) / len(union)


def evaluate_pin_match(pin1: str, pin2: str) -> int:
    """Returns 1 for match, 0 for mismatch, -1 if either is missing."""
    if not pin1 or not pin2:
        return -1
    return 1 if pin1 == pin2 else 0


def extract_pair_features(
    s1_row: dict,
    cand_row: dict,
    latent_cosine_sim: float = 0.0,
) -> Dict[str, float]:
    """Extracts fine-grained pairwise string, phonetic, PIN, and latent features."""
    clean_n1 = clean_text(s1_row.get("name", ""))
    clean_n2 = clean_text(cand_row.get("name", ""))

    clean_a1 = clean_text(s1_row.get("address", ""))
    clean_a2 = clean_text(cand_row.get("address", ""))

    pin1 = extract_pin_code(s1_row.get("address", ""))
    pin2 = extract_pin_code(cand_row.get("address", ""))

    return {
        "name_jaro_winkler": jaro_winkler_similarity(clean_n1, clean_n2),
        "name_token_jaccard": token_jaccard(clean_n1, clean_n2),
        "name_3gram_jaccard": char_ngram_jaccard(clean_n1, clean_n2, n=3),
        "addr_token_jaccard": token_jaccard(clean_a1, clean_a2),
        "addr_3gram_jaccard": char_ngram_jaccard(clean_a1, clean_a2, n=3),
        "pin_code_match": float(evaluate_pin_match(pin1, pin2)),
        "latent_cosine_sim": float(latent_cosine_sim),
        "is_s2": 1.0 if cand_row.get("entity_id", "").startswith("S2-") else 0.0,
    }


def build_candidate_feature_matrix(
    candidate_map: Dict[str, List[str]],
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    s1_embeddings: np.ndarray = None,
    cand_embeddings: np.ndarray = None,
    s1_id_to_idx: Dict[str, int] = None,
    cand_id_to_idx: Dict[str, int] = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Constructs feature matrix for all (S1, Candidate) pairs."""
    print("Building lookup dictionaries for feature extraction...")
    s1_dict = s1_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s2_dict = s2_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s3_dict = s3_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")

    pair_list = []
    feature_rows = []

    print("Extracting features for candidate pairs...")
    for s1_id, candidates in tqdm(candidate_map.items(), desc="Pair Feature Extraction"):
        s1_info = s1_dict.get(s1_id, {})
        s1_row_dict = {"name": str(s1_info.get("business_name", "")), "address": str(s1_info.get("business_address", "")), "country": str(s1_info.get("country", ""))}
        s1_idx = s1_id_to_idx.get(s1_id) if s1_id_to_idx else None

        for cand_id in candidates:
            if cand_id.startswith("S2-"):
                cand_info = s2_dict.get(cand_id, {})
            else:
                cand_info = s3_dict.get(cand_id, {})

            cand_row_dict = {
                "entity_id": cand_id,
                "name": str(cand_info.get("business_name", "")),
                "address": str(cand_info.get("business_address", "")),
                "country": str(cand_info.get("country", "")),
            }

            # Calculate cosine similarity if embeddings are available
            latent_sim = 0.0
            cand_idx = cand_id_to_idx.get(cand_id) if cand_id_to_idx else None
            if s1_idx is not None and cand_idx is not None and s1_embeddings is not None and cand_embeddings is not None:
                vec_s1 = s1_embeddings[s1_idx]
                vec_cand = cand_embeddings[cand_idx]
                latent_sim = np.dot(vec_s1, vec_cand)

            features = extract_pair_features(s1_row_dict, cand_row_dict, latent_cosine_sim=latent_sim)
            feature_rows.append(features)
            pair_list.append((s1_id, cand_id))

    feature_df = pd.DataFrame(feature_rows)
    return feature_df, pair_list
