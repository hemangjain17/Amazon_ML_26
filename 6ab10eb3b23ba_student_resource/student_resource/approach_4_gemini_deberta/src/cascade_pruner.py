import re
import logging
import pandas as pd
from typing import List, Dict, Tuple, Optional

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    HAS_RAPIDFUZZ = False

logger = logging.getLogger("CascadePruner")


def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """Computes token-level Jaccard similarity between two strings."""
    tokens1 = set(re.findall(r"\w+", text1.lower()))
    tokens2 = set(re.findall(r"\w+", text2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / float(len(union))


def compute_string_similarity(text1: str, text2: str) -> float:
    """Computes string similarity score [0.0, 1.0]. Uses rapidfuzz if available."""
    if not text1 or not text2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return fuzz.token_sort_ratio(text1, text2) / 100.0
    else:
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def prune_candidate_list(
    s1_text: str,
    candidates_with_scores: List[Tuple[str, str, float]],  # list of (cand_id, cand_text, gemini_score)
    top_k_out: int = 15,
    w_gemini: float = 0.60,
    w_jaccard: float = 0.20,
    w_levenshtein: float = 0.20,
) -> List[Tuple[str, str, float, float]]:
    """
    Ranks top-50 candidates using a multi-feature composite cascade score and prunes to top_k_out (e.g. 15).
    Returns list of (cand_id, cand_text, gemini_score, composite_score) sorted descending by composite_score.
    """
    if not candidates_with_scores:
        return []

    scored_cands = []
    for cand_id, cand_text, gemini_score in candidates_with_scores:
        # Scale cosine similarity from [-1, 1] or [0, 1] to [0, 1] range safely
        norm_gemini = max(0.0, min(1.0, (gemini_score + 1.0) / 2.0 if gemini_score < 0 else gemini_score))

        jaccard_sim = compute_jaccard_similarity(s1_text, cand_text)
        lev_sim = compute_string_similarity(s1_text, cand_text)

        composite = (w_gemini * norm_gemini) + (w_jaccard * jaccard_sim) + (w_levenshtein * lev_sim)
        scored_cands.append((cand_id, cand_text, gemini_score, composite))

    # Sort descending by composite score
    scored_cands.sort(key=lambda x: x[3], reverse=True)
    return scored_cands[:top_k_out]


def run_cascade_pruning(
    s1_df: pd.DataFrame,
    s2_s3_df: pd.DataFrame,
    retrieval_map: Dict[str, List[Tuple[str, float]]],
    top_k_pruned: int = 15,
    w_gemini: float = 0.60,
    w_jaccard: float = 0.20,
    w_levenshtein: float = 0.20,
) -> Dict[str, List[Tuple[str, str, float, float]]]:
    """
    Applies candidate pruning across all S1 entities.
    Returns map: s1_id -> list of (cand_id, cand_text, gemini_score, composite_score).
    """
    from dataset_loader import format_entity_text, clean_str

    logger.info(f"Running Cascade Pruner: Pruning candidates from Top-50 down to Top-{top_k_pruned}...")

    # Build ID to text mapping for fast lookup
    s1_texts = format_entity_text(s1_df)
    s1_id_col = "entity_id" if "entity_id" in s1_df.columns else "id"
    s1_text_map = {clean_str(row[s1_id_col]): txt for row, txt in zip(s1_df.to_dict("records"), s1_texts) if clean_str(row.get(s1_id_col))}

    cand_texts = format_entity_text(s2_s3_df)
    cand_id_col = "entity_id" if "entity_id" in s2_s3_df.columns else "id"
    cand_text_map = {clean_str(row[cand_id_col]): txt for row, txt in zip(s2_s3_df.to_dict("records"), cand_texts) if clean_str(row.get(cand_id_col))}

    pruned_map: Dict[str, List[Tuple[str, str, float, float]]] = {}

    for s1_id, cand_list in retrieval_map.items():
        s1_txt = s1_text_map.get(s1_id, "")
        cand_tuples = []
        for cand_id, gemini_score in cand_list:
            c_txt = cand_text_map.get(cand_id, "")
            cand_tuples.append((cand_id, c_txt, gemini_score))

        pruned = prune_candidate_list(
            s1_text=s1_txt,
            candidates_with_scores=cand_tuples,
            top_k_out=top_k_pruned,
            w_gemini=w_gemini,
            w_jaccard=w_jaccard,
            w_levenshtein=w_levenshtein,
        )
        pruned_map[s1_id] = pruned

    logger.info(f"Cascade pruning complete across {len(pruned_map)} S1 entities.")
    return pruned_map
