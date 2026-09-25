import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger("EvalOptimizer")


def compute_macro_f_beta(
    s1_ids: List[str],
    predicted_matches: Dict[str, List[str]],
    ground_truth_matches: Dict[str, List[str]],
    beta: float = 0.5,
) -> float:
    """Computes Macro F_beta score (F_0.5 weighting Precision 2x over Recall)."""
    beta_sq = beta ** 2
    f_scores = []

    for s1_id in s1_ids:
        preds = set(predicted_matches.get(s1_id, []))
        targets = set(ground_truth_matches.get(s1_id, []))

        if not preds and not targets:
            f_scores.append(1.0)
            continue
        elif not preds or not targets:
            f_scores.append(0.0)
            continue

        tp = len(preds.intersection(targets))
        fp = len(preds - targets)
        fn = len(targets - preds)

        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f_scores.append(0.0)
        else:
            score = (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)
            f_scores.append(score)

    return float(np.mean(f_scores))


def optimize_f05_threshold(
    s1_ids: List[str],
    pair_predictions: List[Tuple[str, str, float]],  # list of (s1_id, cand_id, prob)
    ground_truth_matches: Dict[str, List[str]],
    min_thresh: float = 0.50,
    max_thresh: float = 0.95,
    step: float = 0.01,
    beta: float = 0.5,
) -> Tuple[float, float]:
    """Grid-searches decision probability threshold for Macro F_0.5 optimization."""
    logger.info("Grid-searching probability decision threshold for Macro F_0.5 optimization...")

    best_threshold = 0.85
    best_score = -1.0

    thresholds = np.arange(min_thresh, max_thresh + 1e-5, step)

    for thresh in thresholds:
        pred_map: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_ids}
        for s1_id, cand_id, prob in pair_predictions:
            if prob >= thresh:
                if s1_id in pred_map:
                    pred_map[s1_id].append(cand_id)

        score = compute_macro_f_beta(
            s1_ids=s1_ids,
            predicted_matches=pred_map,
            ground_truth_matches=ground_truth_matches,
            beta=beta,
        )

        if score > best_score:
            best_score = score
            best_threshold = float(thresh)

    logger.info(f"🎯 Optimal Probability Threshold: {best_threshold:.2f} | Best Macro F_{beta}: {best_score:.4f}")
    return best_threshold, best_score


def generate_submission_files(
    s1_ids: List[str],
    pair_predictions: List[Tuple[str, str, float]],  # list of (s1_id, cand_id, prob)
    decision_threshold: float,
    output_dir: str,
) -> Tuple[str, str]:
    """
    Generates official competition submission files:
      1. matching_results.tsv (source1_entity_id, matched_entity_ids)
      2. candidate_pairs.tsv (source1_entity_id, candidate_entity_ids)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group candidates and matches by s1_id
    cand_map: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_ids}
    match_map: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_ids}

    for s1_id, cand_id, prob in pair_predictions:
        if s1_id in cand_map:
            cand_map[s1_id].append(cand_id)
            if prob >= decision_threshold:
                match_map[s1_id].append(cand_id)

    # 1. Candidate Pairs TSV
    cand_pairs_path = os.path.join(output_dir, "candidate_pairs.tsv")
    cand_rows = []
    for s1_id in s1_ids:
        c_str = ",".join(cand_map[s1_id])
        cand_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": c_str})

    cand_df = pd.DataFrame(cand_rows)
    cand_df.to_csv(cand_pairs_path, sep="\t", index=False)
    logger.info(f"Saved {len(cand_df)} candidate pair lists -> {cand_pairs_path}")

    # 2. Matching Results TSV
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    match_rows = []
    for s1_id in s1_ids:
        m_str = ",".join(match_map[s1_id])
        match_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": m_str})

    match_df = pd.DataFrame(match_rows)
    match_df.to_csv(matching_path, sep="\t", index=False)
    logger.info(f"Saved {len(match_df)} matching result rows -> {matching_path}")

    return matching_path, cand_pairs_path


def validate_submission_files(matching_path: str, cand_pairs_path: str) -> bool:
    """Validates schema, delimiter, column names, and non-emptiness of submission files."""
    if not os.path.exists(matching_path) or not os.path.exists(cand_pairs_path):
        logger.error("Validation failed: Output file missing!")
        return False

    m_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    c_df = pd.read_csv(cand_pairs_path, sep="\t", dtype=str, keep_default_na=False)

    m_cols = ["source1_entity_id", "matched_entity_ids"]
    c_cols = ["source1_entity_id", "candidate_entity_ids"]

    if list(m_df.columns) != m_cols:
        logger.error(f"Invalid columns in matching_results.tsv: {list(m_df.columns)} vs {m_cols}")
        return False

    if list(c_df.columns) != c_cols:
        logger.error(f"Invalid columns in candidate_pairs.tsv: {list(c_df.columns)} vs {c_cols}")
        return False

    if len(m_df) == 0:
        logger.error("matching_results.tsv is empty!")
        return False

    logger.info("✅ Submission files validated successfully!")
    return True
