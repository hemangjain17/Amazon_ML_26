import os
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from catboost import CatBoostClassifier

try:
    from approach_2_simpleVAE.config import path_config, reranker_config
except (ImportError, ValueError):
    try:
        from config import path_config, reranker_config
    except (ImportError, ValueError):
        from ..config import path_config, reranker_config


def compute_macro_f_beta(
    s1_ids: List[str],
    predicted_matches: Dict[str, List[str]],
    ground_truth_matches: Dict[str, List[str]],
    beta: float = 0.5,
) -> float:
    beta_sq = beta ** 2
    f_scores = []

    for s1_id in s1_ids:
        preds = set(predicted_matches.get(s1_id, []))
        targets = set(ground_truth_matches.get(s1_id, []))

        if not preds and not targets:
            f_scores.append(1.0)
            continue
        elif not preds and targets:
            f_scores.append(0.0)
            continue
        elif preds and not targets:
            f_scores.append(0.0)
            continue

        tp = len(preds.intersection(targets))
        fp = len(preds - targets)
        fn = len(targets - preds)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f_scores.append(0.0)
        else:
            score = (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)
            f_scores.append(score)

    return float(np.mean(f_scores))


def train_reranker_model(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
) -> CatBoostClassifier:
    print(f"Training CatBoost Classifier on {len(feature_df)} candidate pair rows...")
    model = CatBoostClassifier(
        iterations=reranker_config.catboost_iterations,
        depth=reranker_config.catboost_depth,
        learning_rate=reranker_config.catboost_learning_rate,
        loss_function="Logloss",
        eval_metric="Logloss",
        verbose=100,
        random_seed=42,
    )
    model.fit(feature_df, labels)
    return model


def optimize_f05_threshold(
    s1_ids: List[str],
    pairs: List[Tuple[str, str]],
    probabilities: np.ndarray,
    ground_truth_matches: Dict[str, List[str]],
) -> Tuple[float, float]:
    print("Grid-searching probability decision threshold for Macro F_0.5 optimization...")
    best_threshold = reranker_config.default_threshold
    best_score = -1.0

    thresholds = np.arange(
        reranker_config.prob_threshold_min,
        reranker_config.prob_threshold_max,
        reranker_config.prob_threshold_step,
    )

    for thresh in thresholds:
        pred_map = {s1_id: [] for s1_id in s1_ids}
        for (s1_id, cand_id), prob in zip(pairs, probabilities):
            if prob >= thresh:
                pred_map[s1_id].append(cand_id)

        score = compute_macro_f_beta(
            s1_ids=s1_ids,
            predicted_matches=pred_map,
            ground_truth_matches=ground_truth_matches,
            beta=reranker_config.f_beta,
        )

        if score > best_score:
            best_score = score
            best_threshold = thresh

    print(f"--> Optimal Threshold: {best_threshold:.2f} | Best Validation Macro F_0.5: {best_score:.4f}")
    return best_threshold, best_score


def generate_matching_results(
    s1_ids: List[str],
    pairs: List[Tuple[str, str]],
    probabilities: np.ndarray,
    threshold: float,
    output_matching_path: str = None,
) -> str:
    output_matching_path = output_matching_path or os.path.join(path_config.output_dir, "matching_results.tsv")
    pred_map = {s1_id: [] for s1_id in s1_ids}

    for (s1_id, cand_id), prob in zip(pairs, probabilities):
        if prob >= threshold:
            pred_map[s1_id].append(cand_id)

    print(f"Exporting final matches to {output_matching_path}...")
    with open(output_matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            match_str = ",".join(pred_map.get(s1_id, []))
            f.write(f"{s1_id}\t{match_str}\n")

    print(f"Successfully generated matching_results.tsv ({len(s1_ids)} rows)")
    return output_matching_path


def validate_submission_files(
    matching_path: str = None,
    candidate_path: str = None,
    test_dir: str = None,
) -> bool:
    matching_path = matching_path or os.path.join(path_config.output_dir, "matching_results.tsv")
    candidate_path = candidate_path or os.path.join(path_config.output_dir, "candidate_pairs.tsv")
    test_dir = test_dir or path_config.test_dir

    validator_script = os.path.join(path_config.base_dir, "utils", "validate_submission.py")
    if not os.path.exists(validator_script):
        print(f"Validator script not found at {validator_script}, skipping validation check.")
        return True

    cmd = [
        "python",
        validator_script,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir,
    ]

    print(f"Running submission validation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Validation FAILED with output:")
        print(result.stderr)
        return False

    print("Submission Validation PASSED!")
    return True
