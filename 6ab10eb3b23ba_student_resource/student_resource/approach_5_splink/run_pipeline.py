"""Run Approach 5 from Kaggle, SageMaker, or the repository root."""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_RESOURCE = os.path.dirname(CURRENT_DIR)
if STUDENT_RESOURCE not in sys.path:
    sys.path.insert(0, STUDENT_RESOURCE)

from approach_5_splink.config import config
from approach_5_splink.src.pipeline import (
    add_blocking_keys,
    add_clusters,
    choose_threshold,
    estimate_comparisons,
    fit_settings,
    ground_truth,
    load_table,
    make_settings,
    prepare_frame,
    score,
    write_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Approach 5: Splink/DuckDB entity resolution")
    parser.add_argument("--dry-run", action="store_true", help="Use a deterministic subset of labelled/test S1 rows")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--token-cap", type=int, default=config.token_cap)
    parser.add_argument("--house-cap", type=int, default=config.house_cap)
    parser.add_argument("--max-comparisons", type=int, default=config.max_comparisons)
    parser.add_argument("--validation-fraction", type=float, default=config.validation_fraction)
    parser.add_argument("--output-dir", default=config.output_dir)
    return parser.parse_args()


def subset(frame: pd.DataFrame, size: int | None) -> pd.DataFrame:
    if not size or size >= len(frame):
        return frame
    return frame.sample(size, random_state=config.random_seed).sort_index().reset_index(drop=True)


def prepare_pair(s1_path: str, pool_paths: list[str], token_cap: int, house_cap: int, size: int | None):
    s1 = subset(prepare_frame(load_table(s1_path)), size)
    pool = pd.concat([prepare_frame(load_table(path)) for path in pool_paths], ignore_index=True)
    return add_blocking_keys(s1, pool, token_cap, house_cap)


def main():
    args = parse_args()
    t0 = time.time()
    data_root = "/kaggle/input" if os.path.exists("/kaggle/input") else os.path.join(STUDENT_RESOURCE, "dataset")

    def find_file(split: str, name: str) -> str:
        direct = os.path.join(data_root, split, name)
        if os.path.exists(direct):
            return direct
        for root, _, files in os.walk(data_root):
            if name in files and os.path.basename(root) == split:
                return os.path.join(root, name)
        raise FileNotFoundError(f"Could not find {name} under {data_root}")

    train_s1_path = find_file("train", "train_source1.tsv")
    train_pool_paths = [find_file("train", "train_source2.tsv"), find_file("train", "train_source3.tsv")]
    train_gt_path = find_file("train", "train_ground_truth.tsv")
    test_s1_path = find_file("test", "test_source1.tsv")
    test_pool_paths = [find_file("test", "test_source2.tsv"), find_file("test", "test_source3.tsv")]

    train_truth = ground_truth(train_gt_path)
    train_s1, train_pool = prepare_pair(train_s1_path, train_pool_paths, args.token_cap, args.house_cap, args.subset_size if args.dry_run else None)
    train_s1, train_pool = add_clusters(train_s1, train_pool, train_truth)
    comparisons = estimate_comparisons(train_s1, train_pool)
    print(f"train S1={len(train_s1):,} pool={len(train_pool):,} estimated blocking comparisons={comparisons:,}")
    if comparisons > args.max_comparisons:
        raise RuntimeError(f"Blocking gate failed: {comparisons:,} > {args.max_comparisons:,}. Lower caps or add a more selective key.")

    split_at = max(1, int(len(train_s1) * (1.0 - args.validation_fraction)))
    fit_s1 = train_s1.iloc[:split_at].copy()
    validation_s1 = train_s1.iloc[split_at:].copy()
    settings = make_settings()
    print("Fitting Splink parameters on the training partition...")
    fitted_settings = fit_settings(fit_s1, train_pool, settings, config.max_pairs_for_u, config.match_recall_prior)
    validation_scores = score(validation_s1, train_pool, fitted_settings)
    validation_truth = {key: train_truth.get(key, []) for key in validation_s1["entity_id"]}
    threshold, validation_f05 = choose_threshold(
        validation_s1["entity_id"].tolist(), validation_scores, validation_truth,
        config.min_threshold, config.max_threshold, config.threshold_step,
    )
    print(f"validation candidates={len(validation_scores):,} threshold={threshold:.2f} macro F0.5={validation_f05:.4f}")

    print("Retraining Splink parameters on all labelled training rows...")
    settings = make_settings()
    fitted_settings = fit_settings(train_s1, train_pool, settings, config.max_pairs_for_u, config.match_recall_prior)
    test_s1, test_pool = prepare_pair(test_s1_path, test_pool_paths, args.token_cap, args.house_cap, args.subset_size if args.dry_run else None)
    test_scores = score(test_s1, test_pool, fitted_settings)
    os.makedirs(args.output_dir, exist_ok=True)
    candidate_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    matching_path = os.path.join(args.output_dir, "matching_results.tsv")
    write_outputs(test_s1["entity_id"].tolist(), test_scores, threshold, candidate_path, matching_path)
    print(f"test candidates={len(test_scores):,} matches={len(test_scores[test_scores['match_probability'] >= threshold]):,}")
    print(f"candidate output: {candidate_path}")
    print(f"matching output: {matching_path}")
    print(f"total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
