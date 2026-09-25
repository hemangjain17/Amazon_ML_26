import os
import sys
import torch
import pandas as pd
import numpy as np

# Add student_resource directory to path
student_resource_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(student_resource_dir)

from approach_1_contrastiveVAE.config import path_config, model_config, reranker_config, blocking_config
from approach_1_contrastiveVAE.src.trainer import train_contrastive_vae
from approach_1_contrastiveVAE.src.blocking import run_blocking_pipeline
from approach_1_contrastiveVAE.src.feature_extractor import build_candidate_feature_matrix
from approach_1_contrastiveVAE.src.reranker import (
    train_reranker_model,
    optimize_f05_threshold,
    compute_macro_f_beta,
    generate_matching_results,
    validate_submission_files,
)

def run_pipeline():
    print("==========================================================================")
    print("      STARTING LOCAL END-TO-END CONTRASTIVE VAE PIPELINE & EVALUATION     ")
    print("==========================================================================")

    dummy_dir = os.path.join(student_resource_dir, "dummy_folder")
    dummy_dataset_dir = os.path.join(dummy_dir, "dataset")
    dummy_output_dir = os.path.join(dummy_dir, "output")
    dummy_models_dir = os.path.join(dummy_dir, "models")
    os.makedirs(dummy_output_dir, exist_ok=True)
    os.makedirs(dummy_models_dir, exist_ok=True)

    # Point path config to dummy_folder
    path_config.dataset_dir = dummy_dataset_dir
    path_config.train_dir = os.path.join(dummy_dataset_dir, "train")
    path_config.test_dir = os.path.join(dummy_dataset_dir, "test")
    path_config.output_dir = dummy_output_dir
    path_config.models_dir = dummy_models_dir

    # Model settings for fast local test
    model_config.epochs = 2
    model_config.batch_size = 32
    model_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker_config.catboost_iterations = 200

    train_s1_path = os.path.join(path_config.train_dir, "train_source1.tsv")
    train_s2_path = os.path.join(path_config.train_dir, "train_source2.tsv")
    train_s3_path = os.path.join(path_config.train_dir, "train_source3.tsv")
    train_gt_path = os.path.join(path_config.train_dir, "train_ground_truth.tsv")

    test_s1_path = os.path.join(path_config.test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(path_config.test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(path_config.test_dir, "test_source3.tsv")
    test_gt_path = os.path.join(path_config.test_dir, "test_ground_truth.tsv")

    print(f"\n[STEP 1/5] Training Contrastive VAE Model locally on {model_config.device.upper()}...")
    checkpoint_path = train_contrastive_vae(
        s1_path=train_s1_path,
        s2_path=train_s2_path,
        s3_path=train_s3_path,
        gt_path=train_gt_path,
        epochs=model_config.epochs,
        batch_size=model_config.batch_size,
        save_s3=False
    )

    print(f"\n[STEP 2/5] Running Latent Vector FAISS Blocking on Test Subset...")
    test_s1 = pd.read_csv(test_s1_path, sep="\t")
    test_s2 = pd.read_csv(test_s2_path, sep="\t")
    test_s3 = pd.read_csv(test_s3_path, sep="\t")

    cand_tsv_path, candidate_map = run_blocking_pipeline(
        checkpoint_path=checkpoint_path,
        s1_df=test_s1,
        s2_df=test_s2,
        s3_df=test_s3,
        output_candidate_path=os.path.join(dummy_output_dir, "candidate_pairs.tsv")
    )

    print(f"\n[STEP 3/5] Extracting Pairwise Features for CatBoost Reranking...")
    feature_df, pairs = build_candidate_feature_matrix(
        candidate_map=candidate_map,
        s1_df=test_s1,
        s2_df=test_s2,
        s3_df=test_s3,
    )

    # Build Ground Truth labels for training/evaluating CatBoost
    test_gt = pd.read_csv(test_gt_path, sep="\t")
    gt_map = {}
    for _, row in test_gt.iterrows():
        s1_id = row["source1_entity_id"]
        m_str = str(row["matched_entity_ids"]) if pd.notnull(row["matched_entity_ids"]) else ""
        gt_map[s1_id] = [m.strip() for m in m_str.split(",") if m.strip()]

    pair_labels = []
    for s1_id, cand_id in pairs:
        pair_labels.append(1.0 if cand_id in gt_map.get(s1_id, []) else 0.0)
    pair_labels = np.array(pair_labels)

    print(f"Total pairs generated: {len(feature_df):,}, Positive match labels: {int(sum(pair_labels)):,}")

    print(f"\n[STEP 4/5] Training CatBoost Model & Tuning F_0.5 Probability Threshold...")
    catboost_model = train_reranker_model(feature_df, pair_labels)
    probs = catboost_model.predict_proba(feature_df)[:, 1]

    optimal_thresh, best_f05 = optimize_f05_threshold(
        s1_ids=test_s1["entity_id"].tolist(),
        pairs=pairs,
        probabilities=probs,
        ground_truth_matches=gt_map
    )

    matching_tsv_path = generate_matching_results(
        s1_ids=test_s1["entity_id"].tolist(),
        pairs=pairs,
        probabilities=probs,
        threshold=optimal_thresh,
        output_matching_path=os.path.join(dummy_output_dir, "matching_results.tsv")
    )

    print(f"\n[STEP 5/5] Running Official Submission Validator Script...")
    is_valid = validate_submission_files(
        matching_path=matching_tsv_path,
        candidate_path=cand_tsv_path,
        test_dir=path_config.test_dir
    )

    print("\n==========================================================================")
    print("                    EVALUATION RESULTS & SCORES SUMMARY                    ")
    print("==========================================================================")
    print(f" Local Dataset Subset    : {len(test_s1)} Test Entities (US, India, France)")
    print(f" Optimal Threshold (tau) : {optimal_thresh:.2f}")
    print(f" Local Macro F_0.5 Score : {best_f05:.4f}")
    print(f" Submission Format Valid : {'PASS' if is_valid else 'FAIL'}")
    print("==========================================================================")

if __name__ == "__main__":
    run_pipeline()
