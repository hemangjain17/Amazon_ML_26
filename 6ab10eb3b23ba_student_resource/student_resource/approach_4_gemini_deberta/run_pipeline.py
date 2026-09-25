import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np

# Add project root and module directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "src"))

from config import (
    path_config,
    gemini_config,
    cascade_config,
    deberta_config,
    threshold_config,
    telegram_config,
)
from src.telegram_notifier import TelegramNotifier
from src.dataset_loader import (
    load_split_data,
    format_entity_text,
    build_ground_truth_map,
    get_entity_id,
    clean_str,
)
from src.gemini_retriever import (
    get_or_compute_embeddings,
    search_faiss_candidates,
)
from src.cascade_pruner import run_cascade_pruning
from src.deberta_cross_encoder import (
    load_deberta_model_and_tokenizer,
    train_cross_encoder_with_oom_safeguard,
    predict_cross_encoder_probabilities,
)
from src.eval_optimizer import (
    optimize_f05_threshold,
    generate_submission_files,
    validate_submission_files,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Approach4Pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Approach 4: Gemini Bi-Encoder + Cascade Pruner + DeBERTa-v3-Large Cross-Encoder")
    parser.add_argument("--dry-run", action="store_true", help="Runs fast end-to-end dry run on small subset")
    parser.add_argument("--subset-size", type=int, default=None, help="Limit number of S1 items for testing")
    parser.add_argument("--mock-gemini", action="store_true", help="Force mock Gemini embeddings (no API key required)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of DeBERTa training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="DeBERTa micro-batch size")
    parser.add_argument("--gemini-api-key", type=str, default=None, help="Override GEMINI_API_KEY")
    parser.add_argument("--telegram-token", type=str, default=None, help="Override TELEGRAM_BOT_TOKEN")
    parser.add_argument("--telegram-chat-id", type=str, default=None, help="Override TELEGRAM_CHAT_ID")
    return parser.parse_args()


def run_pipeline():
    args = parse_args()

    # Override configs from arguments
    if args.dry_run:
        logger.info("⚡ DRY-RUN MODE ENABLED: Using subset_size=50, 1 epoch, mock gemini if key missing.")
        args.subset_size = args.subset_size or 50
        args.epochs = 1
        gemini_config.mock_mode = True if not args.gemini_api_key and not os.getenv("GEMINI_API_KEY") else args.mock_gemini
    else:
        gemini_config.mock_mode = args.mock_gemini

    if args.gemini_api_key:
        gemini_config.api_key = args.gemini_api_key
    if args.telegram_token:
        telegram_config.bot_token = args.telegram_token
    if args.telegram_chat_id:
        telegram_config.chat_id = args.telegram_chat_id

    # Initialize Telegram Notifier
    notifier = TelegramNotifier(
        bot_token=telegram_config.bot_token,
        chat_id=telegram_config.chat_id,
        enabled=telegram_config.enabled or bool(args.telegram_token and args.telegram_chat_id),
    )

    config_summary = {
        "Mode": "Dry-Run" if args.dry_run else "Full Run",
        "Subset Size": args.subset_size or "Full Dataset",
        "Gemini Model": gemini_config.model_name,
        "Mock Gemini": gemini_config.mock_mode,
        "DeBERTa Model": deberta_config.model_name,
        "Epochs": args.epochs,
        "Batch Size": args.batch_size,
        "Top-K Retrieval": gemini_config.top_k_retrieval,
        "Top-K Pruned": cascade_config.top_k_pruned,
    }

    notifier.send_start("Approach 4: Gemini + DeBERTa-v3-Large Pipeline", config_summary)
    notifier.start_heartbeat(interval_sec=telegram_config.heartbeat_interval_sec)

    try:
        t0 = time.time()

        # =========================================================================
        # STEP 1: LOAD TRAIN & TEST DATA
        # =========================================================================
        logger.info("=== STEP 1/5: Loading Dataset ===")
        notifier.update_status("Loading train and test datasets...")

        train_data = load_split_data(path_config.train_dir, is_train=True, subset_size=args.subset_size)
        test_data = load_split_data(path_config.test_dir, is_train=False, subset_size=args.subset_size)

        train_s1_df = train_data["s1"]
        train_s2_df = train_data["s2"]
        train_s3_df = train_data["s3"]
        train_gt_df = train_data["gt"]
        train_gt_map = build_ground_truth_map(train_gt_df)

        test_s1_df = test_data["s1"]
        test_s2_df = test_data["s2"]
        test_s3_df = test_data["s3"]

        # Concatenate candidate pools (S2 + S3)
        train_cand_df = pd.concat([train_s2_df, train_s3_df], ignore_index=True)
        test_cand_df = pd.concat([test_s2_df, test_s3_df], ignore_index=True)

        logger.info(f"Train S1: {len(train_s1_df)} | Candidate Pool (S2+S3): {len(train_cand_df)}")
        logger.info(f"Test S1 : {len(test_s1_df)} | Candidate Pool (S2+S3): {len(test_cand_df)}")

        # Format Entity Text
        train_s1_texts = format_entity_text(train_s1_df)
        train_cand_texts = format_entity_text(train_cand_df)
        train_s1_ids = [get_entity_id(r) for r in train_s1_df.to_dict("records")]
        train_cand_ids = [get_entity_id(r) for r in train_cand_df.to_dict("records")]

        test_s1_texts = format_entity_text(test_s1_df)
        test_cand_texts = format_entity_text(test_cand_df)
        test_s1_ids = [get_entity_id(r) for r in test_s1_df.to_dict("records")]
        test_cand_ids = [get_entity_id(r) for r in test_cand_df.to_dict("records")]

        # =========================================================================
        # STEP 2: STAGE 1 - GEMINI BI-ENCODER CANDIDATE RETRIEVAL
        # =========================================================================
        logger.info("=== STEP 2/5: Stage 1 - Gemini Candidate Retrieval (FAISS) ===")
        notifier.update_status("Stage 1: Computing Gemini embeddings and FAISS top-50 search...")

        train_s1_emb = get_or_compute_embeddings(
            os.path.join(path_config.embeddings_dir, "train_s1_gemini.npy"),
            train_s1_texts,
            api_key=gemini_config.api_key,
            model_name=gemini_config.model_name,
            batch_size=gemini_config.batch_size,
            max_workers=gemini_config.max_workers,
            mock_mode=gemini_config.mock_mode,
        )
        train_cand_emb = get_or_compute_embeddings(
            os.path.join(path_config.embeddings_dir, "train_cand_gemini.npy"),
            train_cand_texts,
            api_key=gemini_config.api_key,
            model_name=gemini_config.model_name,
            batch_size=gemini_config.batch_size,
            max_workers=gemini_config.max_workers,
            mock_mode=gemini_config.mock_mode,
        )

        test_s1_emb = get_or_compute_embeddings(
            os.path.join(path_config.embeddings_dir, "test_s1_gemini.npy"),
            test_s1_texts,
            api_key=gemini_config.api_key,
            model_name=gemini_config.model_name,
            batch_size=gemini_config.batch_size,
            max_workers=gemini_config.max_workers,
            mock_mode=gemini_config.mock_mode,
        )
        test_cand_emb = get_or_compute_embeddings(
            os.path.join(path_config.embeddings_dir, "test_cand_gemini.npy"),
            test_cand_texts,
            api_key=gemini_config.api_key,
            model_name=gemini_config.model_name,
            batch_size=gemini_config.batch_size,
            max_workers=gemini_config.max_workers,
            mock_mode=gemini_config.mock_mode,
        )

        # FAISS search Top-50
        train_retrieval = search_faiss_candidates(
            query_embeddings=train_s1_emb,
            candidate_embeddings=train_cand_emb,
            query_ids=train_s1_ids,
            candidate_ids=train_cand_ids,
            top_k=gemini_config.top_k_retrieval,
        )

        test_retrieval = search_faiss_candidates(
            query_embeddings=test_s1_emb,
            candidate_embeddings=test_cand_emb,
            query_ids=test_s1_ids,
            candidate_ids=test_cand_ids,
            top_k=gemini_config.top_k_retrieval,
        )

        # =========================================================================
        # STEP 3: STAGE 1.5 - LIGHTWEIGHT CASCADE CANDIDATE PRUNING
        # =========================================================================
        logger.info("=== STEP 3/5: Stage 1.5 - Cascade Pruning (Top-50 -> Top-15) ===")
        notifier.update_status("Stage 1.5: Pruning candidates to Top-15 using String + Gemini Cascade...")

        train_pruned = run_cascade_pruning(
            s1_df=train_s1_df,
            s2_s3_df=train_cand_df,
            retrieval_map=train_retrieval,
            top_k_pruned=cascade_config.top_k_pruned,
            w_gemini=cascade_config.w_gemini,
            w_jaccard=cascade_config.w_jaccard,
            w_levenshtein=cascade_config.w_levenshtein,
        )

        test_pruned = run_cascade_pruning(
            s1_df=test_s1_df,
            s2_s3_df=test_cand_df,
            retrieval_map=test_retrieval,
            top_k_pruned=cascade_config.top_k_pruned,
            w_gemini=cascade_config.w_gemini,
            w_jaccard=cascade_config.w_jaccard,
            w_levenshtein=cascade_config.w_levenshtein,
        )

        # Prepare Cross-Encoder Training & Test Pairs
        train_s1_map = dict(zip(train_s1_ids, train_s1_texts))
        train_cand_map = dict(zip(train_cand_ids, train_cand_texts))

        train_pairs = []
        train_labels = []

        for s1_id, pruned_cands in train_pruned.items():
            s1_txt = train_s1_map.get(s1_id, "")
            gt_matches = set(train_gt_map.get(s1_id, []))

            for cand_id, cand_txt, _, _ in pruned_cands:
                label = 1 if cand_id in gt_matches else 0
                train_pairs.append((s1_txt, cand_txt))
                train_labels.append(label)

        logger.info(f"Constructed {len(train_pairs)} training pairs for DeBERTa Cross-Encoder.")

        # =========================================================================
        # STEP 4: STAGE 2 - DEBERTA-V3-LARGE CROSS-ENCODER TRAINING & INFERENCE
        # =========================================================================
        logger.info("=== STEP 4/5: Stage 2 - DeBERTa-v3-Large Fine-Tuning & Inference ===")
        notifier.update_status("Stage 2: Training DeBERTa-v3-Large Cross-Encoder...")

        model, tokenizer, mode_loaded = load_deberta_model_and_tokenizer(
            model_name=deberta_config.model_name,
            use_qlora=deberta_config.use_qlora,
            device=deberta_config.device,
        )

        model = train_cross_encoder_with_oom_safeguard(
            model=model,
            tokenizer=tokenizer,
            train_pairs=train_pairs,
            train_labels=train_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=deberta_config.grad_accum_steps,
            lr=deberta_config.learning_rate,
            max_seq_length=deberta_config.max_seq_length,
            device=deberta_config.device,
            telegram_notifier=notifier,
        )

        # Run Validation/Train Scoring to find optimal F_0.5 threshold
        logger.info("Scoring training/validation pairs for threshold optimization...")
        notifier.update_status("Stage 2: Scoring pairs and searching optimal F_0.5 threshold...")

        train_pair_tuples = []
        idx = 0
        for s1_id, pruned_cands in train_pruned.items():
            for cand_id, _, _, _ in pruned_cands:
                train_pair_tuples.append((s1_id, cand_id))

        train_probs = predict_cross_encoder_probabilities(
            model=model,
            tokenizer=tokenizer,
            pairs=train_pairs,
            batch_size=deberta_config.eval_batch_size,
            max_seq_length=deberta_config.max_seq_length,
            device=deberta_config.device,
        )

        train_pair_preds = [(s1_id, cand_id, prob) for (s1_id, cand_id), prob in zip(train_pair_tuples, train_probs)]

        best_thresh, best_f05 = optimize_f05_threshold(
            s1_ids=train_s1_ids,
            pair_predictions=train_pair_preds,
            ground_truth_matches=train_gt_map,
            min_thresh=threshold_config.prob_min,
            max_thresh=threshold_config.prob_max,
            step=threshold_config.prob_step,
            beta=threshold_config.f_beta,
        )

        # Run Inference on Test Set
        logger.info("Scoring test candidates with Cross-Encoder...")
        notifier.update_status("Stage 2: Scoring test candidate set...")

        test_s1_map = dict(zip(test_s1_ids, test_s1_texts))
        test_cand_map = dict(zip(test_cand_ids, test_cand_texts))

        test_pair_tuples = []
        test_pairs = []

        for s1_id, pruned_cands in test_pruned.items():
            s1_txt = test_s1_map.get(s1_id, "")
            for cand_id, cand_txt, _, _ in pruned_cands:
                test_pair_tuples.append((s1_id, cand_id))
                test_pairs.append((s1_txt, cand_txt))

        test_probs = predict_cross_encoder_probabilities(
            model=model,
            tokenizer=tokenizer,
            pairs=test_pairs,
            batch_size=deberta_config.eval_batch_size,
            max_seq_length=deberta_config.max_seq_length,
            device=deberta_config.device,
        )

        test_pair_preds = [(s1_id, cand_id, prob) for (s1_id, cand_id), prob in zip(test_pair_tuples, test_probs)]

        # =========================================================================
        # STEP 5: STAGE 3 - GENERATE SUBMISSION & VALIDATE
        # =========================================================================
        logger.info("=== STEP 5/5: Stage 3 - Export Submission Files ===")
        notifier.update_status("Stage 3: Generating matching_results.tsv & candidate_pairs.tsv...")

        # Load complete list of S1 entity IDs from test_source1.tsv to guarantee 100% submission coverage
        test_s1_path = os.path.join(path_config.test_dir, "test_source1.tsv")
        full_test_s1_df = pd.read_csv(test_s1_path, sep="\t", dtype=str, usecols=lambda col: col in ("source1_entity_id", "entity_id", "id"))
        id_col = full_test_s1_df.columns[0]
        all_test_s1_ids = [clean_str(i) for i in full_test_s1_df[id_col]]

        matching_path, cand_pairs_path = generate_submission_files(
            s1_ids=all_test_s1_ids,
            pair_predictions=test_pair_preds,
            decision_threshold=best_thresh,
            output_dir=path_config.output_dir,
        )

        is_valid = validate_submission_files(matching_path, cand_pairs_path)

        elapsed_min = (time.time() - t0) / 60.0

        metrics = {
            "Status": "VALIDATED" if is_valid else "INVALID",
            "Optimal Probability Threshold": f"{best_thresh:.2f}",
            "Train Macro F_0.5 Score": f"{best_f05:.4f}",
            "Test Pairs Processed": len(test_pairs),
            "Execution Time (mins)": f"{elapsed_min:.2f}",
            "DeBERTa Mode": mode_loaded,
        }

        logger.info(f"🎉 Pipeline Completed in {elapsed_min:.2f} mins. Validated: {is_valid}")

        notifier.stop_heartbeat()
        notifier.send_success(metrics, files=[matching_path, cand_pairs_path])

    except Exception as e:
        logger.error(f"❌ Pipeline Execution Error: {e}", exc_info=True)
        notifier.stop_heartbeat()
        notifier.send_error("Approach 4 Pipeline Execution", e)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
