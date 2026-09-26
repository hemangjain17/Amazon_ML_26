"""Resume Approach 1 after Contrastive VAE training.

This entry point never retrains the VAE. It loads the saved checkpoint, reuses
valid latent caches, trains a supervised CatBoost reranker from train ground
truth, scores the test candidates, writes submissions, and notifies Telegram.
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
STUDENT_RESOURCE = os.path.dirname(ROOT)
if STUDENT_RESOURCE not in sys.path:
    sys.path.insert(0, STUDENT_RESOURCE)

from approach_1_contrastiveVAE.config import (  # noqa: E402
    blocking_config,
    model_config,
    path_config,
    reranker_config,
)
from approach_1_contrastiveVAE.src.blocking import (  # noqa: E402
    build_faiss_index_and_search,
    extract_latent_embeddings,
)
from approach_1_contrastiveVAE.src.feature_extractor import (  # noqa: E402
    build_candidate_feature_matrix,
)
from approach_1_contrastiveVAE.src.model import ContrastiveVAE  # noqa: E402
from approach_1_contrastiveVAE.src.reranker import (  # noqa: E402
    compute_macro_f_beta,
    generate_matching_results,
)
from approach_1_contrastiveVAE.src.telegram_notifier import TelegramNotifier  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--test-chunk-size", type=int, default=50000)
    parser.add_argument("--heartbeat-seconds", type=int, default=300)
    parser.add_argument("--send-artifacts", action="store_true")
    return parser.parse_args()


def load_env_file():
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cache_or_extract(
    name: str,
    expected_rows: int,
    df: pd.DataFrame,
    model: ContrastiveVAE,
    device: torch.device,
    batch_size: int,
    aliases: Tuple[str, ...] = (),
):
    candidate_names = (name,) + aliases
    for candidate_name in candidate_names:
        cache_path = os.path.join(path_config.embeddings_dir, candidate_name)
        ids_path = cache_path.replace(".npy", "_ids.npy")
        countries_path = cache_path.replace(".npy", "_countries.npy")
        if not os.path.exists(cache_path):
            continue
        embeddings = np.load(cache_path, mmap_mode="r")
        if embeddings.shape[0] != expected_rows:
            print(f"Ignoring stale cache {cache_path}: expected {expected_rows}, found {embeddings.shape[0]}")
            continue
        if os.path.exists(ids_path) and os.path.exists(countries_path):
            ids = np.load(ids_path, allow_pickle=True).tolist()
            countries = np.load(countries_path, allow_pickle=True).tolist()
        else:
            # Legacy blocking.py saved only the vectors. Its extraction order is
            # exactly the dataframe order, so recover the side metadata safely.
            ids = df["entity_id"].astype(str).tolist()
            countries = df["country"].fillna("US").astype(str).tolist()
            np.save(ids_path, np.asarray(ids, dtype=object), allow_pickle=True)
            np.save(countries_path, np.asarray(countries, dtype=object), allow_pickle=True)
            print(f"Recovered legacy cache metadata beside {cache_path}")
        if len(ids) == expected_rows and len(countries) == expected_rows:
            print(f"Reusing cached embeddings: {cache_path}")
            return ids, countries, np.asarray(embeddings, dtype=np.float32)
        print(f"Ignoring cache metadata beside {cache_path}: expected {expected_rows} rows")

    cache_path = os.path.join(path_config.embeddings_dir, name)
    ids_path = cache_path.replace(".npy", "_ids.npy")
    countries_path = cache_path.replace(".npy", "_countries.npy")

    print(f"Extracting embeddings for {expected_rows:,} rows: {name}")
    ids, countries, embeddings = extract_latent_embeddings(df, model, device, batch_size=batch_size)
    np.save(cache_path, embeddings)
    np.save(ids_path, np.asarray(ids, dtype=object), allow_pickle=True)
    np.save(countries_path, np.asarray(countries, dtype=object), allow_pickle=True)
    return ids, countries, embeddings


def ground_truth_map(gt_df: pd.DataFrame) -> Dict[str, set]:
    result: Dict[str, set] = {}
    for _, row in gt_df.iterrows():
        source_id = str(row["source1_entity_id"])
        values = str(row.get("matched_entity_ids", ""))
        result[source_id] = {item.strip() for item in values.split(",") if item.strip()}
    return result


def label_pairs(pairs: List[Tuple[str, str]], truth: Dict[str, set]) -> np.ndarray:
    return np.asarray([int(candidate in truth.get(source, set())) for source, candidate in pairs], dtype=np.int8)


def train_supervised_reranker(features: pd.DataFrame, labels: np.ndarray) -> CatBoostClassifier:
    if len(np.unique(labels)) < 2:
        raise RuntimeError("Training candidate pairs contain only one class; cannot train CatBoost.")
    model = CatBoostClassifier(
        iterations=reranker_config.catboost_iterations,
        depth=reranker_config.catboost_depth,
        learning_rate=reranker_config.catboost_learning_rate,
        loss_function="Logloss",
        eval_metric="Logloss",
        verbose=100,
        random_seed=42,
        allow_writing_files=False,
    )
    model.fit(features, labels)
    return model


def threshold_by_f05(source_ids, pairs, probabilities, truth):
    best_threshold = reranker_config.default_threshold
    best_score = -1.0
    for threshold in np.arange(
        reranker_config.prob_threshold_min,
        reranker_config.prob_threshold_max + 1e-8,
        reranker_config.prob_threshold_step,
    ):
        predictions = {source_id: [] for source_id in source_ids}
        for (source_id, candidate_id), probability in zip(pairs, probabilities):
            if probability >= threshold:
                predictions[source_id].append(candidate_id)
        score = compute_macro_f_beta(
            source_ids,
            predictions,
            {key: list(value) for key, value in truth.items()},
            reranker_config.f_beta,
        )
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold, best_score


def main():
    args = parse_args()
    load_env_file()
    notifier = TelegramNotifier()
    checkpoint_path = args.checkpoint or os.path.join(path_config.models_dir, "best_contrastive_vae.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    started = time.time()
    notifier.send_message(f"START: Approach 1 resume on Kaggle. Checkpoint: {checkpoint_path}")
    notifier.start_heartbeat(args.heartbeat_seconds)

    try:
        device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = ContrastiveVAE(
            backbone_name=model_config.backbone_name,
            latent_dim=model_config.latent_dim,
            temperature=model_config.temperature,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        notifier.update("checkpoint loaded")

        train_dir, test_dir = path_config.train_dir, path_config.test_dir
        train_s1 = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t")
        train_s2 = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t")
        train_s3 = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t")
        train_gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t")
        test_s1 = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t")
        test_s2 = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t")
        test_s3 = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t")

        truth = ground_truth_map(train_gt)
        train_candidates = pd.concat([train_s2, train_s3], ignore_index=True)
        test_candidates = pd.concat([test_s2, test_s3], ignore_index=True)
        notifier.update("datasets loaded")

        train_s1_ids, train_s1_countries, train_s1_emb = cache_or_extract(
            "train_s1_embeddings.npy", len(train_s1), train_s1, model, device, args.embedding_batch_size,
            aliases=("s1_embeddings.npy",),
        )
        train_cand_ids, train_cand_countries, train_cand_emb = cache_or_extract(
            "train_cand_embeddings.npy", len(train_candidates), train_candidates, model, device, args.embedding_batch_size,
            aliases=("cand_embeddings.npy",),
        )
        notifier.update("training embeddings ready")

        top_k = args.top_k or blocking_config.top_k_candidates
        train_map = build_faiss_index_and_search(
            train_s1_ids, train_s1_countries, train_s1_emb,
            train_cand_ids, train_cand_countries, train_cand_emb, top_k=top_k,
        )
        train_features, train_pairs = build_candidate_feature_matrix(
            train_map, train_s1, train_s2, train_s3,
            train_s1_emb, train_cand_emb,
            dict(zip(train_s1_ids, range(len(train_s1_ids)))),
            dict(zip(train_cand_ids, range(len(train_cand_ids)))),
        )
        train_labels = label_pairs(train_pairs, truth)
        notifier.send_message(f"TRAIN RERANKER: {len(train_pairs):,} blocked pairs; positives={int(train_labels.sum()):,}")
        reranker = train_supervised_reranker(train_features, train_labels)
        train_probabilities = reranker.predict_proba(train_features)[:, 1]
        threshold, train_f05 = threshold_by_f05(train_s1_ids, train_pairs, train_probabilities, truth)
        notifier.send_message(f"THRESHOLD: {threshold:.2f}; train Macro F0.5={train_f05:.4f}")

        test_s1_ids, test_s1_countries, test_s1_emb = cache_or_extract(
            "test_s1_embeddings.npy", len(test_s1), test_s1, model, device, args.embedding_batch_size
        )
        test_cand_ids, test_cand_countries, test_cand_emb = cache_or_extract(
            "test_cand_embeddings.npy", len(test_candidates), test_candidates, model, device, args.embedding_batch_size
        )
        notifier.update("test embeddings ready")

        test_map = build_faiss_index_and_search(
            test_s1_ids, test_s1_countries, test_s1_emb,
            test_cand_ids, test_cand_countries, test_cand_emb, top_k=top_k,
        )
        candidate_path = os.path.join(path_config.output_dir, "candidate_pairs.tsv")
        with open(candidate_path, "w", encoding="utf-8") as output:
            output.write("source1_entity_id\tcandidate_entity_ids\n")
            for source_id in test_s1_ids:
                output.write(f"{source_id}\t{','.join(test_map.get(source_id, []))}\n")

        # Feature extraction for the test set is chunked to avoid holding all pairs in RAM.
        match_path = os.path.join(path_config.output_dir, "matching_results.tsv")
        with open(match_path, "w", encoding="utf-8") as output:
            output.write("source1_entity_id\tmatched_entity_ids\n")
            source_lookup = dict(zip(test_s1_ids, test_s1.to_dict("records")))
            s2_lookup = dict(zip(test_s2["entity_id"], test_s2.to_dict("records")))
            s3_lookup = dict(zip(test_s3["entity_id"], test_s3.to_dict("records")))
            all_pairs = [(source_id, candidate_id) for source_id, candidates in test_map.items() for candidate_id in candidates]
            predictions: Dict[str, List[str]] = {source_id: [] for source_id in test_s1_ids}
            for start in range(0, len(all_pairs), args.test_chunk_size):
                chunk_pairs = all_pairs[start:start + args.test_chunk_size]
                chunk_map: Dict[str, List[str]] = {}
                for source_id, candidate_id in chunk_pairs:
                    chunk_map.setdefault(source_id, []).append(candidate_id)
                chunk_features, ordered_pairs = build_candidate_feature_matrix(
                    chunk_map, test_s1, test_s2, test_s3,
                    test_s1_emb, test_cand_emb,
                    dict(zip(test_s1_ids, range(len(test_s1_ids)))),
                    dict(zip(test_cand_ids, range(len(test_cand_ids)))),
                )
                probabilities = reranker.predict_proba(chunk_features)[:, 1]
                for (source_id, candidate_id), probability in zip(ordered_pairs, probabilities):
                    if probability >= threshold:
                        predictions[source_id].append(candidate_id)
                notifier.update(f"test reranking {min(start + len(chunk_pairs), len(all_pairs)):,}/{len(all_pairs):,}")

            for source_id in test_s1_ids:
                output.write(f"{source_id}\t{','.join(predictions[source_id])}\n")

        validator = os.path.join(path_config.base_dir, "utils", "validate_submission.py")
        validation = subprocess.run(
            [
                sys.executable,
                validator,
                "--matching", match_path,
                "--candidate", candidate_path,
                "--test-dir", path_config.test_dir,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        print(validation.stdout)
        if validation.returncode != 0:
            raise RuntimeError(f"Official submission validation failed:\n{validation.stdout}\n{validation.stderr}")

        notifier.send_message(
            f"DONE: Approach 1 completed in {(time.time() - started) / 60:.1f} min. "
            f"Outputs: {match_path}, {candidate_path}"
        )
        for artifact in (checkpoint_path, candidate_path, match_path):
            if args.send_artifacts:
                notifier.send_document(artifact, caption=f"Approach 1 artifact: {os.path.basename(artifact)}")
    except Exception as exc:
        notifier.error("Approach 1 resume", exc)
        raise
    finally:
        notifier.stop_heartbeat()


if __name__ == "__main__":
    main()
