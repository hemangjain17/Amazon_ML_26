"""Approach 6: B5 blocking with dual-GPU E5-small reranking."""
from __future__ import annotations

import argparse
import gc
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyClassifier
from sklearn.isotonic import IsotonicRegression
from tqdm.auto import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
STUDENT_RESOURCE = CURRENT_DIR.parent
if str(STUDENT_RESOURCE) not in sys.path:
    sys.path.insert(0, str(STUDENT_RESOURCE))

from approach_5_b5.run_pipeline import (  # noqa: E402
    ROOT,
    build_pairs,
    candidates,
    clean,
    contextual_features,
    f05,
    features,
    generate_pairs,
    load,
    make_index,
    positive_probability,
    train_model,
    truth_map,
)


DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def parse_devices(value: str) -> list[str]:
    requested = [part.strip() for part in value.split(",") if part.strip()]
    if not torch.cuda.is_available():
        print("CUDA unavailable; E5 will run on CPU")
        return ["cpu"]
    available = torch.cuda.device_count()
    devices = [f"cuda:{int(part)}" for part in requested if part.isdigit() and int(part) < available]
    if not devices:
        devices = ["cuda:0"]
    print(f"E5 devices: {devices}")
    return devices


def record_text(row: pd.Series, prefix: str) -> str:
    return (
        f"{prefix}: {row.name_n} [SEP] {row.addr_n} [SEP] "
        f"{row.country_n} [SEP] {row.name_skel} [SEP] {row.house} [SEP] {row.pin}"
    )


class DualE5:
    def __init__(self, model_name: str, devices: list[str], batch_size: int):
        self.devices = devices
        self.batch_size = batch_size
        self.models = []
        for device in devices:
            print(f"loading E5 model on {device}")
            self.models.append(SentenceTransformer(model_name, device=device))

    def encode(self, texts: list[str], description: str) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        if len(self.models) == 1:
            result = self.models[0].encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return np.asarray(result, dtype=np.float32)

        partitions = np.array_split(np.arange(len(texts)), len(self.models))

        def encode_partition(model_index: int):
            indices = partitions[model_index].tolist()
            if not indices:
                return model_index, indices, np.empty((0, 384), dtype=np.float32)
            values = self.models[model_index].encode(
                [texts[index] for index in indices],
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return model_index, indices, np.asarray(values, dtype=np.float32)

        result = np.empty((len(texts), 384), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            jobs = [executor.submit(encode_partition, index) for index in range(len(self.models))]
            for job in tqdm(jobs, desc=description):
                _, indices, values = job.result()
                result[indices] = values
        return result

    def close(self):
        for model in self.models:
            model.to("cpu")
        self.models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_hybrid_features(s1: pd.DataFrame, pool: pd.DataFrame, pairs, encoder: DualE5, description: str):
    s1_lookup = s1.set_index("entity_id")
    left_ids = list(dict.fromkeys(left_id for left_id, _ in pairs))
    right_positions = list(dict.fromkeys(position for _, position in pairs))
    left_rows = [s1_lookup.loc[entity_id] for entity_id in left_ids]
    selected_pool = pool.iloc[right_positions]
    right_rows = [row for _, row in selected_pool.iterrows()]
    left_row_map = dict(zip(left_ids, left_rows))
    right_row_map = dict(zip(right_positions, right_rows))
    left_embeddings = encoder.encode(
        [record_text(row, "query") for row in left_rows],
        f"{description}: E5 S1",
    )
    right_embeddings = encoder.encode(
        [record_text(row, "passage") for row in right_rows],
        f"{description}: E5 pool",
    )
    left_embedding_map = dict(zip(left_ids, left_embeddings))
    right_embedding_map = dict(zip(right_positions, right_embeddings))
    rows = []
    for left_id, pool_position in tqdm(pairs, total=len(pairs), desc=f"{description}: features"):
        left = left_row_map[left_id]
        right = right_row_map[pool_position]
        base = features(left, right)
        cosine = float(np.dot(left_embedding_map[left_id], right_embedding_map[pool_position]))
        rows.append(base + [cosine])
    del selected_pool, left_rows, right_rows, left_row_map, right_row_map
    del left_embeddings, right_embeddings, left_embedding_map, right_embedding_map
    return np.asarray(rows, dtype=np.float32)


def find_files(root: Path):
    files = {}
    names_by_split = {
        "train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"],
        "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"],
    }
    for split, names in names_by_split.items():
        for name in names:
            found = next(root.rglob(name), None)
            if found is None:
                raise FileNotFoundError(f"Cannot find {name} below {root}")
            files[(split, name)] = found
    return files


def main():
    parser = argparse.ArgumentParser(description="Approach 6: B5 + dual-GPU E5 + LightGBM")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="/kaggle/working/output6")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--s1-limit", type=int, default=None)
    parser.add_argument("--pool-limit", type=int, default=None)
    parser.add_argument("--token-cap", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--s1-chunk-size", type=int, default=5000)
    parser.add_argument("--max-train-pairs", type=int, default=2_000_000)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--devices", default="0,1", help="CUDA device IDs, for example 0,1")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--no-telegram", action="store_true")
    parser.add_argument("--telegram-chunk-mb", type=int, default=45)
    args = parser.parse_args()

    root = Path(args.data_root) if args.data_root else Path("/kaggle/input")
    if not root.exists():
        root = ROOT / "dataset"
    files = find_files(root)
    limit_s1 = args.s1_limit if args.dry_run else None
    limit_pool = args.pool_limit if args.dry_run else None
    devices = parse_devices(args.devices)
    encoder = DualE5(args.model_name, devices, args.embedding_batch_size)

    print("loading and normalizing training S1")
    train_s1 = load(files[("train", "train_source1.tsv")], limit_s1)
    print("loading and normalizing training pool")
    train_pool = pd.concat(
        [load(files[("train", name)], limit_pool) for name in ("train_source2.tsv", "train_source3.tsv")],
        ignore_index=True,
    )
    truth = truth_map(files[("train", "train_ground_truth.tsv")])
    train_pairs = build_pairs(train_s1, train_pool, args.token_cap, args.top_k, args.max_train_pairs)
    X = build_hybrid_features(train_s1, train_pool, train_pairs, encoder, "training")
    pool_ids = train_pool["entity_id"].to_numpy()
    y = np.asarray([int(pool_ids[position] in truth.get(s1_id, set())) for s1_id, position in train_pairs])
    print(f"training pairs={len(train_pairs):,} positives={int(y.sum()):,}")

    model = train_model(X, y, args.threads)
    stage1_probability = positive_probability(model, X)
    X_stage2 = np.column_stack([X, contextual_features(train_pairs, stage1_probability)])
    model_stage2 = train_model(X_stage2, y, args.threads)
    stage2_probability = positive_probability(model_stage2, X_stage2)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(stage2_probability, y)
    calibrated = calibrator.predict(stage2_probability)
    train_pair_ids = [(s1_id, pool_ids[position]) for s1_id, position in train_pairs]
    thresholds = np.arange(0.05, 0.96, 0.01)
    threshold = max(
        thresholds,
        key=lambda value: f05(train_s1.entity_id.tolist(), train_pair_ids, calibrated, truth, float(value)),
    )
    print(f"calibrated threshold={threshold:.2f}")

    del train_s1, train_pool, train_pairs, X, y, stage1_probability, X_stage2, stage2_probability, calibrated
    gc.collect()
    print("released training data before test inference")
    print("loading and normalizing test S1")
    test_s1 = load(files[("test", "test_source1.tsv")], limit_s1)
    print("loading and normalizing test pool")
    test_pool = pd.concat(
        [load(files[("test", name)], limit_pool) for name in ("test_source2.tsv", "test_source3.tsv")],
        ignore_index=True,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    test_index = make_index(test_pool, args.token_cap)
    test_pool_ids = test_pool["entity_id"].to_numpy()
    global_owner = {}
    total_pairs = 0

    with (output / "candidate_pairs.tsv").open("w", encoding="utf-8") as candidate_file:
        candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
        for start in range(0, len(test_s1), args.s1_chunk_size):
            chunk_s1 = test_s1.iloc[start:start + args.s1_chunk_size]
            pairs = generate_pairs(chunk_s1, test_pool, test_index, args.top_k)
            total_pairs += len(pairs)
            candidate_map = defaultdict(list)
            for s1_id, position in pairs:
                candidate_map[s1_id].append(test_pool_ids[position])
            for s1_id in tqdm(chunk_s1.entity_id, total=len(chunk_s1), desc="write candidate output"):
                candidate_file.write(f"{s1_id}\t{','.join(candidate_map[s1_id])}\n")

            if pairs:
                test_X = build_hybrid_features(chunk_s1, test_pool, pairs, encoder, "test")
                stage1_probability = positive_probability(model, test_X)
                test_X_stage2 = np.column_stack([test_X, contextual_features(pairs, stage1_probability)])
                probability = calibrator.predict(positive_probability(model_stage2, test_X_stage2))
                pair_ids = [(s1_id, test_pool_ids[position]) for s1_id, position in pairs]
                candidates_by_s1 = defaultdict(list)
                for pair, value in zip(pair_ids, probability):
                    if value >= threshold:
                        candidates_by_s1[pair[0]].append((float(value), pair[1]))
                for s1_id, values in candidates_by_s1.items():
                    for value, pool_id in sorted(values, reverse=True):
                        if pool_id not in global_owner or value > global_owner[pool_id][0]:
                            global_owner[pool_id] = (value, s1_id)
                del test_X, stage1_probability, test_X_stage2, probability
            del chunk_s1, pairs, candidate_map
            gc.collect()
            print(f"completed test chunk {min(start + args.s1_chunk_size, len(test_s1)):,}/{len(test_s1):,}")

    matches = defaultdict(list)
    for pool_id, (_, s1_id) in global_owner.items():
        matches[s1_id].append(pool_id)
    with (output / "matching_results.tsv").open("w", encoding="utf-8") as matching_file:
        matching_file.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in tqdm(test_s1.entity_id, total=len(test_s1), desc="write matching output"):
            matching_file.write(f"{s1_id}\t{','.join(matches[s1_id])}\n")

    encoder.close()
    if not args.no_telegram:
        from approach_5_b5.telegram_sender import send_artifacts
        send_artifacts(output / "matching_results.tsv", output / "candidate_pairs.tsv", args.telegram_chunk_mb)
    print(f"test pairs={total_pairs:,} outputs={output}")


if __name__ == "__main__":
    main()
