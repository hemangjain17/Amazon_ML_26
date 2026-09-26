"""B5-compatible country-partitioned blocking and LightGBM matcher."""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rapidfuzz.fuzz import ratio, token_set_ratio
from sklearn.isotonic import IsotonicRegression
from sklearn.dummy import DummyClassifier
from unidecode import unidecode
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"entity_id", "business_name", "business_address", "country"}
LEGAL = {
    "private limited": "pvt ltd", "privatelimited": "pvt ltd", "limited": "ltd",
    "incorporated": "inc", "corporation": "corp", "company": "co",
}
STATE_RE = re.compile(r"(?:^| )([a-z]{2}|[a-z ]{4,25})(?:$| )")
PIN_RE = re.compile(r"\b(\d{5,6})\b")
HOUSE_RE = re.compile(r"(?:^| )([0-9]{1,6})(?: |$)")


def clean(value: object) -> str:
    text = unidecode(str(value or "")).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    for old, new in LEGAL.items():
        text = text.replace(old, new)
    text = re.sub(r"([a-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([a-z])", r"\1 \2", text)
    return " ".join(text.split())


def skeleton(text: str) -> str:
    return re.sub(r"[aeiou]", "", text.replace(" ", ""))[:32]


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy().fillna("")
    frame["name_n"] = [clean(value) for value in tqdm(frame["business_name"], desc="normalize names", leave=False)]
    frame["addr_n"] = [clean(value) for value in tqdm(frame["business_address"], desc="normalize addresses", leave=False)]
    frame["country_n"] = [clean(value) for value in tqdm(frame["country"], desc="normalize countries", leave=False)]
    frame["name_skel"] = frame["name_n"].map(skeleton)
    frame["house"] = frame["addr_n"].str.extract(HOUSE_RE, expand=False).fillna("")
    frame["pin"] = frame["addr_n"].str.extract(PIN_RE, expand=False).fillna("")
    frame["name_tokens"] = frame["name_n"].str.split()
    frame["addr_tokens"] = frame["addr_n"].str.split()
    frame["state"] = frame["addr_n"].str.extract(STATE_RE, expand=False).fillna("")
    return frame


def load(path: Path, limit: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=limit)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return enrich(frame)


def make_index(pool: pd.DataFrame, cap: int = 400):
    counts = defaultdict(int)
    for row in tqdm(pool.itertuples(), total=len(pool), desc="count blocking tokens"):
        keys = set(row.name_tokens + row.addr_tokens)
        for key in keys:
            counts[(row.country_n, key)] += 1
    index = defaultdict(list)
    prefix_counts = defaultdict(int)
    for pos, row in enumerate(tqdm(pool.itertuples(), total=len(pool), desc="build blocking index")):
        keys = set(row.name_tokens + row.addr_tokens)
        for key in keys:
            if counts[(row.country_n, key)] <= cap:
                index[(row.country_n, key)].append(pos)
        if row.house and counts[(row.country_n, row.house)] <= cap:
            index[(row.country_n, "house:" + row.house)].append(pos)
        prefix_counts[(row.country_n, row.name_n[:5])] += 1
    prefix_index = defaultdict(list)
    for pos, row in enumerate(pool.itertuples()):
        prefix = (row.country_n, row.name_n[:5])
        if prefix_counts[prefix] <= cap * 4:
            prefix_index[prefix].append(pos)
    return index, prefix_index


def candidates(row, pool_records, index, prefix_index, top_k: int = 150) -> list[int]:
    hits = set()
    keys = set(row.name_tokens + row.addr_tokens)
    for key in keys:
        hits.update(index.get((row.country_n, key), []))
    if row.house:
        hits.update(index.get((row.country_n, "house:" + row.house), []))
    if not hits:
        hits.update(prefix_index.get((row.country_n, row.name_n[:5]), []))
    scored = []
    for pos in hits:
        other_name, other_addr, other_state = pool_records[pos]
        if row.state and other_state and row.state != other_state:
            continue
        score = 0.55 * ratio(row.name_n, other_name) + 0.45 * ratio(row.addr_n, other_addr)
        scored.append((score, pos))
    scored.sort(reverse=True)
    return [pos for _, pos in scored[:top_k]]


def features(left: pd.Series, right: pd.Series, stage1: float = 0.0, rank: int = 0, gap: float = 0.0, support: int = 0):
    name = ratio(left.name_n, right.name_n) / 100
    addr = ratio(left.addr_n, right.addr_n) / 100
    name_tok = token_set_ratio(left.name_n, right.name_n) / 100
    addr_tok = token_set_ratio(left.addr_n, right.addr_n) / 100
    return [name, addr, name_tok, addr_tok, float(bool(left.pin and left.pin == right.pin)), float(bool(left.house and left.house == right.house)), float(left.name_skel == right.name_skel), stage1, rank, gap, support]


def generate_pairs(s1: pd.DataFrame, pool: pd.DataFrame, index, top_k: int, max_pairs: int | None = None):
    print(f"building candidates for {len(s1):,} S1 rows against {len(pool):,} pool rows")
    pairs = []
    pool_records = list(pool[["name_n", "addr_n", "state"]].itertuples(index=False, name=None))
    blocking_index, prefix_index = index
    for left in tqdm(s1.itertuples(index=False), total=len(s1), desc="generate candidates"):
        for pos in candidates(left, pool_records, blocking_index, prefix_index, top_k):
            pairs.append((left.entity_id, int(pos)))
            if max_pairs and len(pairs) >= max_pairs:
                return pairs
    return pairs


def build_pairs(s1: pd.DataFrame, pool: pd.DataFrame, cap: int, top_k: int, max_pairs: int | None = None):
    return generate_pairs(s1, pool, make_index(pool, cap), top_k, max_pairs)


def build_feature_matrix(s1: pd.DataFrame, pool: pd.DataFrame, pairs, description: str):
    lookup = s1.set_index("entity_id")
    rows = []
    for s1_id, pool_pos in tqdm(pairs, total=len(pairs), desc=description):
        rows.append(features(lookup.loc[s1_id], pool.iloc[pool_pos]))
    return np.asarray(rows, dtype=np.float32)


def train_model(X, y, threads: int):
    print(f"training LightGBM on {len(X):,} pairs")
    if len(y) < 2 or np.unique(y).size < 2:
        print("warning: insufficient class diversity; using a smoke-test dummy model")
        model = DummyClassifier(strategy="prior")
        model.fit(X, y)
        return model
    model = LGBMClassifier(n_estimators=350, num_leaves=63, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=-1, n_jobs=threads)
    model.fit(X, y)
    return model


def positive_probability(model, X):
    probabilities = model.predict_proba(X)
    if probabilities.shape[1] == 1:
        return np.full(len(X), float(model.classes_[0] == 1), dtype=np.float32)
    return probabilities[:, 1]


def contextual_features(pairs, probabilities):
    grouped = defaultdict(list)
    for index, (s1_id, _) in enumerate(pairs):
        grouped[s1_id].append(index)
    extra = np.zeros((len(pairs), 3), dtype=np.float32)
    for indices in tqdm(grouped.values(), total=len(grouped), desc="build context features", leave=False):
        ordered = sorted(indices, key=lambda index: probabilities[index], reverse=True)
        best = probabilities[ordered[0]] if ordered else 0.0
        second = probabilities[ordered[1]] if len(ordered) > 1 else 0.0
        for rank, index in enumerate(ordered):
            extra[index] = (rank / max(len(ordered), 1), best - second, len(ordered))
    return np.column_stack([probabilities, extra])


def truth_map(path: Path):
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {r.source1_entity_id: set(filter(None, r.matched_entity_ids.split(","))) for r in gt.itertuples()}


def select_matches(s1_ids, pairs, probabilities, threshold):
    candidates = defaultdict(list)
    for (s1_id, pool_id), probability in zip(pairs, probabilities):
        if probability >= threshold:
            candidates[s1_id].append((probability, pool_id))
    owner = {}
    for s1_id, values in candidates.items():
        for probability, pool_id in sorted(values, reverse=True):
            if pool_id not in owner or probability > owner[pool_id][0]:
                owner[pool_id] = (probability, s1_id)
    result = defaultdict(list)
    for probability, s1_id in owner.values():
        result[s1_id].append(probability)
    return result, owner


def f05(s1_ids, pairs, probabilities, truth, threshold):
    selected, owner = select_matches(s1_ids, pairs, probabilities, threshold)
    by_s1 = defaultdict(set)
    for pool_id, (_, s1_id) in owner.items():
        by_s1[s1_id].add(pool_id)
    values = []
    for s1_id in s1_ids:
        actual, pred = truth.get(s1_id, set()), by_s1.get(s1_id, set())
        if not actual and not pred:
            values.append(1.0)
        elif not actual or not pred:
            values.append(0.0)
        else:
            tp = len(actual & pred); precision = tp / len(pred); recall = tp / len(actual)
            values.append(1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-dir", default="/kaggle/working/output")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--s1-limit", type=int, default=None)
    ap.add_argument("--pool-limit", type=int, default=None)
    ap.add_argument("--token-cap", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=150)
    ap.add_argument("--s1-chunk-size", type=int, default=25000)
    ap.add_argument("--max-train-pairs", type=int, default=2000000)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--no-telegram", action="store_true", help="Do not upload completed outputs")
    ap.add_argument("--telegram-chunk-mb", type=int, default=45)
    args = ap.parse_args()
    root = Path(args.data_root) if args.data_root else Path("/kaggle/input")
    if not root.exists(): root = ROOT / "dataset"
    files = {}
    for split, names in {"train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"], "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]}.items():
        for name in names:
            found = next(root.rglob(name), None)
            if found is None: raise FileNotFoundError(f"Cannot find {name} below {root}")
            files[(split, name)] = found
    limit_s1 = args.s1_limit if args.dry_run else None
    limit_pool = args.pool_limit if args.dry_run else None
    print("loading and normalizing training S1")
    train_s1 = load(files[("train", "train_source1.tsv")], limit_s1)
    print("loading and normalizing training pool")
    train_pool = pd.concat([load(files[("train", n)], limit_pool) for n in ["train_source2.tsv", "train_source3.tsv"]], ignore_index=True)
    truth = truth_map(files[("train", "train_ground_truth.tsv")])
    pairs = build_pairs(train_s1, train_pool, args.token_cap, args.top_k, args.max_train_pairs)
    X = build_feature_matrix(train_s1, train_pool, pairs, "build training features")
    y = np.array([int(train_pool.iloc[b].entity_id in truth.get(a, set())) for a, b in pairs])
    print(f"training pairs={len(pairs):,} positives={int(y.sum()):,}")
    model = train_model(X, y, args.threads); raw = positive_probability(model, X)
    X_stage2 = np.column_stack([X, contextual_features(pairs, raw)])
    model_stage2 = train_model(X_stage2, y, args.threads)
    stage2_raw = positive_probability(model_stage2, X_stage2)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(stage2_raw, y)
    calibrated = calibrator.predict(stage2_raw)
    thresholds = np.arange(0.05, 0.96, 0.01)
    train_pair_ids = [(a, train_pool.iloc[b].entity_id) for a, b in pairs]
    threshold = max(thresholds, key=lambda t: f05(train_s1.entity_id.tolist(), train_pair_ids, calibrated, truth, float(t)))
    print(f"validation threshold={threshold:.2f}")
    del train_s1, train_pool, pairs, X, y, raw, X_stage2, stage2_raw, calibrated
    gc.collect()
    print("released training data before test inference")
    print("loading and normalizing test S1")
    test_s1 = load(files[("test", "test_source1.tsv")], limit_s1)
    print("loading and normalizing test pool")
    test_pool = pd.concat([load(files[("test", n)], limit_pool) for n in ["test_source2.tsv", "test_source3.tsv"]], ignore_index=True)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    test_index = make_index(test_pool, args.token_cap)
    global_owner = {}
    total_test_pairs = 0
    with (out / "candidate_pairs.tsv").open("w", encoding="utf-8") as candidate_file, (out / "matching_results.tsv").open("w", encoding="utf-8") as matching_file:
        candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
        matching_file.write("source1_entity_id\tmatched_entity_ids\n")
        for start in range(0, len(test_s1), args.s1_chunk_size):
            chunk_s1 = test_s1.iloc[start:start + args.s1_chunk_size]
            test_pairs = generate_pairs(chunk_s1, test_pool, test_index, args.top_k)
            total_test_pairs += len(test_pairs)
            if not test_pairs:
                for s1_id in tqdm(chunk_s1.entity_id, total=len(chunk_s1), desc="write empty candidate output"):
                    candidate_file.write(f"{s1_id}\t\n")
                print(f"completed empty test chunk {min(start + args.s1_chunk_size, len(test_s1)):,}/{len(test_s1):,}")
                del chunk_s1, test_pairs
                gc.collect()
                continue
            test_X = build_feature_matrix(chunk_s1, test_pool, test_pairs, "build test features")
            test_raw = positive_probability(model, test_X)
            test_X_stage2 = np.column_stack([test_X, contextual_features(test_pairs, test_raw)])
            test_prob = calibrator.predict(positive_probability(model_stage2, test_X_stage2))
            test_pair_ids = [(a, test_pool.iloc[b].entity_id) for a, b in test_pairs]
            _, owner = select_matches(chunk_s1.entity_id.tolist(), test_pair_ids, test_prob, threshold)
            candidate_map = defaultdict(list)
            for a, b in test_pairs: candidate_map[a].append(test_pool.iloc[b].entity_id)
            for s1_id in tqdm(chunk_s1.entity_id, total=len(chunk_s1), desc="write candidate output"):
                candidate_file.write(f"{s1_id}\t{','.join(candidate_map[s1_id])}\n")
            for pool_id, assignment in owner.items():
                if pool_id not in global_owner or assignment[0] > global_owner[pool_id][0]:
                    global_owner[pool_id] = assignment
            del chunk_s1, test_pairs, test_X, test_raw, test_X_stage2, test_prob, owner
            gc.collect()
            print(f"completed test chunk {min(start + args.s1_chunk_size, len(test_s1)):,}/{len(test_s1):,}")
    with (out / "matching_results.tsv").open("w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        matches = defaultdict(list)
        for pool_id, (_, s1_id) in global_owner.items():
            matches[s1_id].append(pool_id)
        for s1_id in tqdm(test_s1.entity_id, total=len(test_s1), desc="write matching output"):
            f.write(f"{s1_id}\t{','.join(matches[s1_id])}\n")
    if not args.no_telegram:
        from telegram_sender import send_artifacts
        send_artifacts(out / "matching_results.tsv", out / "candidate_pairs.tsv", args.telegram_chunk_mb)
    print(f"test pairs={total_test_pairs:,} outputs={out}")


if __name__ == "__main__": main()
