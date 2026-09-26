"""Fast, memory-bounded inference pipeline for Approach 1 (Kaggle Steps 2-4).

Replaces the slow notebook path (per-row padding to 128 tokens, single GPU fp32
encoding, CPU FAISS flat search over millions of vectors, pure-Python pairwise
features over ~52M pairs, CatBoost trained on dummy labels) with:

* one cleaning pass per entity (multiprocessing, deduplicated),
* length-sorted, dynamically padded fp16 encoding on every visible GPU,
* exact brute-force top-k inner-product search on GPU (country-partitioned),
* vectorised rapidfuzz features computed in bounded chunks,
* a CatBoost reranker trained on labelled train-split candidates, with the
  decision threshold tuned for Macro F0.5 on held-out train queries.

The text fed to the VAE is byte-identical to ``format_entity_string`` used in
training (checked by ``assert_text_parity``).
"""

import copy
import gc
import glob
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer

try:
    from approach_1_contrastiveVAE.config import path_config, model_config, blocking_config, reranker_config
    from approach_1_contrastiveVAE.src.model import ContrastiveVAE
    from approach_1_contrastiveVAE.src.normalization import clean_text, format_entity_string
except (ImportError, ValueError):
    try:
        from config import path_config, model_config, blocking_config, reranker_config
        from src.model import ContrastiveVAE
        from src.normalization import clean_text, format_entity_string
    except (ImportError, ValueError):
        from ..config import path_config, model_config, blocking_config, reranker_config
        from .model import ContrastiveVAE
        from .normalization import clean_text, format_entity_string


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        log(f"{self.label} ...")
        return self

    def __exit__(self, *exc):
        log(f"{self.label} done in {time.time() - self.start:.1f}s")


# ---------------------------------------------------------------------------
# Entities: cleaned once, stored as compact numpy arrays
# ---------------------------------------------------------------------------

@dataclass
class Entities:
    ids: np.ndarray        # object[str]
    country: np.ndarray    # int16 codes into COUNTRY_VOCAB
    name: np.ndarray       # object[str], clean_text(name)
    addr: np.ndarray       # object[str], clean_text(address)
    pin: np.ndarray        # int64 hash of the 5/6 digit PIN/ZIP (0 = missing)
    num: np.ndarray        # int64 hash of the first number in the address (0 = missing)

    def __len__(self):
        return len(self.ids)

    def subset(self, rows: np.ndarray) -> "Entities":
        return Entities(*(getattr(self, f)[rows] for f in ("ids", "country", "name", "addr", "pin", "num")))


# Country strings exactly as EntityInferenceDataset saw them (fillna("US")).
COUNTRY_VOCAB: List[str] = []


def _country_codes(values: pd.Series) -> np.ndarray:
    codes = np.empty(len(values), dtype=np.int16)
    raw = values.to_numpy(dtype=object)
    uniques, inverse = np.unique(raw.astype(str), return_inverse=True)
    for u_idx, value in enumerate(uniques):
        if value not in COUNTRY_VOCAB:
            COUNTRY_VOCAB.append(value)
        codes[inverse == u_idx] = COUNTRY_VOCAB.index(value)
    return codes


def _clean_chunk(texts: List[str]) -> List[str]:
    return [clean_text(t) for t in texts]


def _clean_column(values: pd.Series, pool) -> np.ndarray:
    """clean_text over a column, computed once per unique string across worker processes."""
    codes, uniques = pd.factorize(values, sort=False)
    uniques = list(uniques)
    n_chunks = max(1, min(len(uniques) // 20000, 256))
    chunks = [uniques[i::n_chunks] for i in range(n_chunks)]
    results = pool.map(_clean_chunk, chunks) if pool is not None else [_clean_chunk(c) for c in chunks]
    cleaned = np.empty(len(uniques), dtype=object)
    for i, res in enumerate(results):
        cleaned[i::n_chunks] = res
    return cleaned[codes]


def _hash_extract(addresses: pd.Series, pattern: str) -> np.ndarray:
    extracted = addresses.str.extract(pattern, expand=False)
    hashed = pd.util.hash_pandas_object(extracted.fillna(""), index=False).to_numpy().view(np.int64).copy()
    hashed[extracted.isna().to_numpy()] = 0
    return hashed


def _raw_text(values: pd.Series) -> pd.Series:
    """Mirrors the dataset/format_entity_string handling: NaN -> "", literal "nan" (any case) -> ""."""
    values = values.fillna("").astype(str)
    return values.mask(values.str.lower() == "nan", "")


def load_entities(paths: List[str], pool=None, sample_rows: Optional[int] = None, seed: int = 42) -> Entities:
    """Reads one or more source TSVs and returns cleaned, compact entity arrays."""
    frames = []
    for path in paths:
        df = pd.read_csv(path, sep="\t", dtype=str)
        # Guard against quote characters merging rows: every line must become a row.
        with open(path, "rb") as handle:
            n_lines = sum(1 for _ in handle) - 1
        if len(df) != n_lines:
            raise RuntimeError(f"{path}: pandas read {len(df):,} rows but file has {n_lines:,} lines")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    del frames
    if sample_rows is not None and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=seed).reset_index(drop=True)
    entities = load_entities_from_frame(df, pool)
    del df
    gc.collect()
    return entities


def make_clean_pool(processes: Optional[int] = None):
    """Fork-based worker pool for text cleaning (create before any CUDA threads start)."""
    processes = processes or os.cpu_count() or 1
    if processes <= 1:
        return None
    return get_context("fork").Pool(processes)


def entity_texts(ent: Entities, rows: np.ndarray) -> List[str]:
    """Rebuilds format_entity_string(name, address, country) from pre-cleaned fields."""
    texts = []
    for r in rows:
        code = COUNTRY_VOCAB[ent.country[r]].strip().upper()
        if code:
            texts.append(f"[{code}] {ent.name[r]} | {ent.addr[r]}")
        else:
            texts.append(f"{ent.name[r]} | {ent.addr[r]}")
    return texts


def assert_text_parity(paths: List[str], n: int = 20000, seed: int = 0):
    """Checks the fast text path is byte-identical to training's format_entity_string."""
    df = pd.concat([pd.read_csv(p, sep="\t", dtype=str, nrows=n) for p in paths], ignore_index=True)
    non_ascii = df["business_name"].fillna("").map(lambda s: not s.isascii())
    df = pd.concat([df.sample(n=min(n, len(df)), random_state=seed), df[non_ascii].head(2000)], ignore_index=True)
    df.loc[:2, "business_name"] = ["nan", "NaN", "Nan"]  # exercise the literal-"nan" rule
    fast = entity_texts(load_entities_from_frame(df), np.arange(len(df)))
    names = df["business_name"].fillna("").astype(str).tolist()
    addrs = df["business_address"].fillna("").astype(str).tolist()
    countries = df["country"].fillna("US").astype(str).tolist()
    reference = [format_entity_string(a, b, c) for a, b, c in zip(names, addrs, countries)]
    mismatches = [(x, y) for x, y in zip(fast, reference) if x != y]
    if mismatches:
        raise AssertionError(f"{len(mismatches)} text mismatches vs format_entity_string, e.g. {mismatches[:3]}")
    log(f"Text parity OK on {len(fast):,} rows ({int(non_ascii.sum()):,} non-ASCII)")


def load_entities_from_frame(df: pd.DataFrame, pool=None) -> Entities:
    names = _raw_text(df["business_name"])
    addrs = _raw_text(df["business_address"])
    return Entities(
        ids=df["entity_id"].astype(str).to_numpy(dtype=object),
        country=_country_codes(df["country"].fillna("US").astype(str)),
        name=_clean_column(names, pool),
        addr=_clean_column(addrs, pool),
        pin=_hash_extract(addrs, r"\b(\d{5,6})\b"),
        num=_hash_extract(addrs, r"(\d+)"),
    )


# ---------------------------------------------------------------------------
# Model loading and multi-GPU encoding
# ---------------------------------------------------------------------------

def find_checkpoint(explicit: Optional[str] = None) -> str:
    if explicit and os.path.exists(explicit):
        return explicit
    roots = ["/kaggle/input", "/kaggle/working", path_config.models_dir]
    for pattern in ("best_contrastive_vae.pt", "*contrastive*vae*.pt", "*.pt", "*.pth"):
        for root in roots:
            hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
            if hits:
                return hits[0]
    raise FileNotFoundError("No Contrastive VAE checkpoint found under /kaggle/input or /kaggle/working. "
                            "Attach the weights dataset or set CHECKPOINT_PATH.")


def _load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except (ModuleNotFoundError, AttributeError):
        # The checkpoint pickles ModelConfig under the module path used at training time.
        cfg_module = sys.modules[type(model_config).__module__]
        for alias in ("config", "approach_1_contrastiveVAE.config"):
            sys.modules.setdefault(alias, cfg_module)
        return torch.load(path, map_location="cpu", weights_only=False)


def load_models(checkpoint_path: str) -> List[Tuple[torch.nn.Module, torch.device]]:
    """Loads the VAE once and replicates it on every visible GPU (or CPU)."""
    checkpoint = _load_checkpoint(checkpoint_path)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    del checkpoint

    base = ContrastiveVAE(backbone_name=model_config.backbone_name, latent_dim=model_config.latent_dim)
    base.load_state_dict(state)
    base.eval()
    del state

    if not torch.cuda.is_available():
        return [(base, torch.device("cpu"))]
    replicas = []
    for gpu in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{gpu}")
        replica = base if gpu == 0 else copy.deepcopy(base)
        replicas.append((replica.to(device), device))
    log(f"Loaded {checkpoint_path} on {[str(d) for _, d in replicas]}")
    return replicas


def _run_threads(targets):
    errors = []

    def wrap(fn, *args):
        try:
            fn(*args)
        except BaseException as exc:  # surfaced in the main thread below
            errors.append(exc)

    threads = [threading.Thread(target=wrap, args=(fn, *args), daemon=True) for fn, *args in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def encode_entities(ent: Entities, models, batch_size: int = 1024) -> np.ndarray:
    """L2-normalised latent mu for every entity, fp16, in original row order."""
    n = len(ent)
    out = np.empty((n, model_config.latent_dim), dtype=np.float16)
    lengths = np.fromiter((len(a) + len(b) for a, b in zip(ent.name, ent.addr)), dtype=np.int32, count=n)
    order = np.argsort(lengths, kind="stable")[::-1]  # longest first: OOM surfaces immediately
    batches = [order[i:i + batch_size] for i in range(0, n, batch_size)]
    progress = tqdm(total=n, desc="Encoding", unit="ent", mininterval=5)
    lock = threading.Lock()

    def producer(worker: int, tokenizer, q: "queue.Queue"):
        for b in batches[worker::len(models)]:
            enc = tokenizer(entity_texts(ent, b), padding=True, truncation=True,
                            max_length=model_config.max_seq_length, return_tensors="pt")
            if torch.cuda.is_available():
                enc = {k: v.pin_memory() for k, v in enc.items()}
            q.put((b, enc))
        q.put(None)

    def consumer(worker: int):
        model, device = models[worker]
        tokenizer = AutoTokenizer.from_pretrained(model_config.backbone_name)
        q: "queue.Queue" = queue.Queue(maxsize=8)
        prod = threading.Thread(target=producer, args=(worker, tokenizer, q), daemon=True)
        prod.start()
        use_amp = device.type == "cuda"
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            while True:
                item = q.get()
                if item is None:
                    break
                rows, enc = item
                ids = enc["input_ids"].to(device, non_blocking=True)
                mask = enc["attention_mask"].to(device, non_blocking=True)
                _, mu, _ = model.encode(ids, mask)
                z = F.normalize(mu.float(), p=2, dim=1).half().cpu().numpy()
                out[rows] = z
                with lock:
                    progress.update(len(rows))
        prod.join()

    _run_threads([(consumer, w) for w in range(len(models))])
    progress.close()
    return out


# ---------------------------------------------------------------------------
# GPU brute-force top-k search (exact inner product, country partitioned)
# ---------------------------------------------------------------------------

def search_topk(
    q_ent: Entities, q_emb: np.ndarray,
    p_ent: Entities, p_emb: np.ndarray,
    top_k: int, models,
    q_block: int = 2048, p_block: int = 1 << 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pool row index [nq, k] with -1 padding, cosine sim [nq, k]) sorted by sim desc."""
    nq = len(q_ent)
    idx_out = np.full((nq, top_k), -1, dtype=np.int32)
    sim_out = np.full((nq, top_k), np.nan, dtype=np.float32)
    devices = [d for _, d in models]

    for code in np.unique(q_ent.country):
        q_rows = np.flatnonzero(q_ent.country == code)
        p_rows = np.flatnonzero(p_ent.country == code)
        country = COUNTRY_VOCAB[code]
        if len(p_rows) == 0:
            log(f"Search {country}: {len(q_rows):,} queries but no candidates; left empty")
            continue
        k = min(top_k, len(p_rows))
        log(f"Search {country}: {len(q_rows):,} queries x {len(p_rows):,} candidates on {len(devices)} device(s)")
        pool_cpu = torch.from_numpy(p_emb[p_rows])
        progress = tqdm(total=len(q_rows), desc=f"Top-{k} {country}", unit="q", mininterval=5)
        lock = threading.Lock()

        def worker(device, my_rows):
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            pool = pool_cpu.to(device=device, dtype=dtype)
            with torch.inference_mode():
                for s in range(0, len(my_rows), q_block):
                    rows = my_rows[s:s + q_block]
                    q = torch.from_numpy(q_emb[rows]).to(device=device, dtype=dtype)
                    best_s = best_i = None
                    for ps in range(0, pool.shape[0], p_block):
                        scores = q @ pool[ps:ps + p_block].T
                        ts, ti = scores.topk(min(k, scores.shape[1]), dim=1)
                        ti += ps
                        if best_s is not None:
                            ts, sel = torch.cat([best_s, ts], 1).topk(k, dim=1)
                            ti = torch.gather(torch.cat([best_i, ti], 1), 1, sel)
                        best_s, best_i = ts, ti
                        del scores
                    # Re-score the k winners in fp32 for an exact similarity feature.
                    exact = torch.einsum("qd,qkd->qk", q.float(), pool[best_i].float())
                    exact, order = exact.sort(dim=1, descending=True)
                    best_i = torch.gather(best_i, 1, order)
                    idx_out[rows, :k] = p_rows[best_i.cpu().numpy()]
                    sim_out[rows, :k] = exact.cpu().numpy()
                    with lock:
                        progress.update(len(rows))
            del pool
            if device.type == "cuda":
                torch.cuda.empty_cache()

        shards = np.array_split(q_rows, len(devices))
        _run_threads([(worker, d, shard) for d, shard in zip(devices, shards) if len(shard)])
        progress.close()
        del pool_cpu
        gc.collect()
    return idx_out, sim_out


def write_candidate_pairs(path: str, q_ent: Entities, p_ent: Entities, idx: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, row in zip(q_ent.ids, idx):
            f.write(f"{s1_id}\t{','.join(p_ent.ids[row[row >= 0]])}\n")
    log(f"Wrote {path} ({len(q_ent):,} rows)")


# ---------------------------------------------------------------------------
# Pairwise features (rapidfuzz, chunked)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "latent_sim", "rank", "sim_gap_to_top1", "top1_sim", "sim_gap_to_next",
    "name_jaro_winkler", "name_ratio", "name_token_set", "name_token_sort", "name_partial",
    "addr_ratio", "addr_token_set", "addr_token_sort",
    "pin_match", "addr_number_match", "is_s2", "name_len_ratio", "addr_len_ratio", "any_name_empty",
]


def _match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where((a == 0) | (b == 0), -1.0, (a == b).astype(np.float32)).astype(np.float32)


def _len_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    hi = np.maximum(a, b)
    return np.where(hi == 0, 0.0, np.minimum(a, b) / np.maximum(hi, 1)).astype(np.float32)


def pair_features(
    q_ent: Entities, p_ent: Entities,
    q_rows: np.ndarray, idx: np.ndarray, sim: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features for every valid (query, candidate) pair of the given query rows.

    Returns (X float32 [n_pairs, F], pair query rows, pair pool rows).
    """
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz.process import cpdist

    sub_idx, sub_sim = idx[q_rows], sim[q_rows]
    valid = sub_idx >= 0
    k = idx.shape[1]
    rank = np.broadcast_to(np.arange(k, dtype=np.float32), sub_idx.shape)
    top1 = np.broadcast_to(sub_sim[:, :1], sub_sim.shape)
    nxt = np.concatenate([sub_sim[:, 1:], np.full((len(q_rows), 1), np.nan, np.float32)], axis=1)
    pq = np.broadcast_to(q_rows[:, None], sub_idx.shape)[valid]
    pp = sub_idx[valid]
    s = sub_sim[valid]

    qn, pn = q_ent.name[pq].tolist(), p_ent.name[pp].tolist()
    qa, pa = q_ent.addr[pq].tolist(), p_ent.addr[pp].tolist()

    def dist(a, b, scorer):
        return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)

    qn_len = np.fromiter(map(len, qn), np.int32, len(qn))
    pn_len = np.fromiter(map(len, pn), np.int32, len(pn))
    qa_len = np.fromiter(map(len, qa), np.int32, len(qa))
    pa_len = np.fromiter(map(len, pa), np.int32, len(pa))

    X = np.column_stack([
        s,
        rank[valid],
        top1[valid] - s,
        top1[valid],
        np.nan_to_num(s - nxt[valid], nan=0.0),
        dist(qn, pn, JaroWinkler.normalized_similarity),
        dist(qn, pn, fuzz.ratio),
        dist(qn, pn, fuzz.token_set_ratio),
        dist(qn, pn, fuzz.token_sort_ratio),
        dist(qn, pn, fuzz.partial_ratio),
        dist(qa, pa, fuzz.ratio),
        dist(qa, pa, fuzz.token_set_ratio),
        dist(qa, pa, fuzz.token_sort_ratio),
        _match(q_ent.pin[pq], p_ent.pin[pp]),
        _match(q_ent.num[pq], p_ent.num[pp]),
        np.array([i.startswith("S2-") for i in p_ent.ids[pp]], dtype=np.float32),
        _len_ratio(qn_len, pn_len),
        _len_ratio(qa_len, pa_len),
        ((qn_len == 0) | (pn_len == 0)).astype(np.float32),
    ]).astype(np.float32)
    return X, pq, pp


def build_features(q_ent, p_ent, q_rows, idx, sim, chunk_queries: int = 100_000):
    parts, pqs, pps = [], [], []
    for s in tqdm(range(0, len(q_rows), chunk_queries), desc="Pair features", mininterval=5):
        X, pq, pp = pair_features(q_ent, p_ent, q_rows[s:s + chunk_queries], idx, sim)
        parts.append(X); pqs.append(pq); pps.append(pp)
    return np.concatenate(parts), np.concatenate(pqs), np.concatenate(pps)


# ---------------------------------------------------------------------------
# Reranker training (train split) and threshold tuning
# ---------------------------------------------------------------------------

def macro_f_beta_sweep(pq, probs, labels, true_counts, n_queries, thresholds, beta=0.5):
    """Vectorised equivalent of reranker.compute_macro_f_beta for many thresholds.

    ``true_counts`` holds the full ground-truth match count per query, so true
    matches missed by blocking still count as false negatives.
    """
    b2 = beta ** 2
    scores = []
    for t in thresholds:
        pred = probs >= t
        n_pred = np.bincount(pq[pred], minlength=n_queries)
        tp = np.bincount(pq[pred & labels], minlength=n_queries)
        precision = np.divide(tp, n_pred, out=np.zeros(n_queries), where=n_pred > 0)
        recall = np.divide(tp, true_counts, out=np.zeros(n_queries), where=true_counts > 0)
        denom = b2 * precision + recall
        f = np.divide((1 + b2) * precision * recall, denom, out=np.zeros(n_queries), where=denom > 0)
        f[(n_pred == 0) & (true_counts == 0)] = 1.0
        scores.append(f.mean())
    return np.asarray(scores)


def load_ground_truth(path: str, s1_ids: np.ndarray, pool_ids: np.ndarray):
    """Returns (positive pair keys q_row * n_pool + pool_row, full true-match count per query row)."""
    gt = pd.read_csv(path, sep="\t", dtype=str)
    gt = gt[gt["source1_entity_id"].isin(pd.Index(s1_ids))]
    q_index = pd.Index(s1_ids).get_indexer(gt["source1_entity_id"])
    matches = gt["matched_entity_ids"].fillna("").str.split(",")
    lens = matches.str.len().to_numpy()
    flat = pd.Series(np.concatenate(matches.to_numpy()) if len(matches) else [], dtype=object).str.strip()
    q_flat = np.repeat(q_index, lens)
    keep = (flat != "").to_numpy()
    flat, q_flat = flat[keep], q_flat[keep]
    true_counts = np.bincount(q_flat, minlength=len(s1_ids))
    pool_index = pd.Index(pool_ids)
    if not pool_index.is_unique:
        pool_index = pool_index.drop_duplicates()  # positions below are re-mapped to first occurrences
        first_pos = pd.Index(pool_ids).get_indexer_for(pool_index)
        p_flat = pool_index.get_indexer(flat)
        p_flat = np.where(p_flat >= 0, first_pos[np.maximum(p_flat, 0)], -1)
    else:
        p_flat = pool_index.get_indexer(flat)
    found = p_flat >= 0
    keys = q_flat[found].astype(np.int64) * len(pool_ids) + p_flat[found]
    return np.unique(keys), true_counts


def train_reranker(
    models,
    train_dir: str,
    n_train_queries: int = 300_000,
    holdout_frac: float = 0.2,
    top_k: int = blocking_config.top_k_candidates,
    clean_processes: Optional[int] = None,
    seed: int = 42,
):
    """Blocks a sample of train S1 against the full train S2+S3 pool, labels pairs from
    ground truth, fits CatBoost (GPU if available) and tunes the F0.5 threshold."""
    from catboost import CatBoostClassifier

    with Timer("Train: load + clean entities"):
        pool = make_clean_pool(clean_processes)
        try:
            q_ent = load_entities([os.path.join(train_dir, "train_source1.tsv")], pool, sample_rows=n_train_queries, seed=seed)
            p_ent = load_entities([os.path.join(train_dir, "train_source2.tsv"),
                                   os.path.join(train_dir, "train_source3.tsv")], pool)
        finally:
            if pool is not None:
                pool.close(); pool.join()
    log(f"Train queries={len(q_ent):,}, pool={len(p_ent):,}")

    with Timer("Train: encode pool"):
        p_emb = encode_entities(p_ent, models)
    with Timer("Train: encode queries"):
        q_emb = encode_entities(q_ent, models)
    with Timer("Train: GPU top-k search"):
        idx, sim = search_topk(q_ent, q_emb, p_ent, p_emb, top_k, models)
    del p_emb, q_emb
    gc.collect()

    with Timer("Train: features"):
        X, pq, pp = build_features(q_ent, p_ent, np.arange(len(q_ent)), idx, sim)
    pos_keys, true_counts = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"), q_ent.ids, p_ent.ids)
    labels = np.isin(pq.astype(np.int64) * len(p_ent) + pp, pos_keys)
    blocking_recall = labels.sum() / max(true_counts.sum(), 1)
    log(f"Train pairs={len(labels):,}, positives={int(labels.sum()):,}, blocking recall@{top_k}={blocking_recall:.4f}")
    del p_ent, idx, sim
    gc.collect()

    rng = np.random.default_rng(seed)
    is_holdout_q = rng.random(len(q_ent)) < holdout_frac
    hold = is_holdout_q[pq]

    def make_model(task_type: str) -> "CatBoostClassifier":
        gpu_params = {"devices": "0", "gpu_ram_part": 0.8} if task_type == "GPU" else {}
        return CatBoostClassifier(
            iterations=3000,
            learning_rate=0.08,
            depth=8,
            loss_function="Logloss",
            eval_metric="Logloss",
            task_type=task_type,
            early_stopping_rounds=100,
            random_seed=seed,
            verbose=250,
            allow_writing_files=False,
            thread_count=-1,
            **gpu_params,
        )

    with Timer("Train: CatBoost fit"):
        fit_args = dict(X=X[~hold], y=labels[~hold], eval_set=(X[hold], labels[hold]), use_best_model=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                model = make_model("GPU")
                model.fit(**fit_args)
            except Exception as exc:  # e.g. GPU OOM: don't lose the encoded train split
                log(f"CatBoost GPU fit failed ({exc}); retrying on CPU")
                model = make_model("CPU")
                model.fit(**fit_args)
        else:
            model = make_model("CPU")
            model.fit(**fit_args)

    probs = model.predict_proba(X[hold], thread_count=-1)[:, 1]
    hold_q = np.flatnonzero(is_holdout_q)
    remap = np.full(len(q_ent), -1, dtype=np.int64)
    remap[hold_q] = np.arange(len(hold_q))
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = macro_f_beta_sweep(remap[pq[hold]], probs, labels[hold], true_counts[hold_q], len(hold_q),
                                thresholds, beta=reranker_config.f_beta)
    best = int(np.argmax(scores))
    threshold = float(thresholds[best])
    log(f"Holdout Macro F0.5 = {scores[best]:.4f} at threshold {threshold:.2f} "
        f"(F0.5 @0.50 = {scores[np.argmin(np.abs(thresholds - 0.5))]:.4f})")
    importances = sorted(zip(FEATURE_NAMES, model.get_feature_importance()), key=lambda x: -x[1])
    log("Feature importance: " + ", ".join(f"{n}={v:.1f}" for n, v in importances))
    return model, threshold, {"holdout_f05": float(scores[best]), "blocking_recall": float(blocking_recall)}


# ---------------------------------------------------------------------------
# Test inference
# ---------------------------------------------------------------------------

def block_test(models, test_dir: str, output_dir: str, top_k: int = blocking_config.top_k_candidates,
               clean_processes: Optional[int] = None):
    """Encodes and blocks the full test set, writes candidate_pairs.tsv, returns state for reranking."""
    with Timer("Test: load + clean entities"):
        pool = make_clean_pool(clean_processes)
        try:
            q_ent = load_entities([os.path.join(test_dir, "test_source1.tsv")], pool)
            p_ent = load_entities([os.path.join(test_dir, "test_source2.tsv"),
                                   os.path.join(test_dir, "test_source3.tsv")], pool)
        finally:
            if pool is not None:
                pool.close(); pool.join()
    log(f"Test S1={len(q_ent):,}, pool={len(p_ent):,}")

    with Timer("Test: encode pool"):
        p_emb = encode_entities(p_ent, models)
    with Timer("Test: encode queries"):
        q_emb = encode_entities(q_ent, models)
    with Timer("Test: GPU top-k search"):
        idx, sim = search_topk(q_ent, q_emb, p_ent, p_emb, top_k, models)
    del p_emb, q_emb
    gc.collect()

    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    write_candidate_pairs(candidate_path, q_ent, p_ent, idx)
    return {"q_ent": q_ent, "p_ent": p_ent, "idx": idx, "sim": sim, "candidate_path": candidate_path}


def rerank_test(state: Dict, reranker, threshold: float, output_dir: str, chunk_queries: int = 100_000) -> str:
    """Scores all test candidate pairs in bounded chunks and writes matching_results.tsv."""
    q_ent, p_ent, idx, sim = state["q_ent"], state["p_ent"], state["idx"], state["sim"]
    keep = np.zeros(idx.shape, dtype=bool)
    all_rows = np.arange(len(q_ent))
    for s in tqdm(range(0, len(q_ent), chunk_queries), desc="Rerank test", mininterval=5):
        rows = all_rows[s:s + chunk_queries]
        X, _, _ = pair_features(q_ent, p_ent, rows, idx, sim)
        probs = reranker.predict_proba(X, thread_count=-1)[:, 1]
        valid = idx[rows] >= 0
        block = np.zeros(valid.shape, dtype=bool)
        block[valid] = probs >= threshold  # row-major order matches pair_features
        keep[rows] = block

    match_path = os.path.join(output_dir, "matching_results.tsv")
    n_matched = 0
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, row, k in zip(q_ent.ids, idx, keep):
            chosen = p_ent.ids[row[k]]
            n_matched += len(chosen) > 0
            f.write(f"{s1_id}\t{','.join(chosen)}\n")
    log(f"Wrote {match_path}: {n_matched:,}/{len(q_ent):,} S1 rows with >=1 match, "
        f"{int(keep.sum()):,} matched pairs")
    return match_path


def validate_outputs(matching_path: str, candidate_path: str, test_dir: str) -> bool:
    validator = os.path.join(path_config.base_dir, "utils", "validate_submission.py")
    result = subprocess.run(
        [sys.executable, validator, "--matching", matching_path, "--candidate", candidate_path, "--test-dir", test_dir],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0
