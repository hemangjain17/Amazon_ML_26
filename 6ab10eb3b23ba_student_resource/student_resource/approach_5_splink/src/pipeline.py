"""Core data preparation, Splink fitting, scoring, and output helpers."""
from __future__ import annotations

import collections
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
from splink.comparison_library import ExactMatch, JaroWinklerAtThresholds
from unidecode import unidecode


REQUIRED_COLUMNS = {"entity_id", "business_name", "business_address", "country"}
BLOCK_COLUMNS = ["country_norm", "nr1", "nr2", "ar1", "house_k"]
MODEL_COLUMNS = [
    "unique_id", "entity_id", "business_name", "business_address", "country",
    "country_norm", "name_core", "addr_norm", "addr_house", "cluster",
    "nr1", "nr2", "ar1", "house_k",
]


def load_table(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.fillna("")


def normalize(value: object) -> str:
    text = unidecode(str(value or "")).lower()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text).split())


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["country_norm"] = frame["country"].map(normalize)
    frame["name_core"] = frame["business_name"].map(normalize)
    frame["addr_norm"] = frame["business_address"].map(normalize)
    frame["addr_house"] = frame["addr_norm"].str.extract(r"(?:^| )([0-9]{1,6})(?: |$)", expand=False).fillna("")
    frame["addr_words"] = frame["addr_norm"]
    return frame


def _token_frequency(frames: Sequence[pd.DataFrame], column: str) -> collections.Counter:
    frequencies: collections.Counter = collections.Counter()
    for frame in frames:
        for text in frame[column]:
            frequencies.update(set(text.split()))
    return frequencies


def _rare_key(frame: pd.DataFrame, column: str, frequencies: collections.Counter, cap: int, rank: int) -> pd.Series:
    values = []
    for text in frame[column]:
        tokens = sorted(
            (token for token in set(text.split()) if frequencies[token] <= cap),
            key=lambda token: (frequencies[token], token),
        )
        values.append(tokens[rank] if len(tokens) > rank else None)
    return pd.Series(values, index=frame.index, dtype=object)


def add_blocking_keys(s1: pd.DataFrame, pool: pd.DataFrame, token_cap: int, house_cap: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s1, pool = s1.copy(), pool.copy()
    name_freq = _token_frequency([s1, pool], "name_core")
    address_freq = _token_frequency([s1, pool], "addr_words")
    for frame in (s1, pool):
        frame["nr1"] = _rare_key(frame, "name_core", name_freq, token_cap, 0)
        frame["nr2"] = _rare_key(frame, "name_core", name_freq, token_cap, 1)
        frame["ar1"] = _rare_key(frame, "addr_words", address_freq, token_cap, 0)
    house_frequency = pool["addr_house"].value_counts()
    allowed_houses = set(house_frequency[house_frequency <= house_cap].index) - {""}
    for frame in (s1, pool):
        frame["house_k"] = frame["addr_house"].where(frame["addr_house"].isin(allowed_houses), None)
    return s1, pool


def estimate_comparisons(s1: pd.DataFrame, pool: pd.DataFrame) -> int:
    total = 0
    for key in BLOCK_COLUMNS[1:]:
        left = s1.groupby(["country_norm", key], dropna=True).size()
        right = pool.groupby(["country_norm", key], dropna=True).size()
        joined = left.to_frame("left").join(right.to_frame("right"), how="inner")
        total += int((joined["left"] * joined["right"]).sum())
    return total


def make_settings():
    return SettingsCreator(
        link_type="link_only",
        blocking_rules_to_generate_predictions=[
            block_on("country_norm", "nr1"),
            block_on("country_norm", "nr2"),
            block_on("country_norm", "ar1"),
            block_on("country_norm", "house_k"),
        ],
        comparisons=[
            JaroWinklerAtThresholds("name_core", [0.9, 0.7]).configure(term_frequency_adjustments=True),
            JaroWinklerAtThresholds("addr_norm", [0.9, 0.7]),
            ExactMatch("addr_house").configure(term_frequency_adjustments=True),
        ],
        retain_intermediate_calculation_columns=False,
    )


def _linker(s1: pd.DataFrame, pool: pd.DataFrame, settings) -> Linker:
    return Linker(
        [s1[MODEL_COLUMNS], pool[MODEL_COLUMNS]],
        settings,
        db_api=DuckDBAPI(),
        input_table_aliases=["s1", "pool"],
    )


def fit_settings(s1: pd.DataFrame, pool: pd.DataFrame, settings, max_pairs_for_u: int, recall_prior: float) -> dict:
    linker = _linker(s1, pool, settings)
    linker.training.estimate_probability_two_random_records_match(
        [block_on("country_norm", "house_k")], recall=recall_prior
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=max_pairs_for_u)
    linker.training.estimate_m_from_label_column("cluster")
    return linker._settings_obj.as_dict()


def score(s1: pd.DataFrame, pool: pd.DataFrame, settings: dict) -> pd.DataFrame:
    predictions = _linker(s1, pool, settings).inference.predict(threshold_match_weight=-100).as_pandas_dataframe()
    if predictions.empty:
        return pd.DataFrame(columns=["s1_id", "pool_id", "match_probability"])
    left_is_s1 = predictions["source_dataset_l"].eq("s1")
    s1_id = predictions["unique_id_l"].where(left_is_s1, predictions["unique_id_r"])
    pool_id = predictions["unique_id_r"].where(left_is_s1, predictions["unique_id_l"])
    return pd.DataFrame({
        "s1_id": s1_id.to_numpy(),
        "pool_id": pool_id.to_numpy(),
        "match_probability": predictions["match_probability"].to_numpy(),
    })


def ground_truth(path: str) -> Dict[str, List[str]]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        row.source1_entity_id: [value for value in row.matched_entity_ids.split(",") if value]
        for row in frame.itertuples(index=False)
    }


def add_clusters(s1: pd.DataFrame, pool: pd.DataFrame, truth: Dict[str, List[str]]) -> pd.DataFrame:
    s1 = s1.copy()
    pool = pool.copy()
    s1["cluster"] = s1["entity_id"]
    pool["cluster"] = pool["entity_id"]
    owner: Dict[str, str] = {}
    for s1_id, pool_ids in truth.items():
        for pool_id in pool_ids:
            owner.setdefault(pool_id, s1_id)
    pool["cluster"] = pool["entity_id"].map(owner).fillna(pool["cluster"])
    return s1, pool


def assign_pool_unique(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    selected = scores[scores["match_probability"] >= threshold].sort_values(
        ["match_probability", "s1_id", "pool_id"], ascending=[False, True, True]
    )
    return selected.drop_duplicates("pool_id", keep="first")


def macro_f05(s1_ids: Iterable[str], scores: pd.DataFrame, truth: Dict[str, List[str]], threshold: float) -> float:
    selected = assign_pool_unique(scores, threshold)
    predicted = selected.groupby("s1_id")["pool_id"].apply(set).to_dict()
    values = []
    for s1_id in s1_ids:
        actual = set(truth.get(s1_id, []))
        found = set(predicted.get(s1_id, set()))
        if not actual and not found:
            values.append(1.0)
            continue
        if not actual or not found:
            values.append(0.0)
            continue
        tp = len(actual & found)
        precision = tp / len(found)
        recall = tp / len(actual)
        values.append(1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0)
    return float(np.mean(values)) if values else 0.0


def choose_threshold(s1_ids: Iterable[str], scores: pd.DataFrame, truth: Dict[str, List[str]], minimum: float, maximum: float, step: float) -> Tuple[float, float]:
    best_threshold, best_score = 0.85, -math.inf
    for threshold in np.arange(minimum, maximum + step / 2, step):
        score_value = macro_f05(s1_ids, scores, truth, float(threshold))
        if score_value > best_score:
            best_threshold, best_score = float(threshold), score_value
    return best_threshold, best_score


def write_outputs(s1_ids: Sequence[str], scores: pd.DataFrame, threshold: float, candidate_path: str, matching_path: str) -> None:
    candidate_map = scores.groupby("s1_id")["pool_id"].apply(list).to_dict() if not scores.empty else {}
    matches = assign_pool_unique(scores, threshold)
    match_map = matches.groupby("s1_id")["pool_id"].apply(list).to_dict() if not matches.empty else {}
    candidate_rows = [{"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(candidate_map.get(s1_id, []))} for s1_id in s1_ids]
    matching_rows = [{"source1_entity_id": s1_id, "matched_entity_ids": ",".join(match_map.get(s1_id, []))} for s1_id in s1_ids]
    pd.DataFrame(candidate_rows).to_csv(candidate_path, sep="\t", index=False)
    pd.DataFrame(matching_rows).to_csv(matching_path, sep="\t", index=False)
