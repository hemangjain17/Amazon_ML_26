import os
import pandas as pd
from typing import Dict, List, Tuple, Optional


def clean_str(val) -> str:
    """Cleans null, None, NaN values to empty string."""
    if pd.isna(val) or val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "null") else s


def get_entity_id(row: dict) -> str:
    """Gets entity ID from row dict checking various possible column names."""
    for col in ("entity_id", "id", "s1_id", "s2_id", "s3_id", "source1_entity_id"):
        if col in row:
            cleaned = clean_str(row[col])
            if cleaned:
                return cleaned
    return ""


def format_entity_text(df: pd.DataFrame) -> List[str]:
    """Formats entity record into clean, unified text representation."""
    formatted = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        name = clean_str(row_dict.get("business_name") or row_dict.get("name") or "")
        address = clean_str(row_dict.get("business_address") or row_dict.get("address") or "")
        city = clean_str(row_dict.get("city") or "")
        postal = clean_str(row_dict.get("postal_code") or "")
        country = clean_str(row_dict.get("country") or "")

        parts = []
        if name:
            parts.append(f"Name: {name}")
        if address or city or postal or country:
            addr_parts = [p for p in (address, city, postal, country) if p]
            parts.append(f"Address: {', '.join(addr_parts)}")

        text = " | ".join(parts) if parts else "Entity: Unknown"
        formatted.append(text)
    return formatted


def load_split_data(split_dir: str, is_train: bool = True, subset_size: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Loads source1, source2, source3, and ground truth dataframes for train/test split."""
    prefix = "train" if is_train else "test"

    s1_path = os.path.join(split_dir, f"{prefix}_source1.tsv")
    s2_path = os.path.join(split_dir, f"{prefix}_source2.tsv")
    s3_path = os.path.join(split_dir, f"{prefix}_source3.tsv")

    if not os.path.exists(s1_path):
        raise FileNotFoundError(f"Missing expected TSV file: {s1_path}")

    s1_nrows = subset_size if subset_size else None
    cand_nrows = (subset_size * 50) if subset_size else None

    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, nrows=s1_nrows)
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False, nrows=cand_nrows)
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False, nrows=cand_nrows)

    gt_df = None
    gt_path = os.path.join(split_dir, f"{prefix}_ground_truth.tsv")
    if is_train and os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        # Keep ground truth matching s1_df subset
        id_col = "entity_id" if "entity_id" in s1_df.columns else "id"
        valid_s1_ids = set(s1_df[id_col]) if id_col in s1_df.columns else set()
        gt_s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else "s1_id"
        if valid_s1_ids and gt_s1_col in gt_df.columns:
            gt_df = gt_df[gt_df[gt_s1_col].isin(valid_s1_ids)].copy()

    return {
        "s1": s1_df,
        "s2": s2_df,
        "s3": s3_df,
        "gt": gt_df,
    }


def build_ground_truth_map(gt_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Builds ground truth mapping: s1_id -> list of ground truth match IDs."""
    gt_map: Dict[str, List[str]] = {}
    if gt_df is None or gt_df.empty:
        return gt_map

    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else "s1_id"
    match_col = "matched_entity_ids" if "matched_entity_ids" in gt_df.columns else None

    for _, row in gt_df.iterrows():
        s1_id = clean_str(row.get(s1_col, ""))
        if not s1_id:
            continue

        if s1_id not in gt_map:
            gt_map[s1_id] = []

        if match_col and match_col in row:
            raw_matches = clean_str(row[match_col])
            if raw_matches:
                matches = [m.strip() for m in raw_matches.split(",") if m.strip()]
                gt_map[s1_id].extend(matches)
        else:
            # Fallback for old s2_id / s3_id separate columns
            s2_id = clean_str(row.get("s2_id", ""))
            s3_id = clean_str(row.get("s3_id", ""))
            if s2_id:
                gt_map[s1_id].append(s2_id)
            if s3_id:
                gt_map[s1_id].append(s3_id)

    return gt_map
