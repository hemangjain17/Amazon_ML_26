import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

try:
    from approach_1_contrastiveVAE.src.normalization import format_entity_string
except (ImportError, ValueError):
    try:
        from src.normalization import format_entity_string
    except (ImportError, ValueError):
        from .normalization import format_entity_string


class ContrastiveEntityTripletDataset(Dataset):
    """Dataset for training Contrastive VAE using Anchor (S1), Positive (Matched S2/S3),

    and Hard Negative (Non-matched S2/S3 in same country).
    """
    def __init__(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        gt_df: pd.DataFrame,
        tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_length: int = 128,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Build index maps for fast lookup
        print("Building fast lookup index maps for Contrastive Triplet Dataset...")
        self.s1_records = self._dataframe_to_dict(s1_df)
        self.s2_records = self._dataframe_to_dict(s2_df)
        self.s3_records = self._dataframe_to_dict(s3_df)

        # Country partitioning for hard negative mining
        self.country_entities = {"US": [], "India": [], "France": []}
        for s2_id, rec in self.s2_records.items():
            c = rec.get("country", "US")
            if c in self.country_entities:
                self.country_entities[c].append(s2_id)
        for s3_id, rec in self.s3_records.items():
            c = rec.get("country", "US")
            if c in self.country_entities:
                self.country_entities[c].append(s3_id)

        # Build positive training pairs from Ground Truth
        self.triplets = []
        for _, row in gt_df.iterrows():
            s1_id = row["source1_entity_id"]
            matched_str = str(row["matched_entity_ids"]) if pd.notnull(row["matched_entity_ids"]) else ""
            if not matched_str.strip() or s1_id not in self.s1_records:
                continue

            matches = [m.strip() for m in matched_str.split(",") if m.strip()]
            for pos_id in matches:
                if pos_id.startswith("S2-") and pos_id in self.s2_records:
                    self.triplets.append((s1_id, pos_id))
                elif pos_id.startswith("S3-") and pos_id in self.s3_records:
                    self.triplets.append((s1_id, pos_id))

        print(f"Total positive training pairs constructed: {len(self.triplets)}")

    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, dict]:
        # Using lists zipped together is 50-100x faster than df.iterrows()
        ids = df["entity_id"].tolist()
        names = df["business_name"].fillna("").astype(str).tolist()
        addrs = df["business_address"].fillna("").astype(str).tolist()
        countries = df["country"].fillna("").astype(str).tolist()
        
        return {
            entity_id: {"name": name, "address": addr, "country": country}
            for entity_id, name, addr, country in zip(ids, names, addrs, countries)
        }

    def _get_entity_text(self, entity_id: str, record_dict: dict) -> str:
        rec = record_dict.get(entity_id, {})
        return format_entity_string(
            name=rec.get("name", ""),
            address=rec.get("address", ""),
            country=rec.get("country", ""),
        )

    def _get_hard_negative_text(self, country: str, pos_id: str) -> str:
        pool = self.country_entities.get(country, self.country_entities["US"])
        if not pool:
            return "unknown business | unknown address"
        
        neg_id = random.choice(pool)
        while neg_id == pos_id and len(pool) > 1:
            neg_id = random.choice(pool)

        if neg_id.startswith("S2-"):
            return self._get_entity_text(neg_id, self.s2_records)
        else:
            return self._get_entity_text(neg_id, self.s3_records)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx: int):
        s1_id, pos_id = self.triplets[idx]
        
        # Anchor Text (S1)
        anchor_text = self._get_entity_text(s1_id, self.s1_records)
        
        # Positive Text (S2 or S3)
        if pos_id.startswith("S2-"):
            pos_text = self._get_entity_text(pos_id, self.s2_records)
            country = self.s2_records[pos_id].get("country", "US")
        else:
            pos_text = self._get_entity_text(pos_id, self.s3_records)
            country = self.s3_records[pos_id].get("country", "US")

        # Hard Negative Text
        neg_text = self._get_hard_negative_text(country, pos_id)

        # Tokenize Anchor, Positive, Negative
        anc_enc = self.tokenizer(
            anchor_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        pos_enc = self.tokenizer(
            pos_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        neg_enc = self.tokenizer(
            neg_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "anchor_input_ids": anc_enc["input_ids"].squeeze(0),
            "anchor_attention_mask": anc_enc["attention_mask"].squeeze(0),
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
        }


class EntityInferenceDataset(Dataset):
    """Dataset for batch encoding entity records into latent space vectors."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_length: int = 128,
    ):
        self.entity_ids = df["entity_id"].tolist()
        self.countries = df["country"].fillna("US").astype(str).tolist() if "country" in df.columns else ["US"] * len(df)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        names = df["business_name"].fillna("").astype(str).tolist()
        addrs = df["business_address"].fillna("").astype(str).tolist()
        countries = self.countries

        self.formatted_texts = [
            format_entity_string(name, addr, country)
            for name, addr, country in zip(names, addrs, countries)
        ]

    def __len__(self):
        return len(self.formatted_texts)

    def __getitem__(self, idx: int):
        text = self.formatted_texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "entity_id": self.entity_ids[idx],
            "country": self.countries[idx],
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
