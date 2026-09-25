import pandas as pd
from typing import Dict, List
from torch.utils.data import Dataset
from sentence_transformers import InputExample
from transformers import AutoTokenizer

try:
    from approach_3_biEncoder.src.normalization import format_entity_string
except (ImportError, ValueError):
    try:
        from src.normalization import format_entity_string
    except (ImportError, ValueError):
        from .normalization import format_entity_string


class BiEncoderPairDataset:
    """Constructs InputExamples for SentenceTransformer training using MultipleNegativesRankingLoss."""

    def __init__(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        gt_df: pd.DataFrame,
        max_pairs: int = 250000, # Subsample capping
    ):
        print("Building lookup dictionaries for Bi-Encoder dataset...")
        self.s1_records = self._dataframe_to_dict(s1_df)
        self.s2_records = self._dataframe_to_dict(s2_df)
        self.s3_records = self._dataframe_to_dict(s3_df)

        self.examples = []
        for _, row in gt_df.iterrows():
            s1_id = row["source1_entity_id"]
            matched_str = str(row["matched_entity_ids"]) if pd.notnull(row["matched_entity_ids"]) else ""
            if not matched_str.strip() or s1_id not in self.s1_records:
                continue

            s1_text = self._get_entity_text(s1_id, self.s1_records)
            matches = [m.strip() for m in matched_str.split(",") if m.strip()]

            for pos_id in matches:
                pos_text = ""
                if pos_id.startswith("S2-") and pos_id in self.s2_records:
                    pos_text = self._get_entity_text(pos_id, self.s2_records)
                elif pos_id.startswith("S3-") and pos_id in self.s3_records:
                    pos_text = self._get_entity_text(pos_id, self.s3_records)

                if pos_text:
                    self.examples.append(InputExample(texts=[s1_text, pos_text], label=1.0))

        print(f"Total positive Bi-Encoder training pairs constructed: {len(self.examples):,}")
        
        # Subsample positive training pairs to keep training around ~10 minutes on Kaggle T4 GPU
        if max_pairs and len(self.examples) > max_pairs:
            print(f"--> Capping training pairs to a diverse sample of {max_pairs:,}...")
            import random
            random.seed(42)
            self.examples = random.sample(self.examples, max_pairs)

    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, dict]:
        # Using zipped lists is 50-100x faster than iterrows()
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

    def get_examples(self) -> List[InputExample]:
        return self.examples


class EntityInferenceDataset(Dataset):
    """Dataset for batch encoding entity records using Bi-Encoder sentence transformer."""

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
