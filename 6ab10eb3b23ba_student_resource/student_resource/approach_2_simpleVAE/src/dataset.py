import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

try:
    from approach_2_simpleVAE.src.normalization import format_entity_string
except (ImportError, ValueError):
    try:
        from src.normalization import format_entity_string
    except (ImportError, ValueError):
        from .normalization import format_entity_string


class UnsupervisedEntityDataset(Dataset):
    """Dataset for unsupervised VAE training over all entity records across S1, S2, S3."""

    def __init__(
        self,
        dfs: list[pd.DataFrame],
        tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_length: int = 128,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        self.formatted_texts = []
        for df in dfs:
            for _, row in df.iterrows():
                formatted = format_entity_string(
                    name=row.get("business_name", ""),
                    address=row.get("business_address", ""),
                    country=row.get("country", ""),
                )
                if formatted.strip():
                    self.formatted_texts.append(formatted)

        print(f"Total unsupervised training entity strings: {len(self.formatted_texts):,}")

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
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
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
        self.countries = df["country"].tolist() if "country" in df.columns else ["US"] * len(df)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        self.formatted_texts = [
            format_entity_string(row["business_name"], row["business_address"], row.get("country", ""))
            for _, row in df.iterrows()
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
