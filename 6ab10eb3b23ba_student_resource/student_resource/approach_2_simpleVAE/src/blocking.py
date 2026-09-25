import os
import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple

from ..config import path_config, model_config, blocking_config
from .dataset import EntityInferenceDataset
from .model import SimpleVAE


def extract_latent_embeddings(
    df: pd.DataFrame,
    model: SimpleVAE,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[List[str], List[str], np.ndarray]:
    dataset = EntityInferenceDataset(
        df=df,
        tokenizer_name=model_config.backbone_name,
        max_length=model_config.max_seq_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
    )

    model.eval()
    entity_ids = []
    countries = []
    embeddings_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Latent Embeddings"):
            ids = batch["entity_id"]
            cnts = batch["country"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            _, mu, _ = model.encode(input_ids, attention_mask)
            z_norm = torch.nn.functional.normalize(mu, p=2, dim=1)

            entity_ids.extend(ids)
            countries.extend(cnts)
            embeddings_list.append(z_norm.cpu().numpy())

    embeddings = np.vstack(embeddings_list).astype(np.float32)
    return entity_ids, countries, embeddings


def build_faiss_index_and_search(
    s1_ids: List[str],
    s1_countries: List[str],
    s1_embeddings: np.ndarray,
    cand_ids: List[str],
    cand_countries: List[str],
    cand_embeddings: np.ndarray,
    top_k: int = blocking_config.top_k_candidates,
) -> Dict[str, List[str]]:
    candidate_map = {s1_id: [] for s1_id in s1_ids}
    
    cand_country_indices = {}
    for idx, country in enumerate(cand_countries):
        cand_country_indices.setdefault(country, []).append(idx)

    for country in blocking_config.countries:
        s1_indices = [i for i, c in enumerate(s1_countries) if c == country]
        cand_indices = cand_country_indices.get(country, [])

        if not s1_indices or not cand_indices:
            print(f"Skipping country {country}: S1 count={len(s1_indices)}, Candidate count={len(cand_indices)}")
            continue

        print(f"Indexing {len(cand_indices)} candidates for country {country}...")
        sub_cand_embeddings = cand_embeddings[cand_indices]
        sub_cand_ids = [cand_ids[i] for i in cand_indices]

        sub_s1_embeddings = s1_embeddings[s1_indices]
        sub_s1_ids = [s1_ids[i] for i in s1_indices]

        dimension = sub_cand_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(sub_cand_embeddings)

        print(f"Querying FAISS index for {len(sub_s1_ids)} S1 queries in {country}...")
        distances, indices = index.search(sub_s1_embeddings, min(top_k, len(sub_cand_ids)))

        for i, s1_id in enumerate(sub_s1_ids):
            top_neighbors = []
            for neighbor_idx in indices[i]:
                if neighbor_idx != -1:
                    top_neighbors.append(sub_cand_ids[neighbor_idx])
            candidate_map[s1_id] = top_neighbors

        # Free memory per country loop
        del sub_cand_embeddings, sub_s1_embeddings, index, distances, indices
        import gc; gc.collect()

    return candidate_map


def run_blocking_pipeline(
    checkpoint_path: str,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    output_candidate_path: str = None,
) -> Tuple[str, Dict[str, List[str]]]:
    output_candidate_path = output_candidate_path or os.path.join(path_config.output_dir, "candidate_pairs.tsv")

    device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")
    print(f"Loading Simple VAE checkpoint from {checkpoint_path}...")
    model = SimpleVAE(
        backbone_name=model_config.backbone_name,
        latent_dim=model_config.latent_dim,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Extracting S1 Latent Embeddings...")
    s1_ids, s1_countries, s1_emb = extract_latent_embeddings(s1_df, model, device)

    print("Extracting S2 & S3 Latent Embeddings...")
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    cand_ids, cand_countries, cand_emb = extract_latent_embeddings(cand_df, model, device)

    np.save(os.path.join(path_config.embeddings_dir, "s1_embeddings.npy"), s1_emb)
    np.save(os.path.join(path_config.embeddings_dir, "cand_embeddings.npy"), cand_emb)

    print("Building FAISS Indices and Searching Top Candidates...")
    candidate_map = build_faiss_index_and_search(
        s1_ids=s1_ids,
        s1_countries=s1_countries,
        s1_embeddings=s1_emb,
        cand_ids=cand_ids,
        cand_countries=cand_countries,
        cand_embeddings=cand_emb,
        top_k=blocking_config.top_k_candidates,
    )

    print(f"Exporting candidate pairs to {output_candidate_path}...")
    with open(output_candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids:
            cand_list_str = ",".join(candidate_map.get(s1_id, []))
            f.write(f"{s1_id}\t{cand_list_str}\n")

    print(f"Successfully generated candidate_pairs.tsv ({len(s1_ids)} S1 rows)")
    return output_candidate_path, candidate_map
