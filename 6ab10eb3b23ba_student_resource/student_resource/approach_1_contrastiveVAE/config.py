import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class PathConfig:
    # Local & SageMaker directories
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir: str = os.path.join(base_dir, "dataset")
    train_dir: str = os.path.join(dataset_dir, "train")
    test_dir: str = os.path.join(dataset_dir, "test")
    
    # Output directories
    output_dir: str = os.path.join(base_dir, "output")
    artifacts_dir: str = os.path.join(base_dir, "approach_1_contrastiveVAE", "artifacts")
    processed_dir: str = os.path.join(artifacts_dir, "processed")
    embeddings_dir: str = os.path.join(artifacts_dir, "embeddings")
    models_dir: str = os.path.join(artifacts_dir, "models")

    # AWS S3 Configuration
    s3_bucket: str = "amazon-ml-challenge-2026-entity-resolution"
    s3_prefix: str = "contrastive_vae_pipeline"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

@dataclass
class ModelConfig:
    # Backbone Transformer Model
    backbone_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_seq_length: int = 128
    latent_dim: int = 256
    
    # Training Parameters
    batch_size: int = 128
    learning_rate: float = 2e-5
    epochs: int = 5
    beta_kl: float = 0.01          # Weight for KL divergence loss
    lambda_contrastive: float = 1.0 # Weight for InfoNCE / Triplet loss
    temperature: float = 0.07       # Temperature for InfoNCE contrastive loss
    device: str = "cuda"

@dataclass
class BlockingConfig:
    countries: List[str] = field(default_factory=lambda: ["US", "India", "France"])
    top_k_candidates: int = 30
    faiss_index_type: str = "HNSW" # HNSW or Flat
    hnsw_m: int = 32

@dataclass
class RerankerConfig:
    # Feature Threshold Optimization for F_0.5 Macro Average
    f_beta: float = 0.5
    prob_threshold_min: float = 0.50
    prob_threshold_max: float = 0.95
    prob_threshold_step: float = 0.02
    default_threshold: float = 0.78
    
    # GBDT Parameters
    catboost_iterations: int = 500
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.05

path_config = PathConfig()
model_config = ModelConfig()
blocking_config = BlockingConfig()
reranker_config = RerankerConfig()
