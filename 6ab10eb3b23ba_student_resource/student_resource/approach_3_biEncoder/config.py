import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class PathConfig:
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir: str = os.path.join(base_dir, "dataset")
    train_dir: str = os.path.join(dataset_dir, "train")
    test_dir: str = os.path.join(dataset_dir, "test")
    
    output_dir: str = os.path.join(base_dir, "output")
    artifacts_dir: str = os.path.join(base_dir, "approach_3_biEncoder", "artifacts")
    processed_dir: str = os.path.join(artifacts_dir, "processed")
    embeddings_dir: str = os.path.join(artifacts_dir, "embeddings")
    models_dir: str = os.path.join(artifacts_dir, "models")

    s3_bucket: str = "amazon-ml-challenge-2026-entity-resolution"
    s3_prefix: str = "bi_encoder_pipeline"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

@dataclass
class ModelConfig:
    backbone_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_seq_length: int = 128
    
    batch_size: int = 128
    learning_rate: float = 2e-5
    epochs: int = 5
    warmup_steps: int = 500
    device: str = "cuda"

@dataclass
class BlockingConfig:
    countries: List[str] = field(default_factory=lambda: ["US", "India", "France"])
    top_k_candidates: int = 30
    faiss_index_type: str = "HNSW"

@dataclass
class RerankerConfig:
    f_beta: float = 0.5
    prob_threshold_min: float = 0.50
    prob_threshold_max: float = 0.95
    prob_threshold_step: float = 0.02
    default_threshold: float = 0.78
    
    catboost_iterations: int = 500
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.05

path_config = PathConfig()
model_config = ModelConfig()
blocking_config = BlockingConfig()
reranker_config = RerankerConfig()
