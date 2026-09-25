import os
from dataclasses import dataclass, field
from typing import List, Tuple


def resolve_paths(base_dir: str, approach_name: str) -> Tuple[str, str, str, str, str]:
    """Resolves local, SageMaker, or Kaggle input/working paths dynamically."""
    if os.path.exists("/kaggle/input"):
        output_dir = "/kaggle/working/output"
        artifacts_dir = f"/kaggle/working/{approach_name}/artifacts"

        train_dir, test_dir, dataset_dir = None, None, None
        for root, _, files in os.walk("/kaggle/input"):
            if "train_source1.tsv" in files:
                train_dir = root
                parent = os.path.dirname(root)
                test_candidate = os.path.join(parent, "test")
                if os.path.exists(test_candidate) and os.path.exists(os.path.join(test_candidate, "test_source1.tsv")):
                    test_dir = test_candidate
                dataset_dir = parent
                break

        if not test_dir:
            for root, _, files in os.walk("/kaggle/input"):
                if "test_source1.tsv" in files:
                    test_dir = root
                    break

        if train_dir and test_dir:
            return dataset_dir or os.path.dirname(train_dir), train_dir, test_dir, output_dir, artifacts_dir

    dataset_dir = os.path.join(base_dir, "dataset")
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    output_dir = os.path.join(base_dir, "output")
    artifacts_dir = os.path.join(base_dir, approach_name, "artifacts")

    return dataset_dir, train_dir, test_dir, output_dir, artifacts_dir


@dataclass
class PathConfig:
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir: str = ""
    train_dir: str = ""
    test_dir: str = ""
    output_dir: str = ""
    artifacts_dir: str = ""
    processed_dir: str = ""
    embeddings_dir: str = ""
    models_dir: str = ""

    s3_bucket: str = "amazon-ml-challenge-2026-entity-resolution"
    s3_prefix: str = "simple_vae_pipeline"

    def __post_init__(self):
        d_dir, tr_dir, ts_dir, out_dir, art_dir = resolve_paths(self.base_dir, "approach_2_simpleVAE")
        self.dataset_dir = d_dir
        self.train_dir = tr_dir
        self.test_dir = ts_dir
        self.output_dir = out_dir
        self.artifacts_dir = art_dir
        self.processed_dir = os.path.join(self.artifacts_dir, "processed")
        self.embeddings_dir = os.path.join(self.artifacts_dir, "embeddings")
        self.models_dir = os.path.join(self.artifacts_dir, "models")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

@dataclass
class ModelConfig:
    backbone_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_seq_length: int = 128
    latent_dim: int = 256
    
    batch_size: int = 128
    learning_rate: float = 2e-5
    epochs: int = 5
    beta_kl: float = 0.01          # Weight for KL divergence loss
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
