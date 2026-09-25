import os
import zipfile
from dataclasses import dataclass, field
from typing import List, Tuple


def ensure_dataset_extracted(local_dir: str) -> bool:
    """Extracts dataset.zip automatically if TSV files are missing from local_dir."""
    if not local_dir:
        return False
    known_files = [
        "train/train_source1.tsv",
        "train/train_source2.tsv",
        "train/train_source3.tsv",
        "train/train_ground_truth.tsv",
        "test/test_source1.tsv",
        "test/test_source2.tsv",
        "test/test_source3.tsv",
    ]
    all_exist = all(
        os.path.exists(os.path.join(local_dir, f)) and os.path.getsize(os.path.join(local_dir, f)) > 100
        for f in known_files
    )
    if all_exist:
        return True

    search_dirs = [
        local_dir,
        os.path.dirname(local_dir),
        os.path.dirname(os.path.dirname(local_dir)),
    ]
    zip_path = None
    for d in search_dirs:
        candidate = os.path.join(d, "dataset.zip")
        if os.path.exists(candidate) and os.path.getsize(candidate) > 100:
            zip_path = candidate
            break

    if not zip_path:
        return False

    target_extract_dir = os.path.dirname(local_dir)
    print(f"📦 Extracting dataset archive from {zip_path} -> {target_extract_dir}...")
    os.makedirs(target_extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_extract_dir)
    print("✅ Dataset successfully extracted!")
    return True


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
    # Local, SageMaker & Kaggle directories
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir: str = ""
    train_dir: str = ""
    test_dir: str = ""
    
    # Output directories
    output_dir: str = ""
    artifacts_dir: str = ""
    processed_dir: str = ""
    embeddings_dir: str = ""
    models_dir: str = ""

    # AWS S3 Configuration
    s3_bucket: str = "amazon-ml-challenge-2026-entity-resolution"
    s3_prefix: str = "contrastive_vae_pipeline"

    def __post_init__(self):
        d_dir, tr_dir, ts_dir, out_dir, art_dir = resolve_paths(self.base_dir, "approach_1_contrastiveVAE")
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

        ensure_dataset_extracted(self.dataset_dir)

@dataclass
class ModelConfig:
    # Backbone Transformer Model
    backbone_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_seq_length: int = 128
    latent_dim: int = 256
    
    # Training Parameters
    batch_size: int = 32
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
