import os
import zipfile
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def ensure_dataset_extracted(local_dir: str) -> bool:
    """Extracts dataset.zip automatically if TSV dataset files are missing from local_dir."""
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
        os.path.dirname(os.path.dirname(os.path.dirname(local_dir))),
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
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir: str = ""
    train_dir: str = ""
    test_dir: str = ""
    output_dir: str = ""
    artifacts_dir: str = ""
    processed_dir: str = ""
    embeddings_dir: str = ""
    models_dir: str = ""

    def __post_init__(self):
        d_dir, tr_dir, ts_dir, out_dir, art_dir = resolve_paths(self.base_dir, "approach_4_gemini_deberta")
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
class GeminiConfig:
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model_name: str = "models/text-embedding-004"
    batch_size: int = 100
    max_workers: int = 8
    top_k_retrieval: int = 50
    mock_mode: bool = False  # Used during offline/testing when API key is missing


@dataclass
class CascadeConfig:
    top_k_pruned: int = 15
    w_gemini: float = 0.60
    w_jaccard: float = 0.20
    w_levenshtein: float = 0.20


@dataclass
class DebertaConfig:
    model_name: str = "microsoft/deberta-v3-large"
    max_seq_length: int = 160
    train_batch_size: int = 4
    eval_batch_size: int = 16
    grad_accum_steps: int = 4
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_ratio: float = 0.1
    use_qlora: bool = True
    use_fp16: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["query_proj", "key_proj", "value_proj"])
    device: str = "cuda"


@dataclass
class ThresholdConfig:
    f_beta: float = 0.5
    default_threshold: float = 0.85
    prob_min: float = 0.50
    prob_max: float = 0.95
    prob_step: float = 0.01


@dataclass
class TelegramConfig:
    bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    heartbeat_interval_sec: int = 300  # Send heartbeat every 5 mins during long operations
    enabled: bool = field(default_factory=lambda: bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")))


# Singletons for convenience
path_config = PathConfig()
gemini_config = GeminiConfig()
cascade_config = CascadeConfig()
deberta_config = DebertaConfig()
threshold_config = ThresholdConfig()
telegram_config = TelegramConfig()
