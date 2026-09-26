from dataclasses import dataclass
import os


@dataclass
class Config:
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir: str = "/kaggle/working/output" if os.path.exists("/kaggle/working") else "output"
    token_cap: int = 300
    house_cap: int = 1200
    max_comparisons: int = 300_000_000
    validation_fraction: float = 0.20
    random_seed: int = 42
    max_pairs_for_u: int = 2_000_000
    match_recall_prior: float = 0.70
    min_threshold: float = 0.05
    max_threshold: float = 0.99
    threshold_step: float = 0.01


config = Config()
