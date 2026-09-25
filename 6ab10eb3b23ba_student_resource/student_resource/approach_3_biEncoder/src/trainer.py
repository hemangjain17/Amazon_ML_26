import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses

try:
    from approach_3_biEncoder.config import path_config, model_config
    from approach_3_biEncoder.src.dataset import BiEncoderPairDataset
    from approach_3_biEncoder.src.model import build_bi_encoder_model, build_mnr_loss_function
    from approach_3_biEncoder.src.s3_utils import upload_file_to_s3
except (ImportError, ValueError):
    try:
        from config import path_config, model_config
        from src.dataset import BiEncoderPairDataset
        from src.model import build_bi_encoder_model, build_mnr_loss_function
        from src.s3_utils import upload_file_to_s3
    except (ImportError, ValueError):
        from ..config import path_config, model_config
        from .dataset import BiEncoderPairDataset
        from .model import build_bi_encoder_model, build_mnr_loss_function
        from .s3_utils import upload_file_to_s3


def train_bi_encoder(
    s1_path: str = None,
    s2_path: str = None,
    s3_path: str = None,
    gt_path: str = None,
    epochs: int = model_config.epochs,
    batch_size: int = model_config.batch_size,
    learning_rate: float = model_config.learning_rate,
    save_s3: bool = True,
):
    """Trains Two-Tower Bi-Encoder model using SentenceTransformers MultipleNegativesRankingLoss."""
    s1_path = s1_path or os.path.join(path_config.train_dir, "train_source1.tsv")
    s2_path = s2_path or os.path.join(path_config.train_dir, "train_source2.tsv")
    s3_path = s3_path or os.path.join(path_config.train_dir, "train_source3.tsv")
    gt_path = gt_path or os.path.join(path_config.train_dir, "train_ground_truth.tsv")

    print(f"Loading training datasets for Bi-Encoder fine-tuning...")
    s1_df = pd.read_csv(s1_path, sep="\t")
    s2_df = pd.read_csv(s2_path, sep="\t")
    s3_df = pd.read_csv(s3_path, sep="\t")
    gt_df = pd.read_csv(gt_path, sep="\t")

    pair_builder = BiEncoderPairDataset(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        gt_df=gt_df,
    )
    train_examples = pair_builder.get_examples()

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    model = build_bi_encoder_model(backbone_name=model_config.backbone_name)
    train_loss = build_mnr_loss_function(model)

    output_model_path = os.path.join(path_config.models_dir, "bi_encoder_model")

    print(f"Starting Bi-Encoder Training for {epochs} Epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=model_config.warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=output_model_path,
        show_progress_bar=True,
    )

    print(f"--> Saved fine-tuned Bi-Encoder model to {output_model_path}")

    if save_s3:
        s3_prefix = f"{path_config.s3_prefix}/models/bi_encoder_model"
        print(f"Uploading fine-tuned Bi-Encoder model to S3 bucket: {path_config.s3_bucket}...")
        for root, _, files in os.walk(output_model_path):
            for file in files:
                local_f = os.path.join(root, file)
                rel_f = os.path.relpath(local_f, output_model_path)
                upload_file_to_s3(local_f, path_config.s3_bucket, f"{s3_prefix}/{rel_f}")

    return output_model_path
