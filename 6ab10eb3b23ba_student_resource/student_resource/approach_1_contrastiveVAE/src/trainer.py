import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm

try:
    from approach_1_contrastiveVAE.config import path_config, model_config
    from approach_1_contrastiveVAE.src.dataset import ContrastiveEntityTripletDataset
    from approach_1_contrastiveVAE.src.model import ContrastiveVAE
    from approach_1_contrastiveVAE.src.s3_utils import upload_file_to_s3
except (ImportError, ValueError):
    try:
        from config import path_config, model_config
        from src.dataset import ContrastiveEntityTripletDataset
        from src.model import ContrastiveVAE
        from src.s3_utils import upload_file_to_s3
    except (ImportError, ValueError):
        from ..config import path_config, model_config
        from .dataset import ContrastiveEntityTripletDataset
        from .model import ContrastiveVAE
        from .s3_utils import upload_file_to_s3


def train_contrastive_vae(
    s1_path: str = None,
    s2_path: str = None,
    s3_path: str = None,
    gt_path: str = None,
    epochs: int = model_config.epochs,
    batch_size: int = model_config.batch_size,
    learning_rate: float = model_config.learning_rate,
    save_s3: bool = True,
):
    """Trains Contrastive VAE model on Ground Truth triplet pairs."""
    s1_path = s1_path or os.path.join(path_config.train_dir, "train_source1.tsv")
    s2_path = s2_path or os.path.join(path_config.train_dir, "train_source2.tsv")
    s3_path = s3_path or os.path.join(path_config.train_dir, "train_source3.tsv")
    gt_path = gt_path or os.path.join(path_config.train_dir, "train_ground_truth.tsv")

    print(f"Loading training source datasets...")
    s1_df = pd.read_csv(s1_path, sep="\t")
    s2_df = pd.read_csv(s2_path, sep="\t")
    s3_df = pd.read_csv(s3_path, sep="\t")
    gt_df = pd.read_csv(gt_path, sep="\t")

    print(f"Creating Contrastive Entity Triplet Dataset...")
    dataset = ContrastiveEntityTripletDataset(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        gt_df=gt_df,
        tokenizer_name=model_config.backbone_name,
        max_length=model_config.max_seq_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")
    print(f"Initializing Contrastive VAE model on device: {device}")
    model = ContrastiveVAE(
        backbone_name=model_config.backbone_name,
        latent_dim=model_config.latent_dim,
        temperature=model_config.temperature,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    # Enable Mixed Precision gradient scaler for FP16 training to save 2x GPU VRAM
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"Starting Contrastive VAE Training for {epochs} Epochs ({total_steps} steps)...")
    best_loss = float("inf")
    checkpoint_path = os.path.join(path_config.models_dir, "best_contrastive_vae.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_total_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        running_contrastive_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            anc_ids = batch["anchor_input_ids"].to(device)
            anc_mask = batch["anchor_attention_mask"].to(device)
            pos_ids = batch["pos_input_ids"].to(device)
            pos_mask = batch["pos_attention_mask"].to(device)
            neg_ids = batch["neg_input_ids"].to(device)
            neg_mask = batch["neg_attention_mask"].to(device)

            optimizer.zero_grad()
            
            # Autocast FP16 forward pass
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                losses = model.compute_loss(
                    anc_ids=anc_ids,
                    anc_mask=anc_mask,
                    pos_ids=pos_ids,
                    pos_mask=pos_mask,
                    neg_ids=neg_ids,
                    neg_mask=neg_mask,
                    beta=model_config.beta_kl,
                    lambda_contrastive=model_config.lambda_contrastive,
                )
                loss = losses["total_loss"]

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_total_loss += loss.item()
            running_recon_loss += losses["loss_recon"]
            running_kl_loss += losses["loss_kl"]
            running_contrastive_loss += losses["loss_contrastive"]

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{losses['loss_recon']:.4f}",
                "kl": f"{losses['loss_kl']:.4f}",
                "contrast": f"{losses['loss_contrastive']:.4f}",
            })

        avg_epoch_loss = running_total_loss / len(dataloader)
        print(f"Epoch {epoch} Complete. Average Loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": model_config,
                },
                checkpoint_path,
            )
            print(f"--> Saved new best checkpoint to {checkpoint_path}")

            if save_s3:
                s3_key = f"{path_config.s3_prefix}/models/best_contrastive_vae.pt"
                upload_file_to_s3(checkpoint_path, path_config.s3_bucket, s3_key)

    print(f"Training finished successfully! Best Loss: {best_loss:.4f}")
    return checkpoint_path
