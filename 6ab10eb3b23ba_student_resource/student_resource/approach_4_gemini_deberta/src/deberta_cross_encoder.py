import os
import gc
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger("DebertaCrossEncoder")


class CrossEncoderPairDataset(Dataset):
    """Dataset for DeBERTa Cross-Encoder pair classification."""

    def __init__(self, pairs: List[Tuple[str, str]], labels: Optional[List[int]] = None):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1_text, cand_text = self.pairs[idx]
        item = {"s1_text": s1_text, "cand_text": cand_text}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def load_deberta_model_and_tokenizer(
    model_name: str = "microsoft/deberta-v3-large",
    use_qlora: bool = True,
    device: str = "cuda",
):
    """
    Loads DeBERTa-v3-large model and tokenizer with 4-bit QLoRA if available/requested,
    falling back cleanly to FP16/FP32 on CPU/GPU without bitsandbytes errors.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    logger.info(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.warning(f"Fast tokenizer failed ({e}). Loading with use_fast=False...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    is_cuda = torch.cuda.is_available() and device.startswith("cuda")
    target_device = "cuda" if is_cuda else "cpu"

    model = None
    loaded_mode = "FP32"

    if is_cuda and use_qlora:
        try:
            from transformers import BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

            logger.info("Attempting 4-bit QLoRA loading via bitsandbytes...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                quantization_config=bnb_config,
                device_map="auto",
            )
            model = prepare_model_for_kbit_training(model)
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query_proj", "key_proj", "value_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_CLS",
            )
            model = get_peft_model(model, lora_config)
            loaded_mode = "4-bit QLoRA"
            logger.info("Successfully loaded 4-bit QLoRA DeBERTa model.")
        except Exception as e:
            logger.warning(f"4-bit QLoRA loading unavailable ({e}). Falling back to FP16/FP32 LoRA...")

    if model is None:
        logger.info(f"Loading standard DeBERTa model on {target_device}...")
        dtype = torch.float16 if is_cuda else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            torch_dtype=dtype if is_cuda else torch.float32,
        ).to(target_device)
        loaded_mode = "FP16" if is_cuda else "FP32 CPU"

    logger.info(f"DeBERTa model initialized in {loaded_mode} mode.")
    return model, tokenizer, loaded_mode


def train_cross_encoder_with_oom_safeguard(
    model,
    tokenizer,
    train_pairs: List[Tuple[str, str]],
    train_labels: List[int],
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    lr: float = 2e-5,
    max_seq_length: int = 160,
    device: str = "cuda",
    telegram_notifier=None,
) -> Any:
    """
    Trains DeBERTa Cross-Encoder with gradient accumulation, mixed precision,
    and automatic batch-size reduction on CUDA OOM.
    """
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    is_cuda = torch.cuda.is_available() and device.startswith("cuda")
    target_device = "cuda" if is_cuda else "cpu"

    logger.info(f"Starting Cross-Encoder training on {len(train_pairs)} pairs ({epochs} epochs)...")

    dataset = CrossEncoderPairDataset(train_pairs, train_labels)

    current_batch_size = batch_size
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=is_cuda)

    total_steps = (len(dataset) // (current_batch_size * grad_accum_steps)) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=max(1, total_steps),
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        step_count = 0

        dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True)

        for step, batch in enumerate(dataloader):
            try:
                s1_texts = batch["s1_text"]
                cand_texts = batch["cand_text"]
                labels = batch["label"].to(target_device)

                inputs = tokenizer(
                    s1_texts,
                    cand_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                ).to(target_device)

                with torch.cuda.amp.autocast(enabled=is_cuda, dtype=torch.float16 if is_cuda else torch.float32):
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss / grad_accum_steps

                scaler.scale(loss).backward()
                total_loss += loss.item() * grad_accum_steps
                step_count += 1

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

            except torch.cuda.OutOfMemoryError as oom:
                logger.warning(f"⚠️ CUDA OOM encountered during training step {step}. Emergency memory clearing...")
                gc.collect()
                if is_cuda:
                    torch.cuda.empty_cache()

                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(f"Halved batch size to {current_batch_size} and continuing...")
                    if telegram_notifier:
                        telegram_notifier.send_message(f"⚠️ CUDA OOM caught in training. Reduced batch size to {current_batch_size}.")
                else:
                    logger.error("CUDA OOM even at batch_size=1. Aborting training batch.")
                    break

        avg_loss = total_loss / max(1, step_count)
        log_msg = f"Epoch {epoch+1}/{epochs} Complete. Average Loss: {avg_loss:.4f}"
        logger.info(log_msg)
        if telegram_notifier:
            telegram_notifier.update_status(log_msg)

    return model


@torch.inference_mode()
def predict_cross_encoder_probabilities(
    model,
    tokenizer,
    pairs: List[Tuple[str, str]],
    batch_size: int = 16,
    max_seq_length: int = 160,
    device: str = "cuda",
) -> List[float]:
    """
    Runs batch inference on pair list with FP16, dynamic length sorting,
    and CUDA OOM catch-and-retry logic.
    """
    if not pairs:
        return []

    is_cuda = torch.cuda.is_available() and device.startswith("cuda")
    target_device = "cuda" if is_cuda else "cpu"

    model.eval()

    # Dynamic length bucketing: sort pairs by combined char length to minimize batch padding
    indexed_pairs = list(enumerate(pairs))
    indexed_pairs.sort(key=lambda x: len(x[1][0]) + len(x[1][1]))

    sorted_indices = [idx for idx, _ in indexed_pairs]
    sorted_pairs = [pair for _, pair in indexed_pairs]

    probabilities_sorted = []
    current_batch_size = batch_size

    i = 0
    while i < len(sorted_pairs):
        chunk = sorted_pairs[i : i + current_batch_size]
        s1_texts, cand_texts = zip(*chunk)

        try:
            inputs = tokenizer(
                list(s1_texts),
                list(cand_texts),
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            ).to(target_device)

            with torch.cuda.amp.autocast(enabled=is_cuda, dtype=torch.float16 if is_cuda else torch.float32):
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1]

            probabilities_sorted.extend(probs.cpu().tolist())
            i += len(chunk)

        except torch.cuda.OutOfMemoryError:
            logger.warning("⚠️ CUDA OOM during inference batch. Clearing cache and halving batch size...")
            gc.collect()
            if is_cuda:
                torch.cuda.empty_cache()
            if current_batch_size > 1:
                current_batch_size = max(1, current_batch_size // 2)
            else:
                logger.error("CUDA OOM at batch_size=1 during inference. Assigning default low probability 0.0.")
                probabilities_sorted.append(0.0)
                i += 1

    # Re-order probabilities back to original pair order
    probabilities = [0.0] * len(pairs)
    for orig_idx, prob in zip(sorted_indices, probabilities_sorted):
        probabilities[orig_idx] = prob

    return probabilities
