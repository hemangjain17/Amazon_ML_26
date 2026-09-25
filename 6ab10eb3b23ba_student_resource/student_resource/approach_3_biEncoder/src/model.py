import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, losses


def build_bi_encoder_model(
    backbone_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> SentenceTransformer:
    """Builds two-tower Bi-Encoder model using SentenceTransformer."""
    print(f"Loading pre-trained Bi-Encoder backbone: {backbone_name}")
    model = SentenceTransformer(backbone_name)
    return model


def build_mnr_loss_function(model: SentenceTransformer) -> losses.MultipleNegativesRankingLoss:
    """Constructs MultipleNegativesRankingLoss for contrastive positive pair training."""
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    return loss_fn
