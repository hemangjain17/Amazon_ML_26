import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SimpleVAE(nn.Module):
    """Standard Unsupervised Variational Autoencoder (Simple VAE) for Business Entity Resolution.

    Encodes text into a continuous latent space z in R^latent_dim using ELBO loss (Reconstruction + KL Divergence).
    """

    def __init__(
        self,
        backbone_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        latent_dim: int = 256,
    ):
        super(SimpleVAE, self).__init__()
        self.latent_dim = latent_dim

        # Transformer Encoder Backbone
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_dim = self.encoder.config.hidden_size

        # Variational Bottleneck Projection Heads
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder / Reconstruction Head
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs, attention_mask)
        
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return pooled, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic mean for inference

    def decode(self, z):
        return self.decoder(z)

    def forward(self, input_ids, attention_mask):
        h, mu, logvar = self.encode(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar)
        h_recon = self.decode(z)
        return h, mu, logvar, z, h_recon

    def compute_loss(
        self,
        input_ids,
        attention_mask,
        beta: float = 0.01,
    ):
        h, mu, logvar, z, h_recon = self.forward(input_ids, attention_mask)

        # 1. Reconstruction Loss (MSE in hidden feature space)
        loss_recon = F.mse_loss(h_recon, h)

        # 2. KL Divergence Loss
        loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # Total ELBO Loss
        total_loss = loss_recon + (beta * loss_kl)

        return {
            "total_loss": total_loss,
            "loss_recon": loss_recon.item(),
            "loss_kl": loss_kl.item(),
        }
