import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ContrastiveVAE(nn.Module):
    """Contrastive Variational Autoencoder (Contrastive VAE) for Business Entity Resolution.

    Combines a pre-trained Transformer backbone with a Variational bottleneck (z in R^latent_dim)
    and a Supervised Contrastive / InfoNCE loss to structure the latent embedding space.
    """

    def __init__(
        self,
        backbone_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        latent_dim: int = 256,
        temperature: float = 0.07,
    ):
        super(ContrastiveVAE, self).__init__()
        self.latent_dim = latent_dim
        self.temperature = temperature

        # Transformer Encoder Backbone
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_dim = self.encoder.config.hidden_size

        # Variational Bottleneck Projection Heads
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder / Reconstruction Projection Head
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def mean_pooling(self, model_output, attention_mask):
        """Extracts mean-pooled sentence representation from token embeddings."""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids, attention_mask):
        """Passes text through transformer backbone and variational heads."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs, attention_mask)
        
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return pooled, mu, logvar

    def reparameterize(self, mu, logvar):
        """Applies the reparameterization trick: z = mu + eps * std."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic evaluation during inference

    def decode(self, z):
        """Reconstructs the hidden state representation from latent vector z."""
        return self.decoder(z)

    def forward(self, input_ids, attention_mask):
        """Single-sequence forward pass."""
        h, mu, logvar = self.encode(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar)
        h_recon = self.decode(z)
        return h, mu, logvar, z, h_recon

    def compute_loss(
        self,
        anc_ids,
        anc_mask,
        pos_ids,
        pos_mask,
        neg_ids,
        neg_mask,
        beta: float = 0.01,
        lambda_contrastive: float = 1.0,
    ):
        """Computes total loss: Reconstruction Loss + beta * KL Divergence + lambda * InfoNCE Contrastive Loss."""
        # 1. Forward pass for Anchor, Positive, Negative
        h_a, mu_a, logvar_a, z_a, h_recon_a = self.forward(anc_ids, anc_mask)
        h_p, mu_p, logvar_p, z_p, h_recon_p = self.forward(pos_ids, pos_mask)
        h_n, mu_n, logvar_n, z_n, h_recon_n = self.forward(neg_ids, neg_mask)

        # 2. Reconstruction Loss (MSE between raw hidden state and reconstructed hidden state)
        loss_recon = F.mse_loss(h_recon_a, h_a) + F.mse_loss(h_recon_p, h_p) + F.mse_loss(h_recon_n, h_n)

        # 3. KL Divergence Loss
        loss_kl = -0.5 * torch.mean(
            torch.sum(1 + logvar_a - mu_a.pow(2) - logvar_a.exp(), dim=1) +
            torch.sum(1 + logvar_p - mu_p.pow(2) - logvar_p.exp(), dim=1) +
            torch.sum(1 + logvar_n - mu_n.pow(2) - logvar_n.exp(), dim=1)
        )

        # 4. InfoNCE Contrastive Loss in Latent Space (z)
        z_a_norm = F.normalize(z_a, p=2, dim=1)
        z_p_norm = F.normalize(z_p, p=2, dim=1)
        z_n_norm = F.normalize(z_n, p=2, dim=1)

        sim_pos = torch.sum(z_a_norm * z_p_norm, dim=1) / self.temperature
        sim_neg = torch.sum(z_a_norm * z_n_norm, dim=1) / self.temperature

        # InfoNCE = -log ( exp(sim_pos) / (exp(sim_pos) + exp(sim_neg)) )
        logits = torch.stack([sim_pos, sim_neg], dim=1)  # Shape: [B, 2]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # Class 0 is positive
        loss_contrastive = F.cross_entropy(logits, labels)

        # Total Loss
        total_loss = loss_recon + (beta * loss_kl) + (lambda_contrastive * loss_contrastive)

        return {
            "total_loss": total_loss,
            "loss_recon": loss_recon.item(),
            "loss_kl": loss_kl.item(),
            "loss_contrastive": loss_contrastive.item(),
        }
