import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predicted_embedding, target_embedding):
        return self.mse(predicted_embedding, target_embedding)


class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, importance_weights):
        """
        Entropy of the importance distribution.
        importance_weights: (batch_size, seq_len)
        """
        # Add epsilon for numerical stability
        eps = 1e-8
        entropy = importance_weights * torch.log(importance_weights + eps)
        return entropy.sum(dim=1).mean()  # mean over batch


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        """
        KL divergence between predicted importance distribution q(α) ~ N(μ, σ²)
        and prior p(α) ~ N(0, I)
        """
        # KL(q || p) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()
