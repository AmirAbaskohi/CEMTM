import torch
import torch.nn as nn


class ImportanceNetwork(nn.Module):
    """
    Importance Network:
    - Input: Contextual token embeddings (B, N, D)
    - Output: μ and log σ² for each token’s importance (B, N)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mean_proj = nn.Linear(input_dim, 1)
        self.logvar_proj = nn.Linear(input_dim, 1)

    def forward(self, contextual_embeddings: torch.Tensor):
        """
        Args:
            contextual_embeddings: (B, N, D)

        Returns:
            mu:     (B, N)
            logvar: (B, N)
        """
        x = self.transformer(contextual_embeddings)  # (B, N, D)
        mu = self.mean_proj(x).squeeze(-1)           # (B, N)
        logvar = self.logvar_proj(x).squeeze(-1)     # (B, N)
        return mu, logvar
