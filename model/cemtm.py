import torch
import torch.nn as nn
from model.vlm2vec import VLM2Vec
from model.encoder import TopicEncoder
from model.importance_net import ImportanceNetwork


class CEMTM(nn.Module):
    def __init__(
        self,
        vlm_model_name: str,
        input_dim: int,
        num_topics: int,
        transformer_layers: int,
        transformer_heads: int,
        dropout: float = 0.1,
        freeze_vlm: bool = False
    ):
        super().__init__()
        self.vlm2vec = VLM2Vec(model_name=vlm_model_name, freeze=freeze_vlm)
        
        # Verify that input_dim matches the VLM's hidden dimension
        actual_dim = self.vlm2vec.hidden_dim
        if input_dim != actual_dim:
            print(f"⚠️  WARNING: Config specifies input_dim={input_dim}, but {vlm_model_name} has hidden_dim={actual_dim}")
            print(f"✓ Using actual dimension: {actual_dim}")
            input_dim = actual_dim
        
        self.input_dim = input_dim
        self.topic_encoder = TopicEncoder(input_dim=input_dim, num_topics=num_topics)
        self.importance_net = ImportanceNetwork(
            input_dim=input_dim,
            hidden_dim=input_dim * 2,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=dropout
        )

        # Decoder to map topic vector back to document embedding
        self.decoder = nn.Linear(num_topics, input_dim)

    def forward(self, text: str, images):
        """
        Forward pass for a single document
        Args:
            text (str)
            images (List[PIL.Image])

        Returns:
            dict with:
                - e_d: reference document embedding
                - e_d_prime: reconstructed document embedding
                - beta: importance weights
                - mu: token-level mean of importance
                - logvar: token-level log-variance
                - topic_d: document-level topic vector
        """
        # === Step 1: Get contextual embeddings ===
        contextual_embeddings, e_d = self.vlm2vec(text, images)  # (N, D), (D,)
        H = contextual_embeddings.unsqueeze(0)  # (1, N, D)

        # === Step 2: Topic vectors ===
        topic_vectors = self.topic_encoder(H)  # (1, N, K)

        # === Step 3: Importance scores ===
        mu, logvar = self.importance_net(H)  # (1, N), (1, N)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        alpha = mu + eps * std  # (1, N)
        beta = torch.softmax(alpha, dim=-1)  # (1, N)

        # === Step 4: Document topic vector ===
        topic_d = torch.sum(beta.unsqueeze(-1) * topic_vectors, dim=1)  # (1, K)
        topic_d = torch.softmax(topic_d, dim=-1)  # normalized topic vector

        # === Step 5: Decode back to document embedding ===
        e_d_prime = self.decoder(topic_d)  # (1, D)

        return {
            "e_d": e_d.unsqueeze(0),                  # (1, D)
            "e_d_prime": e_d_prime,                   # (1, D)
            "mu": mu,                                 # (1, N)
            "logvar": logvar,                         # (1, N)
            "beta": beta,                             # (1, N)
            "topic_d": topic_d,                       # (1, K)
            "token_topic_vectors": topic_vectors,     # (1, N, K)
            "contextual_embeddings": H                # (1, N, D)
        }
