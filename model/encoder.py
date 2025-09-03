import torch
import torch.nn as nn
import torch.nn.functional as F


class TopicEncoder(nn.Module):
    """
    Projects token-level contextual embeddings into a topic distribution.
    Given contextual embedding h_i ∈ ℝ^D, output t_i = softmax(W_t h_i) ∈ ℝ^K.
    """

    def __init__(self, input_dim: int, num_topics: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, num_topics)

    def forward(self, contextual_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            contextual_embeddings: (batch_size, seq_len, input_dim)

        Returns:
            topic_distributions: (batch_size, seq_len, num_topics)
        """
        logits = self.proj(contextual_embeddings)
        topic_distributions = F.softmax(logits, dim=-1)
        return topic_distributions
