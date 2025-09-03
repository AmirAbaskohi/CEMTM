import torch
from collections import defaultdict
from typing import List, Dict


def extract_topic_word_distribution(
    token_topic_vectors: torch.Tensor,  # (N, K)
    token_importance: torch.Tensor,     # (N,)
    token_strings: List[str],           # token text for each position
    vocab: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Aggregate topic vectors across token positions to get per-word topic distributions.
    Args:
        token_topic_vectors: (N, K) soft topic assignment for each token
        token_importance: (N,) importance weight β_i for each token
        token_strings: List of length N, text tokens
        vocab: global vocabulary for filtering

    Returns:
        word_topic: dict of word → (K,) topic vector
    """
    word_topic_sum = defaultdict(lambda: torch.zeros(token_topic_vectors.size(1), device=token_topic_vectors.device))
    word_weight_sum = defaultdict(float)

    for i, word in enumerate(token_strings):
        if word not in vocab:
            continue
        weight = token_importance[i].item()
        word_topic_sum[word] += token_topic_vectors[i] * weight
        word_weight_sum[word] += weight

    topic_distributions = {}
    for word in word_topic_sum:
        if word_weight_sum[word] > 0:
            topic_distributions[word] = word_topic_sum[word] / word_weight_sum[word]

    return topic_distributions


def get_top_words_by_topic(
    word_topic_dict: Dict[str, torch.Tensor],
    num_topics: int,
    top_k: int = 10
) -> List[List[str]]:
    """
    Sort and return top words per topic.
    Args:
        word_topic_dict: {word: (K,) topic distribution}
        num_topics: total number of topics
        top_k: how many top words to return per topic

    Returns:
        top_words: List of K topics, each with top_k words
    """
    topic_to_words = [[] for _ in range(num_topics)]

    for word, vec in word_topic_dict.items():
        for k in range(num_topics):
            topic_to_words[k].append((word, vec[k].item()))

    # Sort and take top_k
    top_words = []
    for word_scores in topic_to_words:
        sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)
        top_words.append([w for w, _ in sorted_words[:top_k]])

    return top_words
