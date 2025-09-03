import numpy as np
import time
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter
from itertools import combinations
from evaluation.llm_api import rate_topic_with_gpt


# === Topic Diversity (TD) ===
def topic_diversity(top_words: list[list[str]], top_k: int = 10):
    """
    Proportion of unique words among all top-k topic words.
    """
    all_words = [word for topic in top_words for word in topic[:top_k]]
    unique_words = set(all_words)
    return len(unique_words) / len(all_words)


# === Inverse Rank-Biased Overlap (I-RBO) ===
def inverse_rbo(top_words: list[list[str]], p: float = 0.9):
    """
    Compute 1 - average RBO across all topic pairs (i.e., I-RBO).
    """
    def rbo(list1, list2, p=0.9):
        overlap = 0.0
        rbo_score = 0.0
        seen1, seen2 = set(), set()
        for d in range(1, min(len(list1), len(list2)) + 1):
            seen1.add(list1[d - 1])
            seen2.add(list2[d - 1])
            overlap = len(seen1 & seen2)
            rbo_score += (overlap / d) * (p ** (d - 1))
        return (1 - p) * rbo_score

    topic_pairs = list(combinations(top_words, 2))
    scores = [rbo(t1, t2, p) for t1, t2 in topic_pairs]
    return 1.0 - np.mean(scores) if scores else 0.0


# === Normalized PMI (NPMI) ===
def compute_npmi(top_words: list[list[str]], tokenized_corpus: list[list[str]], top_k: int = 10):
    """
    Compute NPMI over all topic word pairs based on the corpus.
    """
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora.dictionary import Dictionary

    dictionary = Dictionary(tokenized_corpus)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_corpus]

    cm = CoherenceModel(
        topics=[topic[:top_k] for topic in top_words],
        texts=tokenized_corpus,
        dictionary=dictionary,
        coherence='c_npmi'
    )
    return cm.get_coherence()


# === Word Embedding (WE) Score ===
def compute_we_score(top_words: list[list[str]], embedding_model, top_k: int = 10):
    """
    Compute average cosine similarity of all word pairs in a topic.
    """
    def pairwise_cosine(words):
        vectors = [embedding_model[word] for word in words if word in embedding_model]
        if len(vectors) < 2:
            return 0.0
        sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-8)
                sims.append(sim)
        return np.mean(sims) if sims else 0.0

    topic_sims = [pairwise_cosine(topic[:top_k]) for topic in top_words]
    return np.mean(topic_sims)


# === LLM Score ===
def compute_llm_score(top_words: list[list[str]], top_k: int = 10):
    scores = []
    for topic in top_words:
        score = rate_topic_with_gpt(topic[:top_k])
        scores.append(score)
        time.sleep(1.5)  # prevent rate limit
    return sum(scores) / len(scores)


# === Clustering Metrics ===
def purity(y_true, y_pred):
    """
    Compute cluster purity.
    """
    contingency_matrix = {}
    for pred, true in zip(y_pred, y_true):
        contingency_matrix.setdefault(pred, Counter())[true] += 1
    total_correct = sum(max(label_counts.values()) for label_counts in contingency_matrix.values())
    return total_correct / len(y_true)


def compute_clustering_metrics(y_true, y_pred):
    return {
        "Purity": purity(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred)
    }
