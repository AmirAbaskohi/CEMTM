import argparse
import yaml
import torch
from data.dataset import get_dataset
from model.cemtm import CEMTM
from evaluation.topic_extraction import extract_topic_word_distribution, get_top_words_by_topic
from evaluation.metrics import (
    topic_diversity, inverse_rbo, compute_npmi,
    compute_we_score, compute_llm_score
)

from gensim.models import KeyedVectors  # For word embeddings like GloVe or fastText
from data.preprocessing import clean_text


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def evaluate_topics(config):
    # === Load config, device, model ===
    device = config["training"]["device"]
    model = CEMTM(
        vlm_model_name=config["vlm2vec"]["model_name"],
        input_dim=config["model"]["d_model"],
        num_topics=config["model"]["n_topics"],
        transformer_layers=config["model"]["transformer_layers"],
        transformer_heads=config["model"]["transformer_heads"],
        dropout=config["model"]["dropout"],
        freeze_vlm=True,  # no training here
    ).to(device)

    ckpt_path = f"{config['output']['save_dir']}/{config['output']['checkpoint_name']}"
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    # === Load data ===
    dataset = get_dataset(config["data"]["name"], config["data"]["dataset_path"])
    vocab = set(config["vocab"]) if "vocab" in config else None
    tokenized_corpus = []

    all_token_topics = []
    all_token_importances = []
    all_tokens = []

    print("Extracting token-topic vectors...")

    for sample in dataset:
        text = clean_text(sample["text"])
        images = sample["images"]

        output = model(text, images)

        beta = output["beta"][0]                         # (N,)
        token_topic = output["token_topic_vectors"][0]   # (N, K)
        tokens = text.split()

        all_token_topics.append(token_topic)
        all_token_importances.append(beta)
        all_tokens.append(tokens)
        tokenized_corpus.append(tokens)

    # === Flatten token-level outputs ===
    flat_topic_vectors = torch.cat(all_token_topics, dim=0)
    flat_importance = torch.cat(all_token_importances, dim=0)
    flat_tokens = [tok for doc in all_tokens for tok in doc]

    # === Extract topic-word vectors ===
    word_topic = extract_topic_word_distribution(
        flat_topic_vectors, flat_importance, flat_tokens, vocab=flat_tokens
    )

    top_words = get_top_words_by_topic(word_topic, num_topics=config["model"]["n_topics"], top_k=10)

    # === Metrics ===
    print("Evaluating metrics...")
    metrics = {
        "Topic Diversity": topic_diversity(top_words),
        "I-RBO": inverse_rbo(top_words),
        "NPMI": compute_npmi(top_words, tokenized_corpus),
        "WE": compute_we_score(top_words, embedding_model=KeyedVectors.load_word2vec_format(config["data"]["embedding_path"], binary=False)),
        "LLM Score": compute_llm_score(top_words),
    }

    print("=== Topic Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_topics(config)