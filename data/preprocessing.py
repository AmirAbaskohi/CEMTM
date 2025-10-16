import re
import string
import nltk
import torch
from PIL import Image
from typing import List
from torchvision import transforms

# Make sure required NLTK resources are downloaded
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

# === Text Preprocessing ===
def clean_text(text: str) -> str:
    """
    Basic NLP cleaning:
    - Lowercase
    - Remove punctuation
    - Remove stopwords
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)  # remove HTML tags
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


# === Vocabulary Construction ===
def build_vocab(corpus: List[str], max_vocab_size: int = 30000, min_freq: int = 5) -> List[str]:
    """
    Build vocabulary from a list of cleaned text strings.
    Returns most frequent tokens (excluding stopwords and rare words).
    
    Args:
        corpus: List of cleaned text strings
        max_vocab_size: Maximum vocabulary size
        min_freq: Minimum frequency threshold (default: 5)
    
    Returns:
        List of vocabulary words
    """
    from collections import Counter

    token_counter = Counter()
    for doc in corpus:
        tokens = doc.split()
        token_counter.update(tokens)

    # Filter by minimum frequency and take most common
    vocab = [word for word, count in token_counter.most_common() 
             if count >= min_freq][:max_vocab_size]
    
    print(f"Built vocabulary: {len(vocab)} words (min_freq={min_freq}, max_size={max_vocab_size})")
    return vocab


# === Image Preprocessing ===
def get_image_transform(image_size: int = 224):
    """
    Return a standard image transform for VLMs.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def preprocess_image(image: Image.Image, transform=None) -> torch.Tensor:
    """
    Apply standard image preprocessing. Returns a tensor.
    """
    if transform is None:
        transform = get_image_transform()
    return transform(image)
