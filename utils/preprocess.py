"""
utils/preprocess.py — Text preprocessing for fake news detection
"""

import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Clean and normalize raw text."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # Remove URLs
    text = re.sub(r'<.*?>', '', text)                 # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)          # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()          # Collapse whitespace
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> list:
    """Tokenize text into word list."""
    tokens = word_tokenize(clean_text(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def build_vocab(texts: list, max_vocab: int = 10000) -> dict:
    """Build word-to-index vocabulary from list of texts."""
    from collections import Counter
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize(text))

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in Counter(all_tokens).most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def text_to_indices(text: str, vocab: dict, max_len: int = 200) -> list:
    """Convert text string to padded list of vocab indices."""
    tokens = tokenize(text)
    indices = [vocab.get(t, 1) for t in tokens]  # 1 = <UNK>
    # Pad or truncate to max_len
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices


def build_graph_features(text: str, vocab: dict) -> dict:
    """
    Build a simple word co-occurrence graph for GNN input.
    Returns node features and edge index.
    """
    tokens = tokenize(text)
    unique_tokens = list(set(tokens))
    node_map = {t: i for i, t in enumerate(unique_tokens)}

    # Node features: one-hot style using vocab index
    node_features = np.array(
        [[vocab.get(t, 1)] for t in unique_tokens], dtype=np.float32
    )

    # Edges: co-occurrence within a sliding window of size 3
    edges = []
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 3, len(tokens))):
            src = node_map[tokens[i]]
            dst = node_map[tokens[j]]
            edges.append([src, dst])
            edges.append([dst, src])  # undirected

    if not edges:
        edges = [[0, 0]]  # self-loop fallback

    edge_index = np.array(edges, dtype=np.int64).T  # shape (2, num_edges)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "num_nodes": len(unique_tokens),
    }
