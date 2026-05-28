"""
data/dataset.py — Dataset loading and preparation
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Generate sample CSV if it doesn't exist ──────────────────────────────────
SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "sample_data.csv")

FAKE_HEADLINES = [
    "Scientists discover chocolate cures all diseases instantly",
    "Government secretly controls the weather using 5G towers",
    "Aliens land in New York; media blackout ordered by president",
    "Drinking bleach proven to boost immunity, experts say",
    "Moon is actually a hologram created by illuminati",
    "Vaccines contain microchips to track your location",
    "Bill Gates admits to funding global pandemic for profit",
    "Earth is flat; NASA finally confesses the truth",
    "CIA releases mind-control chemicals through airplane contrails",
    "New law requires citizens to surrender pets to government",
]

REAL_HEADLINES = [
    "Federal Reserve raises interest rates by 0.25 percent amid inflation concerns",
    "Study shows regular exercise reduces risk of heart disease by 30 percent",
    "Senate passes bipartisan infrastructure bill after months of negotiations",
    "WHO reports global flu vaccination rates at record high this year",
    "Tech companies face new antitrust scrutiny from European regulators",
    "Climate scientists warn Arctic ice melt accelerating faster than predicted",
    "SpaceX successfully launches 60 Starlink satellites into low Earth orbit",
    "UN peacekeepers deployed to conflict zone as ceasefire holds fragile",
    "Researchers develop new battery technology with twice the energy density",
    "Supreme Court rules on landmark privacy case affecting millions",
]

def create_sample_data():
    rows = []
    for h in FAKE_HEADLINES:
        rows.append({"text": h, "label": 1})  # 1 = FAKE
    for h in REAL_HEADLINES:
        rows.append({"text": h, "label": 0})  # 0 = REAL
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(SAMPLE_CSV, index=False)
    return df


# ── PyTorch Datasets ──────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Dataset for RNN and CNN models (index sequences)."""

    def __init__(self, texts, labels, vocab, max_len=200):
        from utils.preprocess import text_to_indices
        self.data = [
            torch.tensor(text_to_indices(t, vocab, max_len), dtype=torch.long)
            for t in texts
        ]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class BERTDataset(Dataset):
    """Dataset for BERT model (tokenizer-based)."""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}, self.labels[idx]


class GNNDataset(Dataset):
    """Dataset for GNN model (graph-structured inputs)."""

    def __init__(self, texts, labels, vocab):
        from utils.preprocess import build_graph_features
        self.graphs = [build_graph_features(t, vocab) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def load_data(csv_path=None):
    """Load CSV and return (texts, labels) arrays."""
    path = csv_path or SAMPLE_CSV
    if not os.path.exists(path):
        df = create_sample_data()
    else:
        df = pd.read_csv(path)
    return df["text"].tolist(), df["label"].tolist()
