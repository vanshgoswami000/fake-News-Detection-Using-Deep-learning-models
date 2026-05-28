"""
models/rnn_model.py — Bidirectional LSTM for fake news detection
"""

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    Bidirectional LSTM classifier.
    Architecture: Embedding → BiLSTM → Global Max Pool → FC → Sigmoid
    Typical Accuracy: ~82%
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.dropout(self.embedding(x))            # (B, T, E)
        out, _ = self.lstm(emb)                          # (B, T, 2H)
        pooled = out.max(dim=1).values                   # (B, 2H)
        feat = torch.relu(self.fc(self.dropout(pooled))) # (B, 64)
        return self.out(feat)                            # (B, 2)


def build_rnn(vocab_size: int) -> RNNModel:
    return RNNModel(vocab_size=vocab_size)
