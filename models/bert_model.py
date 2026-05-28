"""
models/bert_model.py — Fine-tuned BERT for fake news detection
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BERTModel(nn.Module):
    """
    BERT-base fine-tuned for binary classification.
    Uses [CLS] token representation → dropout → linear classifier.
    Typical Accuracy: ~91%
    """

    def __init__(self, bert_name: str = "bert-base-uncased", dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        hidden = self.bert.config.hidden_size         # 768 for bert-base
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_out = outputs.pooler_output               # (B, 768)
        return self.fc(self.dropout(cls_out))         # (B, 2)


def build_bert(bert_name: str = "bert-base-uncased") -> tuple:
    """Return (model, tokenizer) tuple."""
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    model = BERTModel(bert_name)
    return model, tokenizer
