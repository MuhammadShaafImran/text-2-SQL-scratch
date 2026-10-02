import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Section 3.4: embeddings are multiplied by sqrt(d_model).
    One instance is shared by the encoder, the decoder and
    (optionally) the final output projection."""

    def __init__(self, vocab_size, d_model, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.scale = math.sqrt(d_model)

    def forward(self, ids):  # (B, L) -> (B, L, d_model)
        return self.emb(ids) * self.scale


class PositionalEncoding(nn.Module):
    """Section 3.5: fixed sinusoidal encodings.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))"""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)  # (L, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, d)

    def forward(self, x):  # x: (B, L, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


class InputLayer(nn.Module):
    """token ids -> scaled embedding + positional encoding -> dropout"""

    def __init__(self, token_emb, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.tok = token_emb
        self.pos = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, ids):
        return self.pos(self.tok(ids))
