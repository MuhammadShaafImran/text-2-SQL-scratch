from typing import Any

import torch
import torch.nn as nn
from model.attention import Attention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.attention = Attention(d_model, heads)
        self.normalization_attention = nn.LayerNorm(d_model)
        self.normalization_ffn =  nn.LayerNorm(d_model)
        self.ffn = nn.Sequential( nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x, source_mask=None):
        multihead_attention, _ = self.attention(x, mask=source_mask)
        norm1 = self.normalization_attention(multihead_attention + x)
        feed_forward_network = self.ffn(norm1)
        ffn_attention = feed_forward_network + norm1
        norm2 = self.normalization_ffn(ffn_attention)
        return norm2
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.self_attention = Attention(d_model, heads)
        self.cross_attention = Attention(d_model,heads)
        self.normalization_self_attention = nn.LayerNorm(d_model)
        self.normalization_cross_attention = nn.LayerNorm(d_model)
        self.normalization_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),nn.Linear(d_ff, d_model))

    def forward( self, x, encoder_output, target_mask=None, source_mask=None):
        self_attention_output, _ = self.self_attention(x, mask=target_mask)
        norm1 = self.normalization_self_attention(x + self_attention_output)
        cross_attention_output, cross_attention_weights = self.cross_attention(norm1, encoder_output, mask=source_mask)
        norm2 = self.normalization_cross_attention(norm1 + cross_attention_output)
        ffn_output = self.ffn(norm2)
        norm3 = self.normalization_ffn(norm2 + ffn_output)
        return norm3, cross_attention_weights