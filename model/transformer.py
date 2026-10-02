import torch
import torch.nn as nn
from model.layer import EncoderLayer, DecoderLayer
from model.helper import create_padding_mask, create_target_mask
from starter.embeddings import InputLayer
from starter.tokenizer import PAD_ID

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, token_embedding, ffn_dim, max_len, dropout):
        super().__init__()
        self.input = InputLayer(token_embedding, d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, ffn_dim) for _ in range(num_layers)])
    
    def forward(self, id, source_mask=None):
        x = self.input(id)
        for layer in self.layers:
            x = layer(x, source_mask)
        return x
    
class Decoder(nn.Module):
    def __init__(self,num_layers,d_model,heads, token_embedding, ffn_dim, max_len, dropout):
        super().__init__()
        self.input = InputLayer(token_embedding, d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model,heads,ffn_dim) for _ in range(num_layers)])

    def forward(self, tgt_ids, encoder_output, target_mask=None, source_mask=None):
        x = self.input(tgt_ids)
        cross_attention_w = None
        for layer in self.layers:
            x, cross_attention_w = layer(x, encoder_output, target_mask=target_mask, source_mask=source_mask)
        return x, cross_attention_w

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, heads, token_embedding, ffn_dim, max_len, dropout, vocab_size):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, heads, token_embedding, ffn_dim, max_len, dropout)
        self.decoder = Decoder(num_layers, d_model, heads, token_embedding, ffn_dim, max_len, dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.output_projection.weight = token_embedding.emb.weight
    def forward(self, src_ids, tgt_ids, source_mask=None, target_mask=None):
        if source_mask is None:
            source_mask = create_padding_mask(src_ids,PAD_ID)
        if target_mask is None:
            target_mask = create_target_mask(tgt_ids,PAD_ID)
        encoder_output = self.encoder(src_ids,source_mask)
        decoder_output, cross_attention_weights = self.decoder(tgt_ids, encoder_output, target_mask=target_mask, source_mask=source_mask)
        value_score = self.output_projection(decoder_output)
        return value_score, cross_attention_weights