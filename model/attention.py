import torch 
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.head = heads
        self.head_dim = self.d_model//self.head
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model,d_model)
        
    def forward(self, query, key_value=None, mask=None):
        if key_value is None:
            key_value = query
            
        batch_size, query_len, _ = query.shape
        _, key_len, _ = key_value.shape
        
        q = self.W_q(query)
        k = self.W_k(key_value)
        v = self.W_v(key_value)
        
        q = q.view(batch_size, query_len, self.head, self.head_dim)
        k = k.view(batch_size, key_len, self.head, self.head_dim)
        v = v.view(batch_size, key_len, self.head, self.head_dim)
        
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        
        score = q @ k.transpose(-2,-1)
        score = score/math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(score, dim=-1)
        output = attention @ v
        output = output.transpose(1, 2)
        output = output.contiguous().view(batch_size, query_len, self.d_model)
        output = self.W_o(output)
        
        return output, attention
    