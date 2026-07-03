import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        # key, query, value projection/weights for all heads, but in a batch
        self.c_attn = nn.Linear(d_model, 3 * d_model)

        # output projection
        self.c_proj = nn.Linear(d_model, d_model)        # regularization
        self.n_head = n_heads
        self.d_model = d_model

    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1,2).contiguous().view(B,T,C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
