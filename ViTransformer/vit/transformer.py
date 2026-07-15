import torch.nn as nn
import torch.nn.functional as F

'''
Parameter count in MultiHeadAttention:
c_attn: 3 * d_model· * model + 3 * d_model = 3 * d_model * (d_model + 1)
c_proj: d_model * d_model + d_model     = d_model * (d_model + 1)
Total:  4 * d_model * (d_model + 1)
'''
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
        # We adpot FlashAttention here.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1,2).contiguous().view(B,T,C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
'''
Learnable parameter count in FFN:
c_fc: r * d_model * d_model + r * d_model
c_proj: r * d_model * d_model + d_model
Total:  2 * r * d_model * d_model + (r + 1)·d_model
'''
class FFNN(nn.Module):
    def __init__(self, d_model, r_ffnn):
        super().__init__()
        self.c_fc = nn.Linear(d_model, r_ffnn * d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(r_ffnn * d_model, d_model)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_ffnn=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Sub-Layer 1 Normalization
        self.ln_1 = nn.LayerNorm(d_model)
        # Multi-Head Attention
        self.attn = MultiHeadAttention(d_model, n_heads)
        # Sub-Layer 2 Normalization
        self.ln_2 = nn.LayerNorm(d_model)
        # MLP layer
        self.ffn = FFNN(d_model, r_ffnn)

    def forward(self, x):
        # Residual Connection After Sub-Layer 1
        x = x + self.attn(self.ln_1(x))
        # Residual Connection After Sub-Layer 2
        x = x + self.ffn(self.ln_2(x))
        return x