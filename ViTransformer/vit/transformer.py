import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Learnable parameter count in MultiHeadAttention:
    c_attn: 3 * d_model· * model + 3 * d_model = 3 * d_model * (d_model + 1)
    c_proj: d_model * d_model + d_model     = d_model * (d_model + 1)
    Total:  4 * d_model * (d_model + 1)
    """
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

class FFN(nn.Module):
    """
    Learnable parameter count in FFN:
    c_fc: r * d_model * d_model + r * d_model
    c_proj: r * d_model * d_model + d_model
    Total:  2 * r * d_model * d_model + (r + 1)·d_model
    """
    def __init__(self, d_model, r_ffn):
        super().__init__()
        self.c_fc = nn.Linear(d_model, r_ffn * d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(r_ffn * d_model, d_model)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class DropPath(nn.Module):
    """
    Stochastic depth: it drops an entire residual branch for some samples in the batch.

    During training some samples get x = x + attn(x) and others just get x = x,
    The whole attention (or MLP) computation is zeroed out for that sample,
    so it passes through the skip connection only.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # one Bernoulli value per sample, broadcast over the other dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) # For input (B,P,d_model), create (B, 1, 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x / keep_prob * mask

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_ffn=4, drop_path=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Sub-Layer 1 Normalization
        self.ln_1 = nn.LayerNorm(d_model)
        # Multi-Head Attention
        self.attn = MultiHeadAttention(d_model, n_heads)
        # Sub-Layer 2 Normalization
        self.ln_2 = nn.LayerNorm(d_model)
        # FFN(MLP) layer
        self.ffn = FFN(d_model, r_ffn)
        # Stochastic depth on each residual branch
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        # Residual Connection After Sub-Layer 1
        x = x + self.drop_path(self.attn(self.ln_1(x)))
        # Residual Connection After Sub-Layer 2
        x = x + self.drop_path(self.ffn(self.ln_2(x)))
        return x