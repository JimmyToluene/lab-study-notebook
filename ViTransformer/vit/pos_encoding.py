import torch.nn as nn
import torch
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        # Create a learnable parameter named classification token([CLS])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # (1, 1, d_model)
        # Create a learnable positional encoding parameter (ViT paper)
        self.pe = nn.Parameter(torch.randn(1,max_seq_length, d_model))
        '''
        # Creating positional encodings
        pe = torch.zeros(max_seq_length, d_model) # (max_seq_length, d_model)
        # Sinusoidal Positional Embeddings
        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin( pos / ( 10000 ** ( i / d_model )) )
                else:
                    pe[pos][i] = np.cos( pos / 10000 ** ( (i - 1) / d_model) )
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_seq_length, d_model)
        '''

    def forward(self, x): # x: (B, n_patches, d_model)
        # Expand to have class token for every image in batch, 1 in expand() means "keep this dimension as-is."
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1) # (1, 1, d_model) -> (B, 1, d_model)

        # add class tokens to the beginning of each embd
        # Input: (B, 1, d_model) ⧺ (B, n_patches, d_model) along dim 1
        x = torch.cat((tokens_batch, x),dim=1) # -> (B, n_patches + 1, d_model)

        # Adding positional encoding to embeds
        # Input x: (B, n_patches + 1, d_model) — (128, 5, 9), self.pe: (1, max_seq_length, d_model)
        x = x + self.pe # -> (B, max_token_length + 1, d_model)

        return x