import torch.nn as nn
from .patch_embed import PatchEmbedding
from .pos_encoding import PositionalEncoding
from .transformer import TransformerEncoder

class ViTBackbone(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, use_pe=True, r_ffn=4, drop_path=0.0):
        super().__init__()
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img size dim must be divisible by patch_size dim"

        self.d_model = d_model # Dimensionality of model
        self.n_classes = n_classes # Number of classes
        self.img_size = img_size # Image size
        self.patch_size = patch_size # Patch size
        self.n_channels = n_channels # Number of channels
        self.n_heads = n_heads # Number of attention heads
        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1]) # Number of patches
        self.max_seq_length = 1 + self.n_patches # number of patches + [CLS] token
        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, n_channels)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length, use_pe=use_pe)

        # Linearly increasing stochastic-depth rate: 0 at the first block up to
        # `drop_path` at the last (the standard DeiT/timm schedule).
        dpr = [drop_path * i / max(1, n_layers - 1) for i in range(n_layers)]
        # We create number of n_layers of transformer stack, with increasing dropout rate
        self.transformer_encoders = nn.Sequential(
            *[TransformerEncoder(
                self.d_model, self.n_heads, r_ffn, dpr[i]
            ) for i in range(n_layers)])
        # Initialize LayerNorm variable for last layer.
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, images):
        x = self.patch_embedding(images) # (B, P, dim_model)
        x = self.positional_encoding(x)  # (B, max_token_length + 1, d_model)
        x = self.transformer_encoders(x)  # (B, max_token_length + 1, d_model)
        return self.ln_f(x[:, 0]) # Return normalized cls token (B, 1, d_model)

class ViTClassifier(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.d_model, n_classes)

    def forward(self, images):
        return self.head(self.backbone(images))