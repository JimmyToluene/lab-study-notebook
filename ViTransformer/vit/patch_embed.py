import torch.nn as nn

class PatchEmbedding(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels):
    super().__init__()
    self.d_model = d_model # Dimensionality of Model
    self.img_size = img_size # Image Size
    self.patch_size = patch_size # Patch Size
    self.n_channels = n_channels # Number of Channels

    '''
    nn.Conv2d is an learnable parameter which register its weight and bias as 'nn.Parameter',
      weight: shape(d_model, n_channels, patch_size, patch_size)
        It flattening each patch into a vec of length `n_channels * patch_size^2` and multiplying by
        a learned (d_mode;. n_channels * patch_size ** 2) matrix
      bias: shape(d_model,)
      Parameter count: d_model * n_channels * patch_size² + d_model
    '''
    self.linear_project = nn.Conv2d(
      self.n_channels,
      self.d_model,
      kernel_size=self.patch_size,
      stride=self.patch_size)

  # B: Batch Size
  # C: Image Channels
  # H: Image Height
  # W: Image Width
  # P_row: Patch Row = H / patch_size
  # P_col: Patch Column = W / patch_size
  # P = Patch numbers = P_row * P_col

  def forward(self, x):
    x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_row, P_col)
    '''
    Flatten everything from dim 2 onward into one dim, 
    so two spatial dims (P_row, P_col) merge into a single P = P_row × P_col
    The patch at grid position (i, j) lands at sequence index i * P_col + j
    '''
    x = x.flatten(2) # (B, d_model, P_row, P_col) -> (B, d_model, P)

    x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)

    return x
