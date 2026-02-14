import torch
import torch.nn as nn
from mamba_ssm import Mamba

class VSSBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        """
        Visual State Space Block.
        Wraps the Mamba block to handle 2D image data by flattening and reshaping.
        Args:
            d_model: Input dimension C.
            d_state: SSM state dimension N.
            d_conv: Local convolution width.
            expand: Expansion factor E.
        """
        super().__init__()
        self.d_model = d_model
        
        # Mamba Inner Block
        # Ensure we use a version compatible with available hardware
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.norm = nn.LayerNorm(d_model)
        
        # Linear projection for mixing if needed, but Mamba handles it internally usually.
        # We add a skip connection structure similar to Transformers.

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            out: Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Norm and rearrange for Mamba (B, L, C)
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        x_norm = self.norm(x_flat)
        
        # Mamba forward pass
        # Standard Mamba is causal and 1D. For images, we can use it as is for global context
        # or implement SS2D (multi-scan). For simplicity in this base version, 
        # we treat the flattened image as a sequence. 
        # Note: A more advanced version would flip sequences (Bi-directional).
        
        # Forward scan
        out_fwd = self.mamba(x_norm)
        
        # Backward scan (simple bi-directional emulation)
        x_rev = torch.flip(x_norm, [1])
        out_rev = self.mamba(x_rev)
        out_rev = torch.flip(out_rev, [1])
        
        # Combine (average)
        out = (out_fwd + out_rev) / 2
        
        # Residual connection
        out = out + x_flat
        
        # Reshape back: (B, L, C) -> (B, C, L) -> (B, C, H, W)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return out
