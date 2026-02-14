import torch
import torch.nn as nn
from .encoder import PhysicsEncoder
from .blocks import VSSBlock
from .decoder import CrossLatentDecoder

class PhyReMamba(nn.Module):
    def __init__(self, in_channels=5, base_dim=32):
        """
        PhyRe-Mamba: Physics-Guided Rectified Mamba Flow.
        Args:
            in_channels: 5 (RGB + Transmission + Depth)
            base_dim: C (Feature dimension)
        """
        super().__init__()
        
        # --- Encoder (CNN-based for local features) ---
        # Level 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.ReLU()
        )
        # Level 2
        self.down1 = nn.Conv2d(base_dim, base_dim*2, 4, 2, 1) # Downsample
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim*2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_dim*2, base_dim*2, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Level 3
        self.down2 = nn.Conv2d(base_dim*2, base_dim*4, 4, 2, 1) # Downsample
        self.enc3 = nn.Sequential(
             nn.Conv2d(base_dim*4, base_dim*4, 3, padding=1),
             nn.LeakyReLU(0.2)
        )
        
        # --- Bottleneck (Mamba - Global Context) ---
        # We put Mamba blocks here to capture global dependencies at low res
        self.bottleneck_mamba = nn.Sequential(
            VSSBlock(d_model=base_dim*4),
            VSSBlock(d_model=base_dim*4),
            VSSBlock(d_model=base_dim*4) # Depth = 3 blocks
        )
        
        # --- Decoder (Cross-Latent) ---
        # Up 1: 4C -> 2C (skip 2C) -> Output 2C
        self.dec1 = CrossLatentDecoder(in_channels=base_dim*4, skip_channels=base_dim*2, out_channels=base_dim*2)
        
        # Up 2: 2C -> C (skip C) -> Output C
        self.dec2 = CrossLatentDecoder(in_channels=base_dim*2, skip_channels=base_dim, out_channels=base_dim)
        
        # --- Velocity Head (Output) ---
        # Maps features back to 3 channels (RGB Velocity)
        self.head = nn.Conv2d(base_dim, 3, kernel_size=3, padding=1)
        
        # Initialize weights for stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 5, H, W)
        Returns:
            v: Velocity field (B, 3, H, W)
        """
        # Encoder
        e1 = self.enc1(x)       # (B, C, H, W)
        
        x_d1 = self.down1(e1)   # (B, 2C, H/2, W/2)
        e2 = self.enc2(x_d1)    # (B, 2C, H/2, W/2)
        
        x_d2 = self.down2(e2)   # (B, 4C, H/4, W/4)
        e3 = self.enc3(x_d2)    # (B, 4C, H/4, W/4)
        
        # Bottleneck (Mamba)
        b = self.bottleneck_mamba(e3) # (B, 4C, H/4, W/4)
        
        # Decoder
        d1 = self.dec1(b, e2)   # (B, 2C, H/2, W/2)
        d2 = self.dec2(d1, e1)  # (B, C, H, W)
        
        # Velocity Head
        v = self.head(d2)       # (B, 3, H, W)
        
        return v
