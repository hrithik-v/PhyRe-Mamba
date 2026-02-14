import torch
import torch.nn as nn

class PhysicsEncoder(nn.Module):
    def __init__(self, in_channels=5, out_channels=32):
        """
        Encodes the 5-channel input (RGB + Transmission + Depth) into a latent feature space.
        Args:
            in_channels: Number of input channels (5 for PhyRe-Mamba).
            out_channels: Base channel dimension C.
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.encoder(x)
