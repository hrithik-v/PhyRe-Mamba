import torch
import torch.nn as nn

class CrossLatentDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Decoder block that receives skip connections.
        Args:
            in_channels: Channels from the previous layer/bottleneck.
            skip_channels: Channels from the corresponding encoder layer.
            out_channels: Output channels for this block.
        """
        super().__init__()
        
        self.up = nn.PixelShuffle(upscale_factor=2)
        # PixelShuffle reduces channels by factor of 4 (r^2)
        # Input to conv will be (in_channels // 4) + skip_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d((in_channels // 4) + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        """
        Args:
            x: Input tensor (B, C_in, H, W)
            skip: Skip connection tensor (B, C_skip, 2H, 2W)
        """
        x_up = self.up(x)
        
        # Concatenate along channel dimension
        # Ensure sizes match (handle odd dimensions if any, though U-Net usually even)
        if x_up.size(2) != skip.size(2) or x_up.size(3) != skip.size(3):
             x_up = torch.nn.functional.interpolate(x_up, size=(skip.size(2), skip.size(3)), mode='nearest')
             
        out = torch.cat([x_up, skip], dim=1)
        out = self.conv(out)
        return out
