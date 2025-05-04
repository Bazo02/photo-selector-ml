"""
Overview
- Backbone: ResNet18
- Input Fusion: Stack 10 RGB frames into a [30, H, W] tensor
- Encoder: ResNet-style convolutional blocks extract high-level features
- Bottleneck: Deeper fused representation
- Decoder: Upsample and refine output to final resolution
-> Output: 3 × H × W image

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class ResNetBurstGenerator(nn.Module):
    def __init__(self, in_frames=10, base_channels=64):
        super().__init__()
        in_channels = in_frames * 3  # 10 RGB frames → 30 channels

        # Initial convolution
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 2)
        )

        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels)
        )

        # Output layer
        self.output = nn.Conv2d(base_channels, 3, kernel_size=7, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, burst):
        # burst: [B, 10, 3, H, W] → [B, 30, H, W]
        B, T, C, H, W = burst.shape
        x = burst.view(B, T * C, H, W)

        x = self.head(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.tanh(self.output(x))  # Output in [-1, 1] range
        return x
