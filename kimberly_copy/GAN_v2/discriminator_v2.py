"""
PatchGAN based discriminator ment for GAN architecture
"""

import torch
import torch.nn as nn

class BurstDiscriminator(nn.Module):
    def __init__(self, burst_size=10, input_channels=3):
        super(BurstDiscriminator, self).__init__()
        total_input_channels = input_channels * (burst_size + 1)  # burst + 1 generated image

        self.model = nn.Sequential(
            nn.Conv2d(total_input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()  # or identity if using LSGAN
        )

    def forward(self, burst, generated):
        # burst: (B, N, 3, H, W) → (B, N*3, H, W)
        B, N, C, H, W = burst.shape
        burst = burst.view(B, N * C, H, W)

        # generated: (B, 3, H, W)
        x = torch.cat([burst, generated], dim=1)  # (B, (N+1)*3, H, W)
        return self.model(x)
