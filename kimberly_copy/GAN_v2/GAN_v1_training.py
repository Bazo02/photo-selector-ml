"""
GAN v1
- Only train the generator
- Use L1 loss between the output and frame_00.jpg
- Will not include adversarial components (no discriminator)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.utils import make_grid

from burst_dataset import BurstOnlyDataset  # Dataset that returns burst and target
from generator import ResNetBurstGenerator  # Generator model (v1/v2 compatible)

# ------------------------
# 1. Setup
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔧 Using device:", device)

# ------------------------
# 2. Dataset and Dataloader
# ------------------------
dataset = BurstOnlyDataset("bursts_dir", burst_size=10)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# ------------------------
# 3. Model, Optimiser, Loss
# ------------------------
model = ResNetBurstGenerator(in_frames=10).to(device)
optimiser = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.L1Loss()

# ------------------------
# 4. Training Loop
# ------------------------
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for burst in dataloader:
        burst = burst.to(device)
        target = burst[:, 0]  # frame_00 as target

        output = model(burst)
        loss = criterion(output, target)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"\n📘 Epoch {epoch+1}/{num_epochs} | L1 Loss: {avg_loss:.4f}")



# ------------------------
# 5. Save the Generator
# ------------------------
os.makedirs("trained_models_dir", exist_ok=True)
torch.save(model.state_dict(), "trained_models_dir/final_GAN_v1_generator.pth")
