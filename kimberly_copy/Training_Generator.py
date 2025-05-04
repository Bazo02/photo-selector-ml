import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from burst_dataset import BurstDataset  # your dataset file
from generator import ResNetBurstGenerator  # the ResNet generator you defined

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔧 Using device:", device)

# 2. Dataset and Dataloader
dataset = BurstDataset("bursts_dir", burst_size=10)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# 3. Model, Optimiser, Loss
model = ResNetBurstGenerator(in_frames=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.L1Loss()

# 4. Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for burst, target in dataloader:
        burst = burst.to(device)    # [B, 10, 3, H, W]
        target = target.to(device)  # [B, 3, H, W]

        output = model(burst)       # [B, 3, H, W]
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"📘 Epoch {epoch+1}/{num_epochs} | L1 Loss: {avg_loss:.4f}")
