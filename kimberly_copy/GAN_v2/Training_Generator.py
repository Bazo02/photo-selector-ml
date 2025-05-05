"""
Used for pre-training the generator
- trains the generator in isolation for debugging purposes
- pre-trains generator_v2
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T

from burst_dataset import BurstOnlyDataset  # your dataset file
from generator import ResNetBurstGenerator  # the ResNet generator you defined

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔧 Using device:", device)

# 2. Dataset and Dataloader
dataset = BurstOnlyDataset("bursts_dir", burst_size=10)
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


# 5. Visualisation of the generated outputs

denorm = T.Compose([
    T.Normalize(mean=[0., 0., 0.], std=[1/0.5]*3),
    T.Normalize(mean=[-0.5]*3, std=[1.]*3),
])

def show_image(tensor, title):
    image = denorm(tensor).clamp(0, 1)  # undo normalisation
    image = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

model.eval()
burst, target = dataset[0]
burst = burst.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(burst)[0]  # remove batch dimension

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
show_image(target, "Target (frame_00)")
plt.subplot(1, 2, 2)
show_image(output, "Generated Output")
plt.tight_layout()
plt.show()