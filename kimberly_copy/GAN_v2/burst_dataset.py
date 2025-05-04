"""
data pipeline for the generator
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BurstOnlyDataset(Dataset):
    def __init__(self, bursts_dir, burst_size=10, transform=None, image_size=(256, 256)):
        self.bursts_dir = bursts_dir
        self.burst_size = burst_size
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        self.samples = self._gather_burst_folders()

    def _gather_burst_folders(self):
        folders = []
        for root, dirs, _ in os.walk(self.bursts_dir):
            for d in dirs:
                path = os.path.join(root, d)
                if os.path.exists(os.path.join(path, "frame_00.jpg")):
                    folders.append(path)
        return folders

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        burst_path = self.samples[idx]
        burst_frames = []

        for i in range(self.burst_size):
            frame_file = os.path.join(burst_path, f"frame_{i:02d}.jpg")
            img = Image.open(frame_file).convert("RGB").resize(self.image_size)
            burst_frames.append(self.transform(img))

        burst_tensor = torch.stack(burst_frames, dim=0)  # Shape: [10, 3, H, W]
        return burst_tensor

# --- Test block ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    dataset = BurstOnlyDataset("bursts_dir", burst_size=10)
    burst = dataset[0]

    print("Burst shape:", burst.shape)
    print("Value range:", burst.min().item(), "to", burst.max().item())

    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5]*3),
        transforms.Normalize(mean=[-0.5]*3, std=[1.]*3)
    ])
    burst_images = [unnormalize(frame) for frame in burst]

    grid = make_grid(burst_images, nrow=len(burst), padding=2)
    plt.figure(figsize=(20, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title("Sample Burst Frames")
    plt.axis('off')
    plt.show()


# Instantiate and test
dataset = BurstOnlyDataset("bursts_dir", burst_size=10)
burst = dataset[0]  # Only one output now

print("Burst shape:", burst.shape)  # Expected: [10, 3, 256, 256]
print("Burst value range:", burst.min().item(), "to", burst.max().item())
