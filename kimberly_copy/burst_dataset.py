import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BurstDataset(Dataset):
    def __init__(self, bursts_dir, burst_size=10, transform=None, image_size=(256, 256)):
        """
        Args:
            bursts_dir (str): Directory containing folders with burst images.
            burst_size (int): Number of frames in each burst.
            transform (callable): Transform to apply to images.
            image_size (tuple): Resize dimensions for all images.
        """
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
        burst_folders = []
        print(f"🔍 Scanning: {self.bursts_dir}")
        for root, dirs, files in os.walk(self.bursts_dir):
            for d in dirs:
                full_path = os.path.join(root, d)
                frame_00 = os.path.join(full_path, "frame_00.jpg")
                if os.path.exists(frame_00):
                    print(f"✅ Found burst folder: {full_path}")
                    burst_folders.append(full_path)
                else:
                    print(f"❌ Skipping: {full_path} (frame_00.jpg not found)")
        print(f"📦 Total valid bursts: {len(burst_folders)}")
        return burst_folders

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
        gt_tensor = burst_tensor[0]  # Use frame_00 as target

        return burst_tensor, gt_tensor


dataset = BurstDataset("bursts_dir", burst_size=10)
burst, target = dataset[0]
print("Burst shape:", burst.shape)
print("Target shape:", target.shape)

print("Burst value range:", burst.min().item(), "to", burst.max().item())
print("Target value range:", target.min().item(), "to", target.max().item())
