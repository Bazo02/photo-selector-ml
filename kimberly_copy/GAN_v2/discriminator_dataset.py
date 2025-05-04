"""
cCNN Discriminator's data pipeline
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DiscriminatorDataset(Dataset):
    def __init__(self, bursts_dir, real_images_dir, burst_size=10, image_size=(256, 256)):
        self.bursts_dir = bursts_dir
        self.real_images_dir = real_images_dir
        self.burst_size = burst_size
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        self.burst_folders = self._gather_burst_folders()
        self.real_images = self._gather_real_images()
        self._balance_lengths()

    def _gather_burst_folders(self):
        folders = []
        for root, dirs, _ in os.walk(self.bursts_dir):
            for d in dirs:
                path = os.path.join(root, d)
                if os.path.exists(os.path.join(path, "frame_00.jpg")):
                    folders.append(path)
        return folders

    def _gather_real_images(self):
        files = []
        for file in os.listdir(self.real_images_dir):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                files.append(os.path.join(self.real_images_dir, file))
        return files

    def _balance_lengths(self):
        if len(self.real_images) < len(self.burst_folders):
            times = (len(self.burst_folders) + len(self.real_images) - 1) // len(self.real_images)
            self.real_images = (self.real_images * times)[:len(self.burst_folders)]
        elif len(self.burst_folders) < len(self.real_images):
            times = (len(self.real_images) + len(self.burst_folders) - 1) // len(self.burst_folders)
            self.burst_folders = (self.burst_folders * times)[:len(self.real_images)]

    def __len__(self):
        return len(self.burst_folders)

    def __getitem__(self, idx):
        burst_path = self.burst_folders[idx]
        real_img_path = self.real_images[idx]

        burst_frames = []
        for i in range(self.burst_size):
            frame_file = os.path.join(burst_path, f"frame_{i:02d}.jpg")
            img = Image.open(frame_file).convert("RGB")
            burst_frames.append(self.transform(img))
        burst_tensor = torch.stack(burst_frames, dim=0)

        real_img = Image.open(real_img_path).convert("RGB")
        real_tensor = self.transform(real_img)

        return burst_tensor, real_tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    dataset = DiscriminatorDataset("bursts_dir", "real_images_dir", burst_size=10)
    print(f"Total samples: {len(dataset)}")

    burst, real = dataset[0]
    print("Burst shape:", burst.shape)
    print("Real image shape:", real.shape)

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    bursts, real_images = next(iter(loader))

    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5]*3),
        transforms.Normalize(mean=[-0.5]*3, std=[1.]*3)
    ])
    real_images = torch.stack([unnormalize(img) for img in real_images])

    grid = make_grid(real_images, nrow=4, padding=2)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title("Sample Real Images for Discriminator")
    plt.axis("off")
    plt.show()
