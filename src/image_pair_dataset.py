import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImagePairDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = data_dir
        # Use provided transform or default to ToTensor()
        self.transform = transform or transforms.ToTensor()

        # Load labels from JSON
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)

        self.pairs = []
        skipped = 0

        # Build list of valid image pairs
        for entry in self.labels:
            img1_path = os.path.join(self.data_dir, entry["frame_a"])
            img2_path = os.path.join(self.data_dir, entry["frame_b"])
            label = entry["label"]

            # Skip if either image is missing
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                print(f"⚠️  Skipping missing pair: {img1_path} or {img2_path}")
                skipped += 1
                continue

            self.pairs.append((img1_path, img2_path, label))

        # Summary
        print(f"✅ Loaded {len(self.pairs)} valid image pairs")
        if skipped > 0:
            print(f"⚠️  Skipped {skipped} pairs due to missing files")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]

        # Load and transform images
        img1 = self.transform(Image.open(img1_path).convert("RGB"))
        img2 = self.transform(Image.open(img2_path).convert("RGB"))

        return img1, img2, label
