import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from image_pair_dataset import ImagePairDataset

# Optional import of tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def train():
    # Use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Dataset paths (update as needed)
    data_dir = "/Users/alexbazo/Desktop/Project_4630/Synthetic_Data/Images"
    labels_file = "/Users/alexbazo/Desktop/Project_4630/Synthetic_Data/pairwise_labels.json"

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = ImagePairDataset(data_dir, labels_file, transform=transform)

    # DataLoader arguments
    dataloader_args = {
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 0,        # Set to 0 on macOS to avoid multiprocessing issues
        'pin_memory': use_cuda
    }
    loader = DataLoader(dataset, **dataloader_args)

    # Define the model (Siamese-style pairwise comparison)
    class PairwiseNet(nn.Module):
        def __init__(self):
            super().__init__()
            base = resnet18(weights=None)
            base.fc = nn.Identity()
            self.feature_extractor = base
            self.classifier = nn.Sequential(
                nn.Linear(512 * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

        def forward(self, img1, img2):
            f1 = self.feature_extractor(img1)
            f2 = self.feature_extractor(img2)
            x = torch.cat([f1, f2], dim=1)
            return self.classifier(x)

    # Initialize model, loss, optimizer
    model = PairwiseNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    EPOCHS = 10
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch_idx, (img1, img2, label) in enumerate(loop, start=1):
            img1, img2 = img1.to(device), img2.to(device)
            label = label.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_batch_loss = running_loss / batch_idx
            loop.set_postfix(loss=f"{avg_batch_loss:.4f}")

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "pairwise_model.pth")
    print("✅ Model saved as 'pairwise_model.pth'")


if __name__ == '__main__':
    train()
